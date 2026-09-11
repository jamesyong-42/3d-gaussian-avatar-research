namespace GsAvatar;

public sealed class GsAvatarSnapshot
{
    public byte Protocol { get; set; } = Constants.ProtocolU8;
    public byte StreamLod { get; set; } = Constants.LodMedium;
    public uint Seq { get; set; }
    public uint CaptureTick { get; set; }
    public ushort Flags { get; set; } = Constants.HasRoot | Constants.HasFace | Constants.HasHands | Constants.HasExpr;
    public double[] RootPos { get; set; } = new double[3];
    public double[] RootRot { get; set; } = { 0, 0, 0, 1 };
    public double[] Body { get; set; } = new double[Constants.BodyDim];
    public double[] LeftHand { get; set; } = new double[Constants.HandDim];
    public double[] RightHand { get; set; } = new double[Constants.HandDim];
    public double[] Expr { get; set; } = new double[Constants.ExprFull];

    public SmplxPose ToPose()
    {
        var expr = new double[Constants.ExprFull];
        Array.Copy(Expr, expr, Math.Min(Expr.Length, expr.Length));
        return new SmplxPose
        {
            Body = (double[])Body.Clone(),
            LeftHand = (double[])LeftHand.Clone(),
            RightHand = (double[])RightHand.Clone(),
            Expr = expr,
            RootPos = (double[])RootPos.Clone(),
            RootRot = (double[])RootRot.Clone(),
            CaptureTick = CaptureTick,
            Teleport = (Flags & Constants.Teleport) != 0,
        };
    }

    public static GsAvatarSnapshot FromPose(SmplxPose pose, byte lod, uint seq)
    {
        ushort flags = Constants.HasRoot;
        if (lod >= Constants.LodMedium)
            flags |= Constants.HasFace | Constants.HasHands | Constants.HasExpr;
        if (pose.Teleport)
            flags |= Constants.Teleport;
        return new GsAvatarSnapshot
        {
            StreamLod = lod,
            Seq = seq,
            CaptureTick = pose.CaptureTick,
            Flags = flags,
            RootPos = (double[])pose.RootPos.Clone(),
            RootRot = (double[])pose.RootRot.Clone(),
            Body = (double[])pose.Body.Clone(),
            LeftHand = (double[])pose.LeftHand.Clone(),
            RightHand = (double[])pose.RightHand.Clone(),
            Expr = (double[])pose.Expr.Clone(),
        };
    }
}

public static class StreamCodec
{
    public static byte[] Record(SmplxPose pose, byte lod, uint seq) =>
        Encode(GsAvatarSnapshot.FromPose(pose, lod, seq));

    public static SmplxPose Apply(byte[] data) => Decode(data).ToPose();

    public static byte[] Encode(GsAvatarSnapshot snap)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write(Constants.ProtocolU8);
        w.Write(snap.StreamLod);
        w.Write(snap.Seq);
        w.Write(snap.CaptureTick);
        w.Write(snap.Flags);
        if ((snap.Flags & Constants.HasRoot) != 0)
        {
            WriteF32(w, snap.RootPos, 3);
            WriteF32(w, snap.RootRot, 4);
        }

        var body = Pad(snap.Body, Constants.BodyDim);
        var lh = Pad(snap.LeftHand, Constants.HandDim);
        var rh = Pad(snap.RightHand, Constants.HandDim);
        var expr = snap.Expr ?? Array.Empty<double>();

        if (snap.StreamLod == Constants.LodFull)
        {
            WriteF16(w, body);
            if ((snap.Flags & Constants.HasHands) != 0)
            {
                WriteF16(w, lh);
                WriteF16(w, rh);
            }
            if ((snap.Flags & Constants.HasExpr) != 0)
                WriteF16(w, Pad(expr, Constants.ExprFull));
            w.Flush();
            return ms.ToArray();
        }

        if (snap.StreamLod == Constants.LodLow)
        {
            var low = new double[Constants.LowBodyIndices.Length];
            for (int i = 0; i < low.Length; i++)
                low[i] = body[Constants.LowBodyIndices[i]];
            WriteI16(w, QuantizeAa(low));
            w.Flush();
            return ms.ToArray();
        }

        WriteI16(w, QuantizeAa(body));
        if ((snap.Flags & Constants.HasHands) != 0)
        {
            WriteI16(w, QuantizeAa(lh));
            WriteI16(w, QuantizeAa(rh));
        }
        if ((snap.Flags & Constants.HasExpr) != 0)
        {
            int n = snap.StreamLod == Constants.LodMedium ? Constants.ExprMedium : Constants.ExprFull;
            WriteI16(w, QuantizeExpr(Pad(expr, n)));
        }
        w.Flush();
        return ms.ToArray();
    }

    public static GsAvatarSnapshot Decode(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var r = new BinaryReader(ms);
        var snap = new GsAvatarSnapshot
        {
            Protocol = r.ReadByte(),
            StreamLod = r.ReadByte(),
            Seq = r.ReadUInt32(),
            CaptureTick = r.ReadUInt32(),
            Flags = r.ReadUInt16(),
        };
        if (snap.Protocol != Constants.ProtocolU8)
            throw new InvalidDataException($"unsupported protocol {snap.Protocol}");
        if ((snap.Flags & Constants.HasRoot) != 0)
        {
            snap.RootPos = ReadF32(r, 3);
            snap.RootRot = ReadF32(r, 4);
        }
        if (snap.StreamLod == Constants.LodFull)
        {
            snap.Body = ReadF16(r, Constants.BodyDim);
            if ((snap.Flags & Constants.HasHands) != 0)
            {
                snap.LeftHand = ReadF16(r, Constants.HandDim);
                snap.RightHand = ReadF16(r, Constants.HandDim);
            }
            if ((snap.Flags & Constants.HasExpr) != 0)
                snap.Expr = ReadF16(r, Constants.ExprFull);
            return snap;
        }
        if (snap.StreamLod == Constants.LodLow)
        {
            var q = ReadI16(r, Constants.LowBodyIndices.Length);
            var body = new double[Constants.BodyDim];
            var aa = DequantizeAa(q);
            for (int i = 0; i < Constants.LowBodyIndices.Length; i++)
                body[Constants.LowBodyIndices[i]] = aa[i];
            snap.Body = body;
            return snap;
        }
        snap.Body = DequantizeAa(ReadI16(r, Constants.BodyDim));
        if ((snap.Flags & Constants.HasHands) != 0)
        {
            snap.LeftHand = DequantizeAa(ReadI16(r, Constants.HandDim));
            snap.RightHand = DequantizeAa(ReadI16(r, Constants.HandDim));
        }
        if ((snap.Flags & Constants.HasExpr) != 0)
        {
            int n = snap.StreamLod == Constants.LodMedium ? Constants.ExprMedium : Constants.ExprFull;
            var e = DequantizeExpr(ReadI16(r, n));
            var expr = new double[Constants.ExprFull];
            Array.Copy(e, expr, n);
            snap.Expr = expr;
        }
        return snap;
    }

    static double[] Pad(double[] src, int n)
    {
        var d = new double[n];
        if (src != null)
            Array.Copy(src, d, Math.Min(src.Length, n));
        return d;
    }

    static short[] QuantizeAa(double[] rad)
    {
        var q = new short[rad.Length];
        for (int i = 0; i < rad.Length; i++)
            q[i] = ClipI16(rad[i] * Constants.AaScale);
        return q;
    }

    static double[] DequantizeAa(short[] q)
    {
        var d = new double[q.Length];
        for (int i = 0; i < q.Length; i++)
            d[i] = q[i] / Constants.AaScale;
        return d;
    }

    static short[] QuantizeExpr(double[] v)
    {
        var q = new short[v.Length];
        for (int i = 0; i < v.Length; i++)
            q[i] = ClipI16(v[i] * Constants.ExprScale);
        return q;
    }

    static double[] DequantizeExpr(short[] q)
    {
        var d = new double[q.Length];
        for (int i = 0; i < q.Length; i++)
            d[i] = q[i] / Constants.ExprScale;
        return d;
    }

    static short ClipI16(double x)
    {
        var r = Math.Round(x);
        if (r > 32767) return 32767;
        if (r < -32767) return -32767;
        return (short)r;
    }

    static void WriteF32(BinaryWriter w, double[] v, int n)
    {
        for (int i = 0; i < n; i++)
            w.Write((float)v[i]);
    }

    static double[] ReadF32(BinaryReader r, int n)
    {
        var v = new double[n];
        for (int i = 0; i < n; i++)
            v[i] = r.ReadSingle();
        return v;
    }

    static void WriteI16(BinaryWriter w, short[] v)
    {
        foreach (var s in v)
            w.Write(s);
    }

    static short[] ReadI16(BinaryReader r, int n)
    {
        var v = new short[n];
        for (int i = 0; i < n; i++)
            v[i] = r.ReadInt16();
        return v;
    }

    static void WriteF16(BinaryWriter w, double[] v)
    {
        foreach (var x in v)
            w.Write(BitConverter.HalfToUInt16Bits((Half)(float)x));
    }

    static double[] ReadF16(BinaryReader r, int n)
    {
        var v = new double[n];
        for (int i = 0; i < n; i++)
            v[i] = (float)BitConverter.UInt16BitsToHalf(r.ReadUInt16());
        return v;
    }
}
