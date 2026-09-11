namespace GsAvatar;

public sealed class SmplxPose
{
    public double[] Body { get; set; } = new double[Constants.BodyDim];
    public double[] LeftHand { get; set; } = new double[Constants.HandDim];
    public double[] RightHand { get; set; } = new double[Constants.HandDim];
    public double[] Expr { get; set; } = new double[Constants.ExprFull];
    public double[] RootPos { get; set; } = new double[3];
    public double[] RootRot { get; set; } = { 0, 0, 0, 1 };
    public uint CaptureTick { get; set; }
    public bool Teleport { get; set; }

    public static SmplxPose Identity(uint tick = 0) => new() { CaptureTick = tick };

    public SmplxPose Clone()
    {
        return new SmplxPose
        {
            Body = (double[])Body.Clone(),
            LeftHand = (double[])LeftHand.Clone(),
            RightHand = (double[])RightHand.Clone(),
            Expr = (double[])Expr.Clone(),
            RootPos = (double[])RootPos.Clone(),
            RootRot = (double[])RootRot.Clone(),
            CaptureTick = CaptureTick,
            Teleport = Teleport,
        };
    }
}

public interface IPoseConsumer
{
    void ApplySmplx(SmplxPose pose);
}
