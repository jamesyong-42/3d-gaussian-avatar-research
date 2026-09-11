namespace GsAvatar;

/// <summary>Keep in lockstep with python/gsavatar/constants.py and PROTOCOL.md.</summary>
public static class Constants
{
    public const byte ProtocolU8 = 1;
    public const ushort ProtocolSpawn = 1;

    public const byte LodLow = 0;
    public const byte LodMedium = 1;
    public const byte LodHigh = 2;
    public const byte LodFull = 3;

    public const ushort HasRoot = 1 << 0;
    public const ushort HasFace = 1 << 1;
    public const ushort HasHands = 1 << 2;
    public const ushort Teleport = 1 << 3;
    public const ushort HasExpr = 1 << 4;

    public const int BodyDim = 75;
    public const int HandDim = 45;
    public const int ExprFull = 50;
    public const int ExprMedium = 3;

    public const double AaScale = 32767.0 / Math.PI;
    public const double ExprScale = 1000.0;

    public const int MediumMaxBytes = 1200;

    public static readonly int[] LowBodyIndices =
    {
        0, 1, 2, 36, 37, 38, 45, 46, 47, 60, 61, 62, 63, 64, 65
    };
}
