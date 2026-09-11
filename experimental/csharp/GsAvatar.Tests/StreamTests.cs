using GsAvatar;
using Xunit;

public class StreamTests
{
    static SmplxPose Canonical()
    {
        var pose = SmplxPose.Identity(1000);
        for (int i = 0; i < Constants.BodyDim; i++)
            pose.Body[i] = 0.01 * i;
        for (int i = 0; i < Constants.HandDim; i++)
        {
            pose.LeftHand[i] = -0.2 + 0.4 * i / (Constants.HandDim - 1);
            pose.RightHand[i] = 0.2 - 0.4 * i / (Constants.HandDim - 1);
        }
        pose.Expr[0] = 0.5;
        pose.Expr[1] = 0.1;
        pose.RootPos = new[] { 0.1, 0.0, 0.2 };
        pose.RootRot = new[] { 0.0, 0.0, 0.0, 1.0 };
        return pose;
    }

    [Fact]
    public void MediumRoundtrip()
    {
        var pose = Canonical();
        var blob = StreamCodec.Record(pose, Constants.LodMedium, 42);
        var back = StreamCodec.Apply(blob);
        Assert.Equal(1000u, back.CaptureTick);
        Assert.Equal(pose.RootPos[0], back.RootPos[0], 5);
        Assert.Equal(pose.Body[10], back.Body[10], 3);
        Assert.Equal(pose.Expr[0], back.Expr[0], 3);
    }

    [Fact]
    public void MediumFitsPhotonCap()
    {
        var n = StreamCodec.Record(SmplxPose.Identity(), Constants.LodMedium, 0).Length;
        Assert.InRange(n, 100, Constants.MediumMaxBytes);
    }

    [Fact]
    public void MatchesPythonGoldenIfPresent()
    {
        var golden = Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "testdata", "golden_medium.bin"));
        if (!File.Exists(golden))
            return; // python tests write this first
        var expected = File.ReadAllBytes(golden);
        var blob = StreamCodec.Record(Canonical(), Constants.LodMedium, 42);
        Assert.Equal(expected, blob);
    }

    [Fact]
    public void RemoteRules()
    {
        var remote = GsAvatarEntity.Remote();
        Assert.False(remote.Visible);
        Assert.Throws<EntityException>(() => remote.ApplyPose(SmplxPose.Identity()));
        remote.ApplySpawn();
        remote.ApplyPose(SmplxPose.Identity(1));
        Assert.True(remote.Visible);
    }

    [Fact]
    public void LodSizesMonotonic()
    {
        int Low() => StreamCodec.Record(SmplxPose.Identity(), Constants.LodLow, 0).Length;
        int Med() => StreamCodec.Record(SmplxPose.Identity(), Constants.LodMedium, 0).Length;
        int High() => StreamCodec.Record(SmplxPose.Identity(), Constants.LodHigh, 0).Length;
        Assert.True(Low() < Med());
        Assert.True(Med() < High());
    }
}
