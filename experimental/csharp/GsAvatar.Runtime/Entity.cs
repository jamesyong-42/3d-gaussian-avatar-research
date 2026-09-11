namespace GsAvatar;

public enum Preset
{
    Local = 0,
    Remote = 1,
}

public sealed class EntityException : InvalidOperationException
{
    public EntityException(string message) : base(message) { }
}

public sealed class GsAvatarEntity
{
    public bool IsLocal { get; }
    public Preset Preset { get; }
    public bool AllowProceduralAnim { get; }
    public string? TrackingSource { get; }
    public IPoseConsumer? PoseSink { get; set; }
    public bool Visible { get; private set; }
    public bool SpawnApplied { get; private set; }
    public bool FirstSnapshotApplied { get; private set; }
    public SmplxPose? LastPose { get; private set; }

    GsAvatarEntity(bool isLocal, Preset preset, bool allowAnim, string? tracking)
    {
        IsLocal = isLocal;
        Preset = preset;
        AllowProceduralAnim = allowAnim;
        TrackingSource = tracking;
        Validate();
    }

    public static GsAvatarEntity Local(IPoseConsumer? sink = null, string tracking = "headset") =>
        new(true, Preset.Local, true, tracking) { PoseSink = sink, Visible = true, SpawnApplied = true, FirstSnapshotApplied = true };

    public static GsAvatarEntity Remote(IPoseConsumer? sink = null) =>
        new(false, Preset.Remote, false, null) { PoseSink = sink, Visible = false };

    void Validate()
    {
        if (Preset == Preset.Remote || !IsLocal)
        {
            if (TrackingSource != null)
                throw new EntityException("remote entity must have tracking_source=null");
            if (AllowProceduralAnim)
                throw new EntityException("remote entity must have allow_procedural_anim=false");
            if (IsLocal)
                throw new EntityException("Preset.REMOTE requires is_local=false");
        }
    }

    public void ApplySpawn()
    {
        SpawnApplied = true;
        MaybeShow();
    }

    public void ApplyPose(SmplxPose pose)
    {
        if (!IsLocal && !SpawnApplied)
            throw new EntityException("remote must not render until spawn applied");
        LastPose = pose;
        FirstSnapshotApplied = true;
        MaybeShow();
        PoseSink?.ApplySmplx(pose);
    }

    void MaybeShow()
    {
        Visible = IsLocal || (SpawnApplied && FirstSnapshotApplied);
    }
}
