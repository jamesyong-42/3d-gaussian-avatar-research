using UnityEngine;

namespace GsAvatar.Unity
{
    public enum GsPreset { Local = 0, Remote = 1 }

    /// <summary>
    /// Meta Preset_Default / Preset_Remote analogue. Tracking and procedural
    /// animation are illegal on Remote. Remote stays hidden until spawn + first snapshot.
    /// </summary>
    public class GsAvatarEntity : MonoBehaviour
    {
        public bool isLocal = true;
        public GsPreset preset = GsPreset.Local;
        public bool allowProceduralAnim;
        public Component trackingSource;
        public MonoBehaviour poseSink;

        public bool SpawnApplied { get; private set; }
        public bool FirstSnapshotApplied { get; private set; }

        void Awake()
        {
            Validate();
            if (!isLocal)
                gameObject.SetActive(false);
        }

        public void Validate()
        {
            if (preset == GsPreset.Remote || !isLocal)
            {
                if (trackingSource != null)
                    Debug.LogError("GsAvatarEntity remote: trackingSource must be null", this);
                if (allowProceduralAnim)
                    Debug.LogError("GsAvatarEntity remote: allowProceduralAnim must be false", this);
            }
        }

        public void ApplySpawn()
        {
            SpawnApplied = true;
            MaybeShow();
        }

        public void ApplyGsStreamData(byte[] bytes)
        {
            if (!isLocal && !SpawnApplied)
            {
                Debug.LogWarning("GsAvatarEntity remote ignored snapshot before spawn", this);
                return;
            }
            FirstSnapshotApplied = true;
            MaybeShow();
            // Decode lives in the net8 runtime for now; Unity will call a port of StreamCodec.
        }

        void MaybeShow()
        {
            bool show = isLocal || (SpawnApplied && FirstSnapshotApplied);
            if (gameObject.activeSelf != show)
                gameObject.SetActive(show);
        }
    }
}
