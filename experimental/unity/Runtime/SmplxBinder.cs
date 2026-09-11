using UnityEngine;

namespace GsAvatar.Unity
{
    /// <summary>
    /// Topic 3: Movement SDK / Humanoid joints → SmplxPose → IPoseConsumer.
    /// Attach to the LOCAL avatar only. Remote copies receive GsAvatarStream.
    /// </summary>
    public class SmplxBinder : MonoBehaviour
    {
        public Animator humanoid;
        public GsAvatarEntity entity;
        public MonoBehaviour consumer;
        public TextAsset jointMapJson;

        [Tooltip("Hide head Gaussians in first-person (Meta First Person analogue).")]
        public bool hideHeadInFirstPerson = true;

        public SmplxPose LastPose { get; private set; }

        void LateUpdate()
        {
            if (entity != null && !entity.isLocal)
                return;
            // Joint read + retarget is filled in spike 3M after Movement SDK is in the scene.
        }
    }
}
