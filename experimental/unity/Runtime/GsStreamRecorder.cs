using UnityEngine;

namespace GsAvatar.Unity
{
    /// <summary>
    /// Pose-stream recorder. NGO transport is a later swap:
    /// [Networked] byte[1200] + length, owner LateUpdate capture, others apply.
    /// Do not use a 512-byte unreliable RPC (Unity-Movement #125).
    /// </summary>
    public class GsStreamRecorder : MonoBehaviour
    {
        public const int Capacity = 1200;
        public GsAvatarEntity entity;
        public byte lod = 1;
        public uint captureTick;

        public byte[] RecordGsStreamData(byte streamLod)
        {
            return System.Array.Empty<byte>();
        }

        public void ApplyGsStreamData(byte[] data)
        {
            if (data != null && data.Length > Capacity)
            {
                Debug.LogError($"GsStreamRecorder payload {data.Length} > {Capacity}", this);
                return;
            }
            entity?.ApplyGsStreamData(data);
        }

        void LateUpdate()
        {
            if (entity == null || !entity.isLocal)
                return;
            captureTick += 1;
        }
    }
}
