// Drop-in Unity copy of the pose sink. Keep behaviour aligned with
// csharp/GsAvatar.Runtime/Pose.cs. This file does not require the net8 DLL.

using UnityEngine;

namespace GsAvatar.Unity
{
    public struct SmplxPose
    {
        public float[] Body;      // 75
        public float[] LeftHand;  // 45
        public float[] RightHand; // 45
        public float[] Expr;      // 50
        public Vector3 RootPos;
        public Quaternion RootRot;
        public uint CaptureTick;
        public bool Teleport;
    }

    public interface IPoseConsumer
    {
        void ApplySmplx(SmplxPose pose);
    }
}
