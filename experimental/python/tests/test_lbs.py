from __future__ import annotations

import unittest

import numpy as np

from gsavatar.lbs import lbs_points, make_bone_transforms, trs_matrix
from gsavatar.loopback import make_stick_rig, posed_feet, walk_pose


class LbsTests(unittest.TestCase):
    def test_identity_pose_leaves_rest(self) -> None:
        rig = make_stick_rig(n_per_bone=8)
        T, rest = make_bone_transforms(
            rig.rest_local_pos, rig.rest_local_aa, rig.rest_local_aa, rig.parents
        )
        xyz = lbs_points(rig.rest_xyz, rig.weights, rig.bone_ids, T)
        np.testing.assert_allclose(xyz, rig.rest_xyz, atol=1e-9)

    def test_root_translation_moves_all(self) -> None:
        rig = make_stick_rig(n_per_bone=8)
        delta = np.array([0.3, 0.0, 1.1])
        T, _ = make_bone_transforms(
            rig.rest_local_pos,
            rig.rest_local_aa,
            rig.rest_local_aa,
            rig.parents,
            root_pos=delta,
        )
        xyz = lbs_points(rig.rest_xyz, rig.weights, rig.bone_ids, T)
        np.testing.assert_allclose(xyz, rig.rest_xyz + delta, atol=1e-9)

    def test_knee_bend_moves_foot_not_hips(self) -> None:
        rig = make_stick_rig(n_per_bone=20)
        posed = rig.rest_local_aa.copy()
        posed[2, 0] = 1.0  # left knee
        T0, _ = make_bone_transforms(
            rig.rest_local_pos, rig.rest_local_aa, rig.rest_local_aa, rig.parents
        )
        T1, _ = make_bone_transforms(
            rig.rest_local_pos, rig.rest_local_aa, posed, rig.parents
        )
        a = lbs_points(rig.rest_xyz, rig.weights, rig.bone_ids, T0)
        b = lbs_points(rig.rest_xyz, rig.weights, rig.bone_ids, T1)
        # hip cloud (first bone) barely moves; foot moves
        n = 20
        hip_delta = np.linalg.norm(b[:n] - a[:n], axis=1).mean()
        foot_delta = np.linalg.norm(b[rig.left_foot_index] - a[rig.left_foot_index])
        self.assertLess(hip_delta, 0.02)
        self.assertGreater(foot_delta, 0.05)

    def test_walk_feet_stay_near_ground(self) -> None:
        rig = make_stick_rig()
        ys = []
        for tick in range(60):
            pose, aa = walk_pose(tick, dt=1.0 / 30.0)
            lf, rf = posed_feet(rig, aa, pose.root_pos)
            ys.append(lf[1])
            ys.append(rf[1])
        # rest ankle is near y=1.0-0.45-0.45=0.1
        self.assertLess(min(ys), 0.25)
        self.assertGreater(min(ys), -0.05)

    def test_trs_orthonormal(self) -> None:
        T = trs_matrix([1, 2, 3], [0.2, -0.1, 0.4])
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
