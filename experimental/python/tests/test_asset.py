from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gsavatar.asset import validate_folder
from gsavatar.constants import PLY_BYTES_PER_VERTEX
from gsavatar.joints import map_coverage, retarget_axis_angles
from gsavatar.ply import is_vanilla_layout, read_3dgs_ply
from gsavatar.synthetic import write_synthetic


class AssetTests(unittest.TestCase):
    def test_synthetic_folder_validates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            folder = write_synthetic(Path(td) / "avatar", n_per_bone=16)
            errs = validate_folder(folder)
            self.assertEqual(errs, [], msg=errs)
            ok, msg = is_vanilla_layout(folder / "gaussians.ply")
            self.assertTrue(ok, msg)
            cloud = read_3dgs_ply(folder / "gaussians.ply")
            self.assertEqual(cloud.vertex_bytes(), cloud.n * PLY_BYTES_PER_VERTEX)
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["ply_layout"], "vanilla_3dgs_248")
            self.assertNotEqual(meta.get("license"), "apache-2.0")

    def test_reject_wrong_property_count(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.ply"
            p.write_bytes(
                b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
                b"property float x\nproperty float y\nproperty float z\nend_header\n"
                + b"\x00" * 12
            )
            ok, msg = is_vanilla_layout(p)
            self.assertFalse(ok)
            self.assertIn("248", msg)


class JointMapTests(unittest.TestCase):
    def test_default_map_covers_smplx_body(self) -> None:
        cov = map_coverage()
        self.assertTrue(cov["complete"], msg=cov)
        self.assertEqual(cov["n_dst_total"], 22)

    def test_retarget_copies_axis_angles(self) -> None:
        import numpy as np

        src = {"Head": np.array([0.1, 0.0, -0.2]), "LeftHandWrist": np.array([0.0, 0.3, 0.0])}
        pose = retarget_axis_angles(src)
        np.testing.assert_allclose(pose.body[45:48], [0.1, 0.0, -0.2])
        np.testing.assert_allclose(pose.body[60:63], [0.0, 0.3, 0.0])


if __name__ == "__main__":
    unittest.main()
