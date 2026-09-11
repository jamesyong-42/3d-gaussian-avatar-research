from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from gsavatar.convert import convert_to_vanilla
from gsavatar.constants import PLY_BYTES_PER_VERTEX, SH_REST
from gsavatar.ply import read_3dgs_ply, write_3dgs_ply
from gsavatar.synthetic import build_synthetic_asset


class ConvertTests(unittest.TestCase):
    def test_pad_short_rest_to_248(self) -> None:
        asset = build_synthetic_asset(n_per_bone=4)
        cloud = asset.cloud
        # write a compact PLY with only 9 rest coeffs
        n = cloud.n
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
            + "".join(f"property float f_rest_{i}\n" for i in range(9))
            + "property float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\n"
            "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
            "end_header\n"
        )
        parts = [
            cloud.xyz,
            cloud.normals,
            cloud.f_dc,
            cloud.f_rest[:, :9],
            cloud.opacity.reshape(n, 1),
            cloud.scale,
            cloud.rot,
        ]
        packed = np.concatenate([np.asarray(p, dtype="<f4").reshape(n, -1) for p in parts], axis=1)
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "compact.ply"
            dst = Path(td) / "vanilla.ply"
            src.write_bytes(header.encode("ascii") + packed.tobytes())
            out = convert_to_vanilla(src, dst)
            self.assertEqual(out.f_rest.shape[1], SH_REST)
            back = read_3dgs_ply(dst)
            self.assertEqual(back.vertex_bytes(), n * PLY_BYTES_PER_VERTEX)
            np.testing.assert_allclose(back.xyz, cloud.xyz, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
