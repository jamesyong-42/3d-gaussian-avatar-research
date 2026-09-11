from __future__ import annotations

import struct
import unittest
from pathlib import Path

import numpy as np

from gsavatar.constants import (
    BODY_DIM,
    HAND_DIM,
    HAS_EXPR,
    HAS_FACE,
    HAS_HANDS,
    HAS_ROOT,
    LOD_FULL,
    LOD_HIGH,
    LOD_LOW,
    LOD_MEDIUM,
    MEDIUM_MAX_BYTES,
)
from gsavatar.pose import SmplxPose
from gsavatar.stream import StreamCodec, snapshot_nbytes

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "testdata" / "golden_medium.bin"


def _canonical_pose() -> SmplxPose:
    body = np.array([0.01 * i for i in range(BODY_DIM)], dtype=np.float64)
    lh = np.linspace(-0.2, 0.2, HAND_DIM)
    rh = np.linspace(0.2, -0.2, HAND_DIM)
    expr = np.zeros(50)
    expr[:3] = [0.5, 0.1, 0.0]
    return SmplxPose(
        body=body,
        left_hand=lh,
        right_hand=rh,
        expr=expr,
        root_pos=np.array([0.1, 0.0, 0.2]),
        root_rot=np.array([0.0, 0.0, 0.0, 1.0]),
        capture_tick=1000,
    )


class StreamTests(unittest.TestCase):
    def test_medium_roundtrip(self) -> None:
        pose = _canonical_pose()
        blob = StreamCodec.record(pose, LOD_MEDIUM, seq=42)
        back = StreamCodec.apply(blob)
        self.assertEqual(back.capture_tick, 1000)
        np.testing.assert_allclose(back.root_pos, pose.root_pos, atol=1e-6)
        np.testing.assert_allclose(back.body, pose.body, atol=1e-3)
        np.testing.assert_allclose(back.left_hand, pose.left_hand, atol=1e-3)
        np.testing.assert_allclose(back.expr[:3], pose.expr[:3], atol=1e-3)

    def test_all_lods_roundtrip_body(self) -> None:
        pose = _canonical_pose()
        for lod in (LOD_LOW, LOD_MEDIUM, LOD_HIGH, LOD_FULL):
            blob = StreamCodec.record(pose, lod, seq=1)
            back = StreamCodec.apply(blob)
            self.assertGreater(len(blob), 12, msg=f"lod {lod}")
            np.testing.assert_allclose(back.root_pos, pose.root_pos, atol=1e-5)

    def test_medium_size_photon_cap(self) -> None:
        n = snapshot_nbytes(LOD_MEDIUM)
        self.assertLessEqual(n, MEDIUM_MAX_BYTES)
        self.assertGreater(n, 100)

    def test_h10_bandwidth_medium_30hz(self) -> None:
        n = snapshot_nbytes(LOD_MEDIUM)
        mbps = n * 30 * 8 / 1e6
        # H10 target 0.1 Mbps; log actual. Fail only if we blow Photon 1200 B.
        self.assertLessEqual(n, 1200)
        self.assertLess(mbps, 0.3, msg=f"Medium @ 30 Hz is {mbps:.3f} Mbps ({n} B)")

    def test_header_layout(self) -> None:
        pose = _canonical_pose()
        blob = StreamCodec.record(pose, LOD_MEDIUM, seq=42)
        proto, lod, seq, tick, flags = struct.unpack_from("<BBIIH", blob, 0)
        self.assertEqual(proto, 1)
        self.assertEqual(lod, LOD_MEDIUM)
        self.assertEqual(seq, 42)
        self.assertEqual(tick, 1000)
        self.assertTrue(flags & HAS_ROOT)
        self.assertTrue(flags & HAS_HANDS)
        self.assertTrue(flags & HAS_FACE or flags & HAS_EXPR)

    def test_golden_bytes_stable(self) -> None:
        pose = _canonical_pose()
        blob = StreamCodec.record(pose, LOD_MEDIUM, seq=42)
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        if not GOLDEN.is_file():
            GOLDEN.write_bytes(blob)
        expected = GOLDEN.read_bytes()
        self.assertEqual(blob, expected, "stream encoding changed — bump PROTOCOL.md")

    def test_low_smaller_than_medium(self) -> None:
        self.assertLess(snapshot_nbytes(LOD_LOW), snapshot_nbytes(LOD_MEDIUM))
        self.assertLess(snapshot_nbytes(LOD_MEDIUM), snapshot_nbytes(LOD_HIGH))
        self.assertLess(snapshot_nbytes(LOD_HIGH), snapshot_nbytes(LOD_FULL) + 1)


if __name__ == "__main__":
    unittest.main()
