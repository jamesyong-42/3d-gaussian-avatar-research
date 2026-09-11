from __future__ import annotations

import unittest

from gsavatar.entity import EntityError, GsAvatarEntity, Preset
from gsavatar.loopback import LoopbackLink, foot_slide_metrics
from gsavatar.pose import SmplxPose


class EntityTests(unittest.TestCase):
    def test_remote_rejects_tracking(self) -> None:
        with self.assertRaises(EntityError):
            GsAvatarEntity(
                is_local=False,
                preset=Preset.REMOTE,
                tracking_source="headset",
            )

    def test_remote_rejects_animator(self) -> None:
        with self.assertRaises(EntityError):
            GsAvatarEntity(
                is_local=False,
                preset=Preset.REMOTE,
                allow_procedural_anim=True,
            )

    def test_remote_invisible_until_spawn_and_snapshot(self) -> None:
        remote = GsAvatarEntity.remote()
        self.assertFalse(remote.visible)
        with self.assertRaises(EntityError):
            remote.apply_pose(SmplxPose.identity())
        remote.apply_spawn()
        self.assertFalse(remote.visible)
        remote.apply_pose(SmplxPose.identity(tick=1))
        self.assertTrue(remote.visible)


class LoopbackTests(unittest.TestCase):
    def test_lockstep_walk_under_1cm(self) -> None:
        m = foot_slide_metrics(n_ticks=180, delay_s=0.06, desync=False)
        self.assertTrue(m["remote_visible"])
        self.assertLess(m["max_m"], 0.01, msg=m)

    def test_desync_transform_slides_over_5cm(self) -> None:
        m = foot_slide_metrics(n_ticks=180, delay_s=0.06, desync=True)
        self.assertGreater(m["max_m"], 0.05, msg=m)

    def test_loopback_delivers_packets(self) -> None:
        local = GsAvatarEntity.local()
        remote = GsAvatarEntity.remote()
        remote.apply_spawn()
        link = LoopbackLink(delay_s=0.05, dt=1.0 / 30.0)
        got = 0
        for _ in range(20):
            link.record_local(local, SmplxPose.identity(tick=link.tick))
            if link.drain_remote(remote) is not None:
                got += 1
            link.step_time()
        self.assertGreaterEqual(got, 10)
        self.assertTrue(remote.visible)


if __name__ == "__main__":
    unittest.main()
