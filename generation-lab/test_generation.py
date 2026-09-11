import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web/backend"))
from generation_config import DEFAULT_MODEL, MODEL_SPECS, LHMPP_MODEL, checkpoint_path, gpu_lease, validate_betas, validate_choice


class ConfigurationTests(unittest.TestCase):
    def test_default_and_allowed_choices(self):
        self.assertEqual(DEFAULT_MODEL, "LHM-500M-HF")
        for model in MODEL_SPECS:
            for shape in MODEL_SPECS[model]["shapeModes"]:
                validate_choice(model, shape)
        with self.assertRaises(ValueError): validate_choice(LHMPP_MODEL,"estimate")
        with self.assertRaises(ValueError): validate_choice(DEFAULT_MODEL,"predicted")
        for model, shape in [("../model", "zero"), (DEFAULT_MODEL, "file"), (DEFAULT_MODEL, "unknown")]:
            with self.assertRaises(ValueError): validate_choice(model, shape)

    def test_finite_shape_contract(self):
        self.assertEqual(validate_betas([0] * 10), [0.0] * 10)
        for betas in ([0] * 9, [0] * 11, [float("nan")] * 10, [float("inf")] * 10, [True] * 10, ["0"] * 10, None):
            with self.assertRaises(ValueError): validate_betas(betas)

    def test_gpu_lease_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "gpu.lock"
            with gpu_lease(lock):
                with self.assertRaises(RuntimeError):
                    with gpu_lease(lock): pass
            with gpu_lease(lock): pass

    def test_missing_and_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = {"path": Path(tmp), "bytes": 100}
            with patch.dict(MODEL_SPECS, {"test-model": spec}):
                with self.assertRaises(FileNotFoundError): checkpoint_path("test-model")
                (Path(tmp) / "config.json").write_text("{}")
                (Path(tmp) / "model.safetensors").write_bytes(b"partial")
                with self.assertRaises(ValueError): checkpoint_path("test-model")


class ApiTests(unittest.TestCase):
    def setUp(self):
        import app
        from fastapi.testclient import TestClient
        from PIL import Image
        self.backend = app
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [patch.object(app, "GENERATION_ENABLED", True), patch.object(app, "AVATARS", self.root / "avatars"),
                        patch.object(app, "JOBS", self.root / "jobs"), patch.object(app, "jobs", {}),
                        patch.object(app, "pool", MagicMock()), patch.object(app, "checkpoint_path", return_value=self.root),
                        patch.object(app,"lhmpp_readiness",return_value=(True,None))]
        for item in self.patches: item.start()
        self.client = TestClient(app.app)
        self.client.__enter__()
        buffer = io.BytesIO()
        Image.new("RGB", (128, 192)).save(buffer, format="PNG")
        self.photo = buffer.getvalue()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        for item in reversed(self.patches): item.stop()
        self.tmp.cleanup()

    def post(self, **data):
        return self.client.post("/api/jobs", data=data, files={"photo": ("photo.png", self.photo, "image/png")})

    def test_legacy_upload_preserves_defaults(self):
        result = self.post()
        self.assertEqual(result.status_code, 202)
        job = result.json()
        self.assertEqual(job["model"], DEFAULT_MODEL)
        self.assertEqual(job["shapeMode"], "zero")
        self.assertEqual(self.backend.pool.submit.call_args.args[-2:], (DEFAULT_MODEL, "zero"))

    def test_selected_model_persisted_and_forwarded(self):
        result = self.post(model="LHM-1B-HF", shape_mode="estimate")
        self.assertEqual(result.status_code, 202)
        job = result.json()
        stored = json.loads((self.root / "jobs" / (job["id"] + ".json")).read_text())
        self.assertEqual(stored["model"], "LHM-1B-HF")
        self.assertEqual(self.backend.pool.submit.call_args.args[-2:], ("LHM-1B-HF", "estimate"))

    def test_invalid_selection_never_creates_job(self):
        for data in ({"model": "../model"}, {"shape_mode": "file"}):
            self.assertEqual(self.post(**data).status_code, 400)
        self.assertFalse(self.backend.jobs)
        self.backend.pool.submit.assert_not_called()

    def test_uninstalled_model_rejected_before_upload(self):
        with patch.object(self.backend, "checkpoint_path", side_effect=FileNotFoundError("Missing checkpoint")):
            self.assertEqual(self.post(model="LHM-1B-HF").status_code, 409)
        self.backend.pool.submit.assert_not_called()

    def batch(self, count, **data):
        return self.client.post("/api/jobs",data={"model":LHMPP_MODEL,**data},files=[("photos",(f"../../view-{i}.png",self.photo,"image/png")) for i in range(count)])

    def test_multiphoto_order_normalization_and_native_shape(self):
        result=self.batch(4);self.assertEqual(result.status_code,202,result.text)
        job=result.json();self.assertEqual(job["photoCount"],4);self.assertEqual(job["shapeMode"],"predicted")
        folder=self.root/"avatars"/job["id"]/"inputs"
        self.assertEqual(sorted(p.name for p in folder.iterdir()),[f"view-{i:03d}.png" for i in range(4)])
        self.assertEqual(self.backend.pool.submit.call_args.args[-2:],(LHMPP_MODEL,"predicted"))

    def test_multiphoto_rejects_invalid_counts_and_mixed_fields(self):
        self.assertEqual(self.batch(9).status_code,400)
        self.assertEqual(self.batch(2,model=DEFAULT_MODEL).status_code,400)
        self.assertEqual(self.client.post("/api/jobs",data={"model":LHMPP_MODEL}).status_code,400)
        mixed=self.client.post("/api/jobs",data={"model":LHMPP_MODEL},files=[("photo",("a.png",self.photo)),("photos",("b.png",self.photo))])
        self.assertEqual(mixed.status_code,400);self.assertFalse(self.backend.jobs);self.backend.pool.submit.assert_not_called()

    def test_one_bad_view_rejects_entire_batch_before_output(self):
        result=self.client.post("/api/jobs",data={"model":LHMPP_MODEL},files=[("photos",("good.png",self.photo)),("photos",("bad.png",b"not an image"))])
        self.assertEqual(result.status_code,400);self.assertEqual(list((self.root/"avatars").iterdir()),[])
        self.backend.pool.submit.assert_not_called()

    def test_unavailable_docker_never_schedules_job(self):
        with patch.object(self.backend,"lhmpp_readiness",return_value=(False,"Docker unavailable")):
            result=self.batch(1);self.assertEqual(result.status_code,409);self.assertIn("Docker",result.text)
        self.backend.pool.submit.assert_not_called()

    def test_upload_limits_and_queue_limit(self):
        with patch.object(self.backend,"MAX_TOTAL_UPLOAD",len(self.photo)+10):
            self.assertEqual(self.batch(2).status_code,413)
        for _ in range(3): self.assertEqual(self.batch(1).status_code,202)
        self.assertEqual(self.batch(1).status_code,429)
        self.assertEqual(self.backend.pool.submit.call_count,3)


if __name__ == "__main__":
    unittest.main()
