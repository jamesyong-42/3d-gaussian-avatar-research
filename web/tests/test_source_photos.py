import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import app
from fastapi.testclient import TestClient
from PIL import Image
from source_photos import matching_file, source_photos


class SourcePhotoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.avatars = self.root / "avatars"
        self.bundled = self.root / "bundled"
        self.example = self.root / "example" / "example.png"
        for folder in (self.avatars, self.bundled, self.example.parent):
            folder.mkdir()
        self.pngs = []
        for index in range(8):
            buffer = io.BytesIO()
            Image.new("RGB", (128, 192), (index * 30, 70, 100)).save(buffer, format="PNG")
            self.pngs.append(buffer.getvalue())
        self.example.write_bytes(self.pngs[0])
        for name, value in (("AVATARS", self.avatars), ("EXAMPLE", self.example), ("BUNDLED_PHOTOS", self.bundled)):
            replacement = patch.object(app, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        # No lifespan/GPU worker or real job state is touched by these GET tests.
        self.client = TestClient(app.app)
        self.addCleanup(self.client.close)

    def avatar(self, name="test-avatar", model="LHM-500M-HF", generation=None):
        folder = self.avatars / name
        folder.mkdir()
        (folder / "avatar.json").write_text(json.dumps({"model": model, "generation": generation or {}}))
        return folder

    def catalog(self, name="test-avatar"):
        return self.client.get(f"/api/avatars/{name}/source-photos")

    def test_synthetic_demo_has_no_source_photos(self):
        self.avatar(generation={"engine": "synthetic-demo"})
        response = self.catalog()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 0)
        self.assertEqual(response.json()["photos"], [])
        self.assertTrue(response.json()["synthetic"])

    def test_generation_is_explicitly_disabled_without_creating_a_job(self):
        with patch.object(app, "GENERATION_ENABLED", False):
            response = self.client.post("/api/jobs")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(list(self.avatars.iterdir()), [])

    def test_single_photo_roundtrip_and_private_headers(self):
        folder = self.avatar(generation={"inputSha256": hashlib.sha256(self.pngs[0]).hexdigest()})
        (folder / "input.png").write_bytes(self.pngs[0])
        response = self.catalog()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertNotIn(str(self.root), response.text)
        photo = response.json()["photos"][0]
        self.assertEqual(photo["name"], "input.png")
        image = self.client.get(photo["url"])
        self.assertEqual(image.content, self.pngs[0])
        self.assertEqual(image.headers["content-type"], "image/png")
        self.assertEqual(image.headers["x-content-type-options"], "nosniff")
        self.assertEqual(image.headers["cache-control"], "private, no-store")

    def test_eight_inputs_keep_generation_order(self):
        inputs = [{"file": f"view-{i:03d}.png", "sha256": hashlib.sha256(png).hexdigest()} for i, png in enumerate(self.pngs)]
        folder = self.avatar(model="LHMPP-700M-PixelShuffle", generation={"inputs": inputs, "views": 8, "shape": {"mode": "predicted"}})
        (folder / "inputs").mkdir()
        for index in reversed(range(8)):
            (folder / "inputs" / f"view-{index:03d}.png").write_bytes(self.pngs[index])
        result = self.catalog().json()
        self.assertEqual(result["total"], 8)
        self.assertEqual(result["shapeMode"], "predicted")
        for index, photo in enumerate(result["photos"]):
            self.assertEqual(photo["index"], index)
            self.assertEqual(self.client.get(photo["url"]).content, self.pngs[index])

    def test_missing_middle_input_is_not_silently_renumbered(self):
        folder = self.avatar(model="LHMPP-700M-PixelShuffle", generation={"views": 3})
        (folder / "inputs").mkdir()
        for index in (0, 2):
            (folder / "inputs" / f"view-{index:03d}.png").write_bytes(self.pngs[index])
        photos = self.catalog().json()["photos"]
        self.assertEqual([p["available"] for p in photos], [True, False, True])
        self.assertIsNone(photos[1]["url"])
        self.assertEqual(self.client.get("/api/avatars/test-avatar/source-photos/1").status_code, 404)
        self.assertEqual(self.client.get(photos[2]["url"]).content, self.pngs[2])

    def test_bundled_research_inputs_require_exact_hashes(self):
        name = "00000_yuliang_002.png"
        (self.bundled / name).write_bytes(self.pngs[2])
        self.avatar(model="LHMPP-700M-PixelShuffle", generation={"inputs": [{"file": name, "sha256": hashlib.sha256(self.pngs[2]).hexdigest()}]})
        photo = self.catalog().json()["photos"][0]
        self.assertEqual(self.client.get(photo["url"]).content, self.pngs[2])
        (self.bundled / name).write_bytes(self.pngs[3])
        self.assertFalse(self.catalog().json()["photos"][0]["available"])

    def test_local_input_hash_mismatch_is_reported_missing(self):
        folder = self.avatar(generation={"inputSha256": "0" * 64})
        (folder / "input.png").write_bytes(self.pngs[0])
        self.assertFalse(self.catalog().json()["photos"][0]["available"])

    def test_example_fallback_is_only_for_the_builtin_avatar(self):
        self.avatar("example")
        self.avatar()
        self.assertTrue(self.catalog("example").json()["photos"][0]["available"])
        self.assertFalse(self.catalog().json()["photos"][0]["available"])

    def test_metadata_paths_never_grant_arbitrary_file_access(self):
        digest = hashlib.sha256(self.pngs[0]).hexdigest()
        for index, filename in enumerate(("../example/example.png", str(self.example), "C:\\secret.png", "https://example.com/photo.png", "nested/photo.png")):
            name = f"unsafe-{index}"
            self.avatar(name, generation={"inputs": [{"file": filename, "sha256": digest}]})
            self.assertFalse(self.catalog(name).json()["photos"][0]["available"])

    def test_missing_hash_does_not_enable_bundled_fallback(self):
        (self.bundled / "sample.png").write_bytes(self.pngs[0])
        self.avatar(generation={"inputs": [{"file": "sample.png"}]})
        self.assertFalse(self.catalog().json()["photos"][0]["available"])

    def test_bad_metadata_is_a_bounded_error(self):
        for index, generation in enumerate(({"inputs": []}, {"inputs": [{}] * 9}, {"inputs": ["bad"]}, {"shape": None}, {"views": 99})):
            name = f"bad-{index}"
            self.avatar(name, model="LHMPP-700M-PixelShuffle", generation=generation)
            self.assertEqual(self.catalog(name).status_code, 409)

    def test_unknown_or_unfinished_avatar_has_no_source_endpoint(self):
        self.assertEqual(self.catalog("unknown").status_code, 404)
        (self.avatars / "unfinished").mkdir()
        (self.avatars / "unfinished" / "input.png").write_bytes(self.pngs[0])
        self.assertEqual(self.catalog("unfinished").status_code, 404)

    def test_indices_and_identifiers_are_bounded(self):
        self.avatar()
        for index in ("-1", "8", "99999"):
            self.assertEqual(self.client.get(f"/api/avatars/test-avatar/source-photos/{index}").status_code, 404)
        for name in ("..", "../example", "a/b", "a\\b", "a" * 81):
            with self.assertRaises(ValueError):
                source_photos(self.avatars, name, self.example, self.bundled)

    def test_symlink_escape_is_not_served(self):
        folder = self.avatar()
        try:
            (folder / "input.png").symlink_to(self.example)
            (self.avatars / "escaped").symlink_to(self.example.parent, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"OS does not permit test symlinks: {error}")
        self.assertFalse(self.catalog().json()["photos"][0]["available"])
        self.assertEqual(self.catalog("escaped").status_code, 404)

    def test_resolved_paths_must_stay_inside_the_allowed_root(self):
        folder = self.avatar()
        candidate = folder / "input.png"
        original_resolve = Path.resolve
        def redirected(path, *args, **kwargs):
            if path == candidate:
                return original_resolve(self.example)
            return original_resolve(path, *args, **kwargs)
        # Exercise the same containment boundary on Windows without requiring
        # developer mode / permission to create an actual symbolic link.
        with patch.object(Path, "resolve", redirected):
            self.assertIsNone(matching_file(candidate, folder))


if __name__ == "__main__":
    unittest.main()
