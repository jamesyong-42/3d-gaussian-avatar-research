"""CPU-only boundary tests; no downloads, model execution, or environment changes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location('idol_assets', Path(__file__).with_name('prepare_assets.py'))
ASSETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSETS)


class CacheSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='idol-cache-test-')
        self.root = Path(self.temp.name)
        self.override = patch.object(ASSETS, 'DEST', self.root)
        self.override.start()

    def tearDown(self):
        self.override.stop()
        self.temp.cleanup()

    def archive(self, name, contents=b'fixture'):
        if isinstance(name, str):
            raw_name = name
            name = zipfile.ZipInfo(name)
            name.filename = raw_name  # Preserve malformed separators that Windows ZipInfo normalizes.
        with zipfile.ZipFile(self.root / 'cache_sub2.zip', 'w') as stream:
            stream.writestr(name, contents)

    def test_supported_payload_and_idempotent_reuse(self):
        for name in ('cache_sub2/test.npy', 'cache_sub2/template/dense_thuman_newNeutral.obj', 'demo_data/4.json'):
            self.archive(name)
            first = ASSETS.extract_cache()
            self.assertEqual(first, ASSETS.extract_cache())
            self.assertEqual((self.root / name).read_bytes(), b'fixture')

    def test_traversal_and_absolute_paths_rejected(self):
        for name in ('../escape.npy', '/cache_sub2/escape.npy', 'cache_sub2/../escape.npy',
                     'cache_sub2\\escape.npy', 'C:/cache_sub2/escape.npy', 'cache_sub2/test.npy:extra'):
            with self.subTest(name=name):
                self.archive(name)
                with self.assertRaises(ValueError): ASSETS.extract_cache()

    def test_unexpected_executable_rejected(self):
        self.archive('cache_sub2/helper.py')
        with self.assertRaises(ValueError): ASSETS.extract_cache()

    def test_archive_symlink_rejected(self):
        entry = zipfile.ZipInfo('cache_sub2/link.npy')
        entry.create_system = 3
        entry.external_attr = (0o120777 << 16)
        self.archive(entry, b'../target')
        with self.assertRaises(ValueError): ASSETS.extract_cache()

    def test_existing_same_size_corruption_is_not_blessed(self):
        self.archive('cache_sub2/test.npy', b'expected')
        (self.root / 'cache_sub2').mkdir()
        target = self.root / 'cache_sub2/test.npy'
        target.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'content mismatch'): ASSETS.extract_cache()
        self.assertEqual(target.read_bytes(), b'tampered')  # Refuse, never overwrite unknown files.

    def test_pickle_inspection_does_not_execute_globals(self):
        with zipfile.ZipFile(self.root / 'inspect.zip', 'w') as stream:
            stream.writestr('archive/data.pkl', b'cos\nsystem\n.')
        info = ASSETS.inspect_serialization('inspect.zip')
        self.assertEqual(info['pickleGlobals'], ['os system'])
        self.assertEqual(info['torchscriptCodeFiles'], 0)


if __name__ == '__main__': unittest.main(verbosity=2)
