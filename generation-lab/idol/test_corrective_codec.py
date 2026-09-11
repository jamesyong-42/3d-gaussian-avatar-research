"""Small synthetic fixtures; no model, CUDA or checkpoint loading."""
import hashlib
import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from corrective_codec import encode_correctives, decode_coefficients, write_package


def fixture():
    rng = np.random.default_rng(42)
    parents = np.array([[0, 1], [1, 2], [0, 2]])
    coarse = np.zeros((3, 54, 9, 3), dtype='f4')
    coarse[:, [0, 5, 20]] = rng.normal(0, .01, (3, 3, 9, 3))
    coarse[0, 30, 0, 0] = -0.
    coarse[0, 31, 0, 0] = 1e-30  # Never threshold small coefficients.
    fine = np.concatenate([coarse, (coarse[parents[:, 0]]+coarse[parents[:, 1]])*np.float32(.5)])
    pose = fine.transpose(1, 2, 0, 3).reshape(486, 18)
    shape = rng.normal(0, .01, (3, 3, 20)).astype('f4')
    shape = np.concatenate([shape, (shape[parents[:, 0]]+shape[parents[:, 1]])*np.float32(.5)])
    faces = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]], dtype='u4')
    return pose, shape, faces


class CorrectiveCodecTests(unittest.TestCase):
    def test_geometry_comparison_refuses_accidental_broadcasting(self):
        source = Path(__file__).with_name('portability_trial.py').read_text(encoding='utf-8')
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'errors')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<geometry-guard>', 'exec'), namespace)
        a = (np.zeros((2, 1, 3)), np.zeros((2, 6)))
        b = (np.zeros((2, 3)), np.zeros((2, 6)))
        with self.assertRaisesRegex(ValueError, 'broadcasting is forbidden'): namespace['errors'](a, b)

    def test_roundtrip_preserves_every_pose_bit_and_shape_value(self):
        pose, shape, faces = fixture()
        arrays, info = encode_correctives(pose, shape, faces)
        rebuilt_pose, rebuilt_shape = decode_coefficients(arrays, info)
        np.testing.assert_array_equal(rebuilt_pose.view('u4'), pose.view('u4'))
        np.testing.assert_array_equal(rebuilt_shape, shape)
        self.assertEqual(info['nCoarse'], 3)
        self.assertEqual(info['nonzeroBlocks'], 11)
        self.assertEqual(info['coefficientMaxErrors'], {'pose': 0., 'shape': 0.})

    def test_non_midpoint_pose_and_shape_rejected(self):
        for index in [0, 1]:
            values = list(fixture())
            if index == 0: values[0][0, 9] += .01
            else: values[1][3, 0, 0] += .01
            with self.assertRaisesRegex(ValueError, 'exact midpoint'): encode_correctives(*values)

    def test_nonfinite_wrong_types_and_layouts_rejected(self):
        for mutation in ['nan', 'infinity', 'f64', 'short', 'float_faces', 'invalid_face', 'empty_faces']:
            pose, shape, faces = fixture()
            if mutation == 'nan': pose[0, 0] = np.nan
            if mutation == 'infinity': shape[0, 0, 0] = np.inf
            if mutation == 'f64': pose = pose.astype('f8')
            if mutation == 'short': shape = shape[:, :, :19]
            if mutation == 'float_faces': faces = faces.astype('f4')
            if mutation == 'invalid_face': faces[0, 0] = 6
            if mutation == 'empty_faces': faces = faces[:0]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): encode_correctives(pose, shape, faces)

    def test_bad_topology_rejected(self):
        pose, shape, faces = fixture()
        faces[0] = [0, 1, 3]
        with self.assertRaises(ValueError): encode_correctives(pose, shape, faces)

    def test_binary_manifest_is_aligned_hashed_and_refuses_overwrite(self):
        arrays, info = encode_correctives(*fixture())
        with tempfile.TemporaryDirectory(prefix='idol-correctives-test-') as tmp:
            out = Path(tmp)
            manifest = write_package(out, arrays, info)
            binary = (out / 'correctives.bin').read_bytes()
            self.assertEqual(hashlib.sha256(binary).hexdigest(), manifest['sha256'])
            self.assertEqual(len(binary), info['runtimeBytes'])
            self.assertEqual(json.loads((out / 'correctives.json').read_text()), manifest)
            for name, spec in manifest['arrays'].items():
                self.assertEqual(spec['offset'] % 4, 0)
                restored = np.frombuffer(binary, spec['dtype'], count=spec['bytes']//4, offset=spec['offset']).reshape(spec['shape'])
                np.testing.assert_array_equal(restored.view('u4'), arrays[name].view('u4'))
            with self.assertRaises(FileExistsError): write_package(out, arrays, info)


if __name__ == '__main__': unittest.main(verbosity=2)
