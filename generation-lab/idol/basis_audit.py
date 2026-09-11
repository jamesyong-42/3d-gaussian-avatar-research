"""Measure exact IDOL corrective-basis redundancy without executing model weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

# Bound native BLAS before importing NumPy; never alter the installed environment.
os.environ['OPENBLAS_NUM_THREADS'] = '6'
os.environ['OMP_NUM_THREADS'] = '6'
os.environ['MKL_NUM_THREADS'] = '6'
import numpy as np

LAB = Path(__file__).resolve().parents[1]
CACHE = LAB / 'checkpoints/IDOL/cache_sub2'


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): result.update(block)
    return result.hexdigest()


def exact_dictionary(rows):
    """Deduplicate complete FP32 rows byte-for-byte; no rounding or thresholding."""
    contiguous = np.ascontiguousarray(rows, dtype='<f4')
    raw = contiguous.view(np.dtype((np.void, contiguous.shape[1] * 4))).ravel()
    _, indices, inverse = np.unique(raw, return_index=True, return_inverse=True)
    dictionary = contiguous[indices]
    inverse = inverse.astype('<u4')
    # Check all values, including signed zero bits, not just a sample or hash.
    for start in range(0, len(rows), 4096):
        reconstructed = dictionary[inverse[start:start+4096]]
        if not np.array_equal(reconstructed.view('u4'), contiguous[start:start+4096].view('u4')):
            raise ValueError('Dictionary failed exact reconstruction')
    return dictionary, inverse


def audit(output):
    started = time.perf_counter()
    record = dict(status='running', startedAt=datetime.now(timezone.utc).isoformat(),
                  scope='Exact corrective-data redundancy audit, not browser promotion or a quality score')
    def save(): (output / 'metrics.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    save()
    try:
        manifest = json.loads((LAB / 'reports/idol-assets.json').read_text())
        names = ['init_podir_smplx_thu_newNeutral.npy', 'init_spdir_smplx_thu_newNeutral.npy', 'init_faces_smplx_newNeutral.npy']
        record['verifiedSources'] = {}
        for name in names:
            expected = next(x for x in manifest['cache'] if x['name'] == 'cache_sub2/' + name)
            actual = sha(CACHE / name)
            if actual != expected['sha256']: raise ValueError('Cache integrity mismatch: ' + name)
            record['verifiedSources'][name] = actual
        pose = np.load(CACHE / names[0], mmap_mode='r', allow_pickle=False)
        shape = np.load(CACHE / names[1], mmap_mode='r', allow_pickle=False)
        faces = np.load(CACHE / names[2], mmap_mode='r', allow_pickle=False)
        if pose.shape[1] != shape.shape[0] * 3: raise ValueError('Unexpected vertex layout')
        record['vertices'] = len(shape)
        record['gaussians'] = len(faces)
        record['poseFeatures'] = len(pose)
        pose_support = np.count_nonzero(pose, axis=1)
        shape_support = np.count_nonzero(shape, axis=(0, 1))
        record['pose'] = dict(bytes=pose.nbytes, nonzero=int(pose_support.sum()),
                            nonzeroByFeature=pose_support.tolist(), activeFeatures=np.flatnonzero(pose_support).tolist(),
                            finite=bool(np.isfinite(pose).all()))
        record['shape'] = dict(bytes=shape.nbytes, nonzero=int(shape_support.sum()),
                             nonzeroByFeature=shape_support.tolist(), finite=bool(np.isfinite(shape).all()))
        print(json.dumps({key: {k: v for k, v in record[key].items() if k != 'nonzeroByFeature'} for key in ['pose', 'shape']}), flush=True)
        save()
        if not record['pose']['finite'] or not record['shape']['finite']: raise ValueError('Nonfinite cache')
        active = np.flatnonzero(pose_support).astype('<u4')
        rows = pose[active].reshape(len(active), len(shape), 3).transpose(1, 2, 0).reshape(len(shape), -1)
        before = time.perf_counter()
        dictionary, indices = exact_dictionary(rows)
        record['poseDictionary'] = dict(uniqueRows=len(dictionary), rowDimensions=dictionary.shape[1],
            dictionaryBytes=dictionary.nbytes, indicesBytes=indices.nbytes, activeIndicesBytes=active.nbytes,
            totalBytes=dictionary.nbytes + indices.nbytes + active.nbytes,
            bitwiseRoundtrip=True, seconds=time.perf_counter()-before)
        np.save(output / 'pose-dictionary.npy', dictionary, allow_pickle=False)
        np.save(output / 'pose-indices.npy', indices, allow_pickle=False)
        np.save(output / 'pose-active-features.npy', active, allow_pickle=False)
        print('POSE DICTIONARY ' + json.dumps(record['poseDictionary']), flush=True)
        del rows, dictionary, indices
        for label, values in [('shape', shape), ('expression', shape[:, :, 10:])]:
            dictionary, indices = exact_dictionary(values.reshape(len(shape), -1))
            record[label + 'Dictionary'] = dict(uniqueRows=len(dictionary), rowDimensions=dictionary.shape[1],
                dictionaryBytes=dictionary.nbytes, indicesBytes=indices.nbytes,
                totalBytes=dictionary.nbytes + indices.nbytes, bitwiseRoundtrip=True)
            np.save(output / (label + '-dictionary.npy'), dictionary, allow_pickle=False)
            np.save(output / (label + '-indices.npy'), indices, allow_pickle=False)
            print(label.upper() + ' DICTIONARY ' + json.dumps(record[label + 'Dictionary']), flush=True)
        record['status'] = 'audit-passed'
        return 0
    except Exception as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        record['seconds'] = time.perf_counter() - started
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or LAB / 'reports/idol' / ('basis-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    print('EVIDENCE ' + str(output), flush=True)
    return audit(output)


if __name__ == '__main__': raise SystemExit(main())
