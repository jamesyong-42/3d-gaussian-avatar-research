"""Exact coefficient storage for an IDOL corrective-field experiment (not an avatar format)."""
import numpy as np
from structural_audit import recover_midpoints


def encode_correctives(pose, shape, faces):
    if pose.dtype != np.float32 or shape.dtype != np.float32:
        raise ValueError('Expected native float32 directions')
    if shape.ndim != 3 or shape.shape[1:] != (3, 20) or pose.shape != (486, len(shape)*3):
        raise ValueError('Unexpected native basis layout')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu':
        raise ValueError('Expected integer triangle indices')
    if not np.isfinite(pose).all() or not np.isfinite(shape).all(): raise ValueError('Nonfinite basis')
    if faces.min() < 0 or faces.max() >= len(shape): raise ValueError('Face index out of range')
    base_count = int(np.sort(faces, axis=1)[:, 1].min())
    parents = recover_midpoints(faces, len(shape), base_count)
    pose_vertex = pose.reshape(54, 9, len(shape), 3).transpose(2, 0, 1, 3)
    # Preserve every bit of nonzero blocks, including negative zero if present.
    coarse = np.ascontiguousarray(pose_vertex[:base_count])
    support = np.any(coarse.view('u4') != 0, axis=(2, 3))
    bone_indices = np.nonzero(support)[1].astype('<u4')
    offsets = np.concatenate(([0], np.cumsum(support.sum(axis=1)))).astype('<u4')
    values = np.ascontiguousarray(coarse[support], dtype='<f4')
    rebuilt = np.zeros_like(coarse)
    rebuilt[support] = values
    if not np.array_equal(rebuilt.view('u4'), coarse.view('u4')): raise ValueError('Sparse coefficient roundtrip failed')
    maximum = {'pose': 0., 'shape': 0.}
    sign_zero_differences = {'pose': 0, 'shape': 0}
    for label, basis in [('pose', pose_vertex), ('shape', shape)]:
        for start in range(0, len(parents), 2048):
            pair = parents[start:start+2048]
            restored = (basis[pair[:, 0]] + basis[pair[:, 1]]) * np.float32(.5)
            original = np.ascontiguousarray(basis[base_count+start:base_count+start+len(pair)])
            error = float(np.max(np.abs(restored - original)))
            maximum[label] = max(maximum[label], error)
            if error != 0: raise ValueError(f'{label} is not an exact midpoint coefficient field')
            sign_zero_differences[label] += int(np.count_nonzero(np.ascontiguousarray(restored).view('u4') != original.view('u4')))
    arrays = dict(poseOffsets=offsets, poseBones=bone_indices, poseValues=values,
                  shapeValues=np.ascontiguousarray(shape[:base_count], dtype='<f4'),
                  midpointParents=parents, gaussianFaces=np.ascontiguousarray(faces, dtype='<u4'))
    info = dict(deformation='idol-corrective-field-subdivision-v1-lab', nVertices=len(shape),
                nCoarse=base_count, nGaussians=len(faces), nPoseFeatures=486, nShapeFeatures=20,
                nonzeroBlocks=len(values), coefficientMaxErrors=maximum,
                signedZeroBitDifferences=sign_zero_differences,
                sourcePoseBytes=pose.nbytes, sourceShapeBytes=shape.nbytes,
                runtimeBytes=sum(a.nbytes for a in arrays.values()),
                arrayBytes={name: a.nbytes for name, a in arrays.items()},
                limitation='Corrective-field storage only. Floating-point evaluation order changes; validate native results. Not a rigged avatar.')
    return arrays, info


def decode_coefficients(arrays, info):
    coarse_count, vertex_count = info['nCoarse'], info['nVertices']
    coarse = np.zeros((coarse_count, 54, 9, 3), dtype='<f4')
    for vertex in range(coarse_count):
        begin, end = arrays['poseOffsets'][vertex:vertex+2]
        coarse[vertex, arrays['poseBones'][begin:end]] = arrays['poseValues'][begin:end]
    pose = np.empty((vertex_count, 54, 9, 3), dtype='<f4')
    pose[:coarse_count] = coarse
    pairs = arrays['midpointParents']
    pose[coarse_count:] = (coarse[pairs[:, 0]] + coarse[pairs[:, 1]]) * np.float32(.5)
    shape = np.empty((vertex_count, 3, 20), dtype='<f4')
    shape[:coarse_count] = arrays['shapeValues']
    shape[coarse_count:] = (shape[pairs[:, 0]] + shape[pairs[:, 1]]) * np.float32(.5)
    return pose.transpose(1, 2, 0, 3).reshape(486, vertex_count*3), shape


def write_package(output, arrays, info):
    import hashlib
    import json
    digest = hashlib.sha256()
    offset, specs = 0, {}
    with (output / 'correctives.bin').open('xb') as stream:
        for name, array in arrays.items():
            data = np.ascontiguousarray(array).tobytes()
            stream.write(data)
            digest.update(data)
            specs[name] = dict(offset=offset, bytes=len(data), dtype=array.dtype.str, shape=list(array.shape))
            offset += len(data)
    manifest = {**info, 'arrays': specs, 'binary': 'correctives.bin', 'sha256': digest.hexdigest(), 'bytes': offset}
    (output / 'correctives.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    import json
    from pathlib import Path
    cache = Path(__file__).resolve().parents[1] / 'checkpoints/IDOL/cache_sub2'
    arrays, info = encode_correctives(
        np.load(cache / 'init_podir_smplx_thu_newNeutral.npy', mmap_mode='r', allow_pickle=False),
        np.load(cache / 'init_spdir_smplx_thu_newNeutral.npy', mmap_mode='r', allow_pickle=False),
        np.load(cache / 'init_faces_smplx_newNeutral.npy', allow_pickle=False))
    print(json.dumps(info, indent=2))
