"""Test (do not assume) a midpoint subdivision representation of IDOL's cache."""
from datetime import datetime
import json
from pathlib import Path
import time
import uuid
import numpy as np

LAB = Path(__file__).resolve().parents[1]
CACHE = LAB / 'checkpoints/IDOL/cache_sub2'


def recover_midpoints(faces, vertex_count, base_count):
    old = faces < base_count
    counts = old.sum(axis=1)
    distribution = np.bincount(counts, minlength=4)
    if distribution.tolist() != [len(faces) // 4, len(faces) * 3 // 4, 0, 0]:
        raise ValueError('Not a four-child subdivision topology: ' + str(distribution.tolist()))
    corners = faces[counts == 1]
    old_ids = corners[corners < base_count]
    new_ids = corners[corners >= base_count].reshape(-1, 2)
    pairs = np.unique(np.stack([new_ids.ravel(), np.repeat(old_ids, 2)], axis=1), axis=0)
    if not np.array_equal(np.bincount(pairs[:, 0], minlength=vertex_count)[base_count:], np.full(vertex_count-base_count, 2)):
        raise ValueError('Every added vertex must have two original endpoints')
    return pairs[:, 1].reshape(vertex_count - base_count, 2).astype('<u4')


def coarse_faces(faces, base_count, parents):
    central = faces[np.all(faces >= base_count, axis=1)] - base_count
    edges = parents[central]
    vertices = []
    for i in range(3):
        left, right = edges[:, i], edges[:, (i + 1) % 3]
        matches = (left[:, :, None] == right[:, None, :]).any(axis=2)
        if not np.all(matches.sum(axis=1) == 1): raise ValueError('Invalid midpoint face intersection')
        vertices.append(left[matches])
    result = np.stack(vertices, axis=1)
    if np.any(np.diff(np.sort(result, axis=1), axis=1) == 0): raise ValueError('Degenerate parent face')
    return result


def main():
    output = LAB / 'reports/idol' / ('structure-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    output.mkdir()
    record = dict(status='running', hypothesis='Recover midpoint hierarchy from topology, verify all corrective coefficients', levels=[])
    started = time.perf_counter()
    try:
        pose = np.load(CACHE / 'init_podir_smplx_thu_newNeutral.npy', mmap_mode='r', allow_pickle=False)
        shape = np.load(CACHE / 'init_spdir_smplx_thu_newNeutral.npy', mmap_mode='r', allow_pickle=False)
        faces = np.load(CACHE / 'init_faces_smplx_newNeutral.npy', allow_pickle=False)
        vertex_count = len(shape)
        bases = [('pose', pose.reshape(486, len(shape), 3).transpose(1, 2, 0)), ('shape', shape)]
        while len(faces) % 4 == 0:
            base_count = int(np.sort(faces, axis=1)[:, 1].min())
            level = len(record['levels'])
            try:
                parents = recover_midpoints(faces, vertex_count, base_count)
            except ValueError as error:
                record['furtherSubdivisionStopReason'] = str(error)
                break
            np.save(output / f'level-{level}-parents.npy', parents, allow_pickle=False)
            measured = dict(vertices=vertex_count, baseVertices=base_count, addedVertices=len(parents), mappingBytes=parents.nbytes)
            for label, basis in bases:
                maximum, square_sum, bit_mismatches, count = 0., 0., 0, 0
                for start in range(0, len(parents), 2048):
                    block = parents[start:start+2048]
                    actual = np.ascontiguousarray(basis[base_count+start:base_count+start+len(block)])
                    average = (basis[block[:, 0]] + basis[block[:, 1]]) * np.float32(.5)
                    error = actual.astype('f8') - average.astype('f8')
                    maximum = max(maximum, float(np.abs(error).max()))
                    square_sum += np.square(error).sum()
                    bit_mismatches += int(np.count_nonzero(actual.view('u4') != np.ascontiguousarray(average).view('u4')))
                    count += actual.size
                measured[label] = dict(maxAbsoluteError=maximum, rms=float(np.sqrt(square_sum/count)),
                                     bitMismatches=bit_mismatches, testedScalars=count,
                                     baseBasisBytes=basis[:base_count].nbytes)
            record['levels'].append(measured)
            if any(measured[label]['maxAbsoluteError'] != 0 for label, _ in bases):
                record['furtherSubdivisionStopReason'] = 'Corrective coefficients are not exact midpoints'
                break
            faces = coarse_faces(faces, base_count, parents)
            vertex_count = base_count
        record['coarsestVerifiedVertices'] = vertex_count
        record['coarseFaces'] = len(faces)
        record['status'] = 'verified-coarsening' if vertex_count < len(shape) else 'hypothesis-rejected'
    except ValueError as error:
        record.update(status='hypothesis-rejected', error=str(error))
    finally:
        record['seconds'] = time.perf_counter()-started
        (output / 'metrics.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(str(output))
        print(json.dumps(record, indent=2))


if __name__ == '__main__': main()
