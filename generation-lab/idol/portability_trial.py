"""Validate compact corrective fields against native IDOL; no reconstruction network."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch

import native_trial as native
from corrective_codec import encode_correctives, write_package

ROOT, OUT = native.ROOT, native.OUT
RECORD = dict(status='running', startedAt=datetime.now(timezone.utc).isoformat(), stages={},
              scope='Corrective-field portability spike. Native FK, skinning, nearest-neighbor scales and renderer remain native.',
              reconstructionNetworkLoaded=False, browserAvatarExported=False,
              limits={'correctiveMaxAbs': 1e-5, 'centerMaxDistance': 5e-6, 'covarianceMaxRelative': 1e-3},
              limitsNote='Frozen before the trial; distances in native coordinates, covariance uses packed-six-vector norm.')
native.REPORT = RECORD


def pool(field, deformer):
    # Same reduction layout/order as the author's deformer.
    return torch.cat([field[:, deformer.init_faces[..., i]] for i in range(3)], dim=1).mean(1)[0]


def compact_fields(feature, shape, base_pose, base_shape, parents):
    pose_coarse = (feature @ base_pose).reshape(1, -1, 3)
    shape_coarse = torch.einsum('bl,mkl->bmk', shape, base_shape)
    pose_fine = torch.cat([pose_coarse, (pose_coarse[:, parents[:, 0]] + pose_coarse[:, parents[:, 1]]) * .5], dim=1)
    shape_fine = torch.cat([shape_coarse, (shape_coarse[:, parents[:, 0]] + shape_coarse[:, parents[:, 1]]) * .5], dim=1)
    return pose_fine, shape_fine


def pose_slice(joint):
    if joint == 0: return slice(4, 7)
    if joint <= 21: return slice(7 + (joint-1)*3, 10 + (joint-1)*3)
    if joint <= 24: return slice(170 + (joint-22)*3, 173 + (joint-22)*3)
    return slice(80 + (joint-25)*3, 83 + (joint-25)*3)


def load_sources(count):
    sources, cases = [], []
    for i in range(count):
        folder = Path('/native') / str(i)
        metrics = json.loads((folder / 'metrics.json').read_text())
        if metrics['status'] != 'native-passed' or metrics['gaussianCount'] != 200800: raise ValueError('Invalid native source')
        with np.load(folder / 'canonical-diagnostic.npz', allow_pickle=False) as archive:
            attributes = {key: torch.from_numpy(archive[key].copy()).cuda() for key in archive.files}
        source = dict(folder=folder, metrics=metrics, attributes=attributes)
        sources.append(source)
        proof = {'folder': str(folder), 'source': metrics['input']['source'], 'files': {}}
        for path in [folder / 'metrics.json', folder / 'canonical-diagnostic.npz'] + sorted(folder.glob('pose-*.npz')):
            proof['files'][path.name] = native.sha(path)
        RECORD.setdefault('sources', []).append(proof)
        for name in ('front', 'side', 'back', 'arms', 'elbow', 'hand', 'jaw', 'expression'):
            with np.load(folder / ('pose-' + name + '.npz'), allow_pickle=False) as data:
                params = data['smplx'].copy()
            cases.append(dict(name=f'source-{i}-{name}', source=i, params=params, saved=name))
    base = next(c for c in cases if c['name'] == f'source-{count-1}-front')['params']
    for joint in range(54 + 1):
        params = base.copy()
        params[0, pose_slice(joint)] += [.21, -.14, .245]
        cases.append(dict(name=f'joint-{joint:02d}', source=count-1, params=params))
    for feature in range(20):
        params = base.copy()
        params[0, 70+feature if feature < 10 else 179+feature-10] = 2. if feature < 10 else 1.
        cases.append(dict(name=f'{"shape" if feature < 10 else "expression"}-{feature % 10:02d}', source=count-1, params=params))
    rng = np.random.default_rng(20260909)
    for index in range(8):
        params = base.copy()
        for joint in range(55): params[0, pose_slice(joint)] += rng.normal(0, .18, size=3)
        params[0, 1:4] = rng.uniform(-.1, .1, 3)
        params[0, 70:80] = rng.uniform(-2, 2, 10)
        params[0, 179:189] = rng.uniform(-1, 1, 10)
        cases.append(dict(name=f'combined-{index:02d}', source=count-1, params=params))
    return sources, cases


def geometry(canonical, rotation, multiplier, deformer, fixed_mask, init_rotation):
    from simple_knn._C import distCUDA2
    from lib.models.renderers.gau_renderer import batch_rodrigues, get_covariance
    centers, transforms = deformer(canonical[None], rotation[None], mask=fixed_mask, cano=False)
    centers = centers.reshape(-1, 3)
    rotation_matrix = transforms[0, :, :3, :3] @ (init_rotation @ batch_rodrigues(rotation))
    scales = distCUDA2(centers.contiguous()).clamp_min(1e-7).sqrt()[:, None] * multiplier
    covariance = get_covariance(scales, rotation_matrix)
    return centers, covariance


def errors(a, b):
    if a[0].shape != b[0].shape or a[1].shape != b[1].shape:
        raise ValueError('Geometry comparison requires identical shapes; broadcasting is forbidden')
    return dict(centerMaxDistance=float(torch.linalg.vector_norm(a[0]-b[0], dim=-1).max()),
                covarianceMaxRelative=float((torch.linalg.vector_norm(a[1]-b[1], dim=-1) / torch.linalg.vector_norm(a[1], dim=-1).clamp_min(1e-12)).max()),
                covarianceMaxAbsolute=float((a[1]-b[1]).abs().max()))


def render_comparison(source, name, original, compact):
    from PIL import Image
    from lib.models.renderers.gau_renderer import GRenderer
    renderer = GRenderer(image_size=[640, 896], bg_color=1).cuda()
    renderer.prepare(torch.tensor(source['metrics']['camera']['packed'], device='cuda'))
    frames = []
    for tag, values in [('native', original), ('compact', compact)]:
        frame = renderer.render_gaussian(means3D=values[0], cov3D_precomp=values[1],
            colors_precomp=source['attributes']['rgb'], opacities=source['attributes']['opacity'], rotations=None, scales=None)
        pixels = frame.permute(1, 2, 0).cpu().numpy()
        frames.append(pixels)
        Image.fromarray((pixels.clip(0, 1) * 255).round().astype('uint8')).save(OUT / f'{name}-{tag}.png')
    difference = np.abs(frames[0] - frames[1])
    Image.fromarray((difference * 64 * 255).clip(0, 255).round().astype('uint8')).save(OUT / f'{name}-diff64.png')
    return dict(name=name, maxPixelFloatError=float(difference.max()), meanPixelFloatError=float(difference.mean()), differenceGain=64)


def run(count):
    native.environment_and_assets(verify_neural_weights=False)
    cache = ROOT / 'work_dirs/cache_sub2'
    with native.stage('encode_correctives'):
        pose = np.load(cache / 'init_podir_smplx_thu_newNeutral.npy', mmap_mode='r', allow_pickle=False)
        shape = np.load(cache / 'init_spdir_smplx_thu_newNeutral.npy', mmap_mode='r', allow_pickle=False)
        faces = np.load(cache / 'init_faces_smplx_newNeutral.npy', allow_pickle=False)
        arrays, info = encode_correctives(pose, shape, faces)
        manifest = write_package(OUT, arrays, info)
        RECORD['package'] = {key: value for key, value in manifest.items() if key != 'arrays'}
        # Decode the actual sparse stored coefficients, not a fresh copy of the source basis.
        coarse = np.zeros((info['nCoarse'], 54, 9, 3), dtype='<f4')
        owners = np.repeat(np.arange(info['nCoarse']), np.diff(arrays['poseOffsets']))
        coarse[owners, arrays['poseBones']] = arrays['poseValues']
        base_pose = torch.from_numpy(np.ascontiguousarray(coarse.transpose(1, 2, 0, 3).reshape(486, -1))).cuda()
        base_shape = torch.from_numpy(arrays['shapeValues'].copy()).cuda()
        parents = torch.from_numpy(arrays['midpointParents'].astype('int64')).cuda()
        del coarse, arrays, owners, pose, shape, faces
        gc.collect()
    with native.stage('native_deformer_only'):
        import lib.models.deformers.smplx_deformer_gender as module
        original_class = module.SMPLX
        def npz_template(*args, **kwargs):
            kwargs['ext'] = 'npz'
            return original_class(*args, **kwargs)
        module.SMPLX = npz_template
        try: deformer = module.SMPLXDeformer_gender('neutral', is_sub2=True).cuda()
        finally: module.SMPLX = original_class
        init_rotation = torch.from_numpy(np.load(cache / 'init_rot_smplx_newNeutral.npy', allow_pickle=False)).cuda()
        fixed_mask = torch.from_numpy(np.load(cache / 'face_mask_thu_newNeutral.npy', allow_pickle=False) |
                                     np.load(cache / 'hands_mask_thu_newNeutral.npy', allow_pickle=False) |
                                     np.load(cache / 'outside_mask_thu_newNeutral.npy', allow_pickle=False)).cuda()[None]
    with native.stage('frozen_inputs'):
        sources, cases = load_sources(count)
        RECORD['bindings'] = []
        for i, source in enumerate(sources):
            weights = deformer.deformer.query_weights(source['attributes']['centers'][None]).clone()
            weights[:, fixed_mask[0]] = deformer.init_lbsw[fixed_mask]
            RECORD['bindings'].append(dict(source=i, nonzeroInfluences=int(torch.count_nonzero(weights)),
                weightSumMaxError=float((weights.sum(-1)-1).abs().max()), denseBytes=weights.numel()*4))
        del weights
    (OUT / 'validation').mkdir()
    RECORD['cases'], RECORD['renderComparisons'] = [], []
    with native.stage('all_native_control_checks'):
        for case in cases:
            print('CASE ' + case['name'], flush=True)
            source = sources[case['source']]
            attrs = source['attributes']
            params = torch.from_numpy(case['params']).cuda()
            deformer.prepare_deformer(params)
            feature = deformer.smpl_outputs.pose_feature
            coefficients = deformer.smpl_outputs.betas
            original_pose, original_shape = deformer.pose_offset, deformer.shape_offset
            ref_pose, ref_shape = pool(original_pose, deformer), pool(original_shape, deformer)
            compact_pose, compact_shape = compact_fields(feature, coefficients, base_pose, base_shape, parents)
            field_error = max(float((pool(compact_pose, deformer)-ref_pose).abs().max()),
                              float((pool(compact_shape, deformer)-ref_shape).abs().max()))
            args = (attrs['centers'], attrs['rotationAxisAngle'], attrs['radiusMultiplier'], deformer, fixed_mask, init_rotation)
            original_geometry = geometry(*args)
            deformer.pose_offset, deformer.shape_offset = compact_pose, compact_shape
            try: compact_geometry = geometry(*args)
            finally: deformer.pose_offset, deformer.shape_offset = original_pose, original_shape
            measured = errors(original_geometry, compact_geometry)
            row = dict(name=case['name'], source=case['source'], correctiveMaxAbs=field_error, **measured)
            if case.get('saved'):
                with np.load(source['folder'] / f'pose-{case["saved"]}.npz', allow_pickle=False) as saved:
                    saved_geometry = [torch.from_numpy(saved[key].copy()).reshape(info['nGaussians'], width).cuda()
                                      for key, width in [('centers', 3), ('covariance', 6)]]
                row['againstPreviousNative'] = errors(saved_geometry, original_geometry)
            row['passed'] = all(np.isfinite(row[key]) and row[key] <= limit for key, limit in RECORD['limits'].items())
            if row.get('againstPreviousNative'):
                row['passed'] &= all(row['againstPreviousNative'][key] <= RECORD['limits'][key] for key in ['centerMaxDistance', 'covarianceMaxRelative'])
            reference = np.zeros((info['nGaussians'], 8), dtype='<f4')
            reference[:, :3], reference[:, 4:7] = ref_shape.cpu().numpy(), ref_pose.cpu().numpy()
            path = OUT / 'validation' / f'{case["name"]}.bin'
            with path.open('xb') as stream: stream.write(reference.tobytes())
            row['validation'] = dict(file='validation/' + path.name, bytes=path.stat().st_size, sha256=native.sha(path),
                                     coefficients=torch.cat([feature[0], coefficients[0]]).cpu().tolist())
            RECORD['cases'].append(row)
            # Check geometry before asking the rasterizer to handle it. A failed
            # reference must never trigger an unbounded diagnostic render.
            native.save()
            print('CHECK ' + json.dumps({key: value for key, value in row.items() if key not in ['validation']}), flush=True)
            if not row['passed']: raise ValueError('Native corrective parity failed: ' + case['name'])
            if case.get('saved') in ('front', 'arms'):
                RECORD['renderComparisons'].append(render_comparison(source, case['name'], original_geometry, compact_geometry))
            native.save()
    validation = dict(layout='N x 8 float32: shape xyz / zero / pose xyz / zero', limits=RECORD['limits'],
                      cases=[dict(name=row['name'], **row['validation']) for row in RECORD['cases']])
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    RECORD['status'] = 'native-correctives-passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        with torch.no_grad(): run(args.sources)
        return 0
    except Exception as error:
        RECORD.update(status='failed', error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        RECORD.update(wallSeconds=time.perf_counter()-started, finishedAt=datetime.now(timezone.utc).isoformat())
        if torch.cuda.is_initialized():
            RECORD.update(peakAllocatedBytes=torch.cuda.max_memory_allocated(), peakReservedBytes=torch.cuda.max_memory_reserved())
        native.save()
        print(json.dumps({key: RECORD.get(key) for key in ['status', 'wallSeconds', 'error']}), flush=True)


if __name__ == '__main__': raise SystemExit(main())
