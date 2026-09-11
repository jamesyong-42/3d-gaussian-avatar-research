"""Offline, native-only IDOL smoke trial. Never publishes a browser avatar."""
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time
import traceback

ROOT = Path('/opt/idol')
OUT = Path('/evidence')
sys.path.insert(0, str(ROOT))
REPORT = {'status': 'running', 'stages': {}, 'startedAt': datetime.now(timezone.utc).isoformat(),
          'scope': 'Bundled research input; native feasibility, not a quality benchmark or web export',
          'adaptations': ['Use native SMPL-X NPZ loader for the existing hash-verified neutral template',
                          'Disable unused training LPIPS and construct/load the inference model only once',
                          'CPU U2Net alpha matting precedes the unchanged native image crop',
                          'One pose per render batch; native geometry and covariance math retained']}


def save():
    (OUT / 'metrics.json').write_text(json.dumps(REPORT, indent=2), encoding='utf-8')


@contextmanager
def stage(name):
    import torch
    REPORT['activeStage'] = name
    save()
    print('STAGE ' + name, flush=True)
    started = time.perf_counter()
    try:
        yield
        if torch.cuda.is_initialized(): torch.cuda.synchronize()
    finally:
        REPORT['stages'][name] = time.perf_counter() - started
        save()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): result.update(block)
    return result.hexdigest()


def native_functions(names):
    """Execute exact pinned helper bodies without importing unused rembg/video code."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms import ToTensor
    source = ROOT / 'lib/utils/infer_util.py'
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names): raise ValueError('Native helper missing')
    def rgba_required(*args, **kwargs): raise ValueError('Supply pre-matted RGBA, no network preprocessing')
    namespace = dict(Path=Path, os=os, Image=Image, np=np, cv2=cv2, ToTensor=ToTensor,
                     torch=torch, math=math, json=json, remove=rgba_required)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def environment_and_assets(*, verify_neural_weights=True):
    import numpy as np
    import torch
    with stage('environment'):
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(6)
        torch.manual_seed(42)
        np.random.seed(42)
        torch.cuda.reset_peak_memory_stats()
        REPORT['environment'] = {name: importlib.metadata.version(name) for name in
                                 ['torch', 'torchvision', 'pytorch3d', 'numpy', 'pytorch-lightning']}
        REPORT['environment'].update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                                     totalVramBytes=torch.cuda.get_device_properties(0).total_memory)
        import diff_gaussian_rasterization as rasterizer
        REPORT['rasterizer'] = dict(module=rasterizer.__file__, fields=list(rasterizer.GaussianRasterizationSettings._fields),
            provenance=json.loads(importlib.metadata.distribution('diff-gaussian-rasterization').read_text('direct_url.json')))
        if 'antialiasing' not in rasterizer.GaussianRasterizationSettings._fields:
            raise RuntimeError('IDOL requires the pinned dr_aa rasterizer, not the old main-branch API')
        for name in ('fuse_cuda', 'filter_cuda', 'precompute_cuda'): __import__(name)
        from simple_knn._C import distCUDA2
        from pytorch3d.ops import knn_points
        points = torch.randn(1, 128, 3, device='cuda')
        assert torch.isfinite(distCUDA2(points[0])).all()
        assert torch.isfinite(knn_points(points, points, K=3).dists).all()
        from lib.models.renderers.gau_renderer import GRenderer
        renderer = GRenderer(image_size=[32, 32], bg_color=1).cuda()
        view = torch.eye(4, device='cuda')
        view[2, 3] = 2.
        camera = torch.cat([torch.tensor([32., 32., 16., 16.], device='cuda'), view.flatten()])
        renderer.prepare(camera)
        with torch.no_grad():
            smoke = renderer.render_gaussian(means3D=torch.zeros(1, 3, device='cuda'),
                colors_precomp=torch.tensor([[1., 0., 0.]], device='cuda'), rotations=None,
                opacities=torch.ones(1, 1, device='cuda'), scales=None,
                cov3D_precomp=torch.tensor([[.01, 0., 0., .01, 0., .01]], device='cuda'))
        assert torch.isfinite(smoke).all() and int((smoke.min(0).values < .99).sum()) > 0
        REPORT['cudaKernelSmoke'] = 'simple-knn, PyTorch3D and native antialiased Gaussian rasterizer executed; IDOL extensions imported'
    with stage('asset_integrity'):
        manifest = json.loads(Path('/asset-manifest.json').read_text())
        if not manifest['verified']: raise ValueError('Unverified download manifest')
        REPORT['sourceRevision'] = manifest['sourceRevision']
        REPORT['modelRevision'] = manifest['modelRevision']
        REPORT['verifiedAssets'] = {}
        REPORT['verificationScope'] = 'model, cache and template' if verify_neural_weights else 'cache and template only; no neural checkpoint used'
        for entry in (manifest['files'] if verify_neural_weights else []) + manifest['cache']:
            path = ROOT / 'work_dirs' / entry['name']
            if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
                raise ValueError('Asset integrity failed: ' + entry['name'])
            REPORT['verifiedAssets'][entry['name']] = entry['sha256']
        prior = json.loads(Path('/template-manifest.json').read_text())
        item = next(e for e in prior['files'] if e['name'] == 'human_model_files/smplx/SMPLX_NEUTRAL.npz')
        template = ROOT / 'lib/models/deformers/smplx/SMPLX/SMPLX_NEUTRAL.npz'
        if sha(template) != item['sha256']: raise ValueError('Template integrity failed')
        REPORT['templateSha256'] = item['sha256']
        REPORT['cacheShapes'] = {}
        for path in (ROOT / 'work_dirs/cache_sub2').glob('*.npy'):
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            REPORT['cacheShapes'][path.name] = dict(shape=list(array.shape), dtype=str(array.dtype))


def checkpoint_inventory():
    import torch
    with stage('checkpoint_inventory'):
        data = torch.load(ROOT / 'work_dirs/model.ckpt', map_location='cpu', weights_only=True, mmap=True)
        state = data['state_dict']
        REPORT['checkpoint'] = dict(topLevelKeys=list(data), tensors=len(state),
            prefixes=dict(Counter(key.split('.')[0] for key in state)),
            elementsByDtype=dict(Counter()))
        sizes = Counter()
        for value in state.values(): sizes[str(value.dtype)] += value.numel()
        REPORT['checkpoint']['elementsByDtype'] = dict(sizes)
        (OUT / 'checkpoint-keys.json').write_text(json.dumps({key: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                                                           for key, value in state.items()}, indent=2))
        return data, state


def template_check():
    """Same supplied arrays through both native serializers; no upstream PKL equivalence claim."""
    import numpy as np
    import torch
    from lib.models.deformers.smplx import SMPLX
    template_dir = ROOT / 'lib/models/deformers/smplx/SMPLX'
    kwargs = dict(gender='neutral', create_body_pose=False, create_betas=False, create_global_orient=False,
                  create_transl=False, create_expression=False, create_jaw_pose=False, create_leye_pose=False,
                  create_reye_pose=False, create_right_hand_pose=False, create_left_hand_pose=False,
                  use_pca=True, num_pca_comps=12, num_betas=10, flat_hand_mean=False)
    with stage('template_loader_parity'):
        # Retain this private template bridge in scoped scratch, never in the report/export.
        bridge = Path(tempfile.mkdtemp(prefix='idol-template-', dir='/scratch'))
        with np.load(template_dir / 'SMPLX_NEUTRAL.npz', allow_pickle=True) as archive:
            arrays = {key: archive[key] for key in archive.files}
        with (bridge / 'SMPLX_NEUTRAL.pkl').open('wb') as stream: pickle.dump(arrays, stream, protocol=4)
        a, b = SMPLX(str(template_dir), ext='npz', **kwargs), SMPLX(str(bridge), ext='pkl', **kwargs)
        states_a, states_b = a.state_dict(), b.state_dict()
        assert states_a.keys() == states_b.keys()
        assert all(torch.equal(value, states_b[key]) for key, value in states_a.items())
        rows = []
        for name in ('rest', 'root', 'shoulder', 'elbow', 'hand', 'jaw', 'expression'):
            inputs = {key: torch.zeros(1, dims) for key, dims in
                      [('betas', 10), ('body_pose', 63), ('global_orient', 3), ('transl', 3),
                       ('expression', 10), ('left_hand_pose', 45), ('right_hand_pose', 45),
                       ('jaw_pose', 3), ('leye_pose', 3), ('reye_pose', 3)]}
            if name == 'root': inputs['global_orient'][0, 1] = .5
            if name == 'shoulder': inputs['body_pose'][0, 47] = -.6
            if name == 'elbow': inputs['body_pose'][0, 53] = .7
            if name == 'hand': inputs['left_hand_pose'][0, 2] = .4
            if name == 'jaw': inputs['jaw_pose'][0, 0] = .25
            if name == 'expression': inputs['expression'][0, 0] = 1
            with torch.no_grad(): x, y = a(**inputs, use_pca=False), b(**inputs, use_pca=False)
            error = max((getattr(x, key) - getattr(y, key)).abs().max().item() for key in ['vertices', 'joints', 'A', 'pose_feature'])
            rows.append(dict(name=name, maxAbsoluteError=error))
            if error != 0: raise ValueError('Native NPZ/PKL array-loader mismatch')
        REPORT['templateLoaderParity'] = dict(exactBuffers=len(states_a), poses=rows,
            limit='Checks serialization equivalence for our template, not equality to an unavailable original IDOL PKL')
        del a, b, arrays
        gc.collect()


def native_trial(args):
    import numpy as np
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from lib.utils.train_util import instantiate_from_config
    import lib.models.deformers.smplx_deformer_gender as deformation_module
    from lib.models.renderers.gau_renderer import batch_rodrigues, get_covariance

    if args.image is None: raise ValueError('Native trial requires a prepared RGBA input')
    metadata_path = Path('/input-metadata.json')
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if metadata.get('inputSha256') and sha(args.image) != metadata['inputSha256']:
        raise ValueError('Prepared input hash mismatch')
    betas = metadata.get('betas', [0.] * 10)
    if len(betas) != 10 or not all(math.isfinite(float(x)) for x in betas): raise ValueError('Invalid body shape')
    REPORT['input'] = {**metadata, 'mountedSha256': sha(args.image)}
    REPORT['shapeMode'] = metadata.get('shapeMode', 'zero')
    REPORT['precision'] = 'FP32 inference, matching default module construction/load_state_dict; saved encoder BF16 weights are copied to FP32'
    helpers = native_functions(['load_image', 'prepare_camera', 'construct_camera', 'load_smplify_json'])
    with stage('native_preprocessing'):
        with Image.open(args.image) as im:
            if im.mode != 'RGBA': raise ValueError('Input must be pre-matted RGBA')
            im.save(OUT / 'input-alpha.png')
        photo = helpers['load_image'](args.image, str(OUT))
        Image.fromarray((photo.permute(1, 2, 0).numpy() * 255).round().astype('uint8')).save(OUT / 'input-native.png')
        REPORT['nativeInputShape'] = list(photo.shape)

    checkpoint, state = checkpoint_inventory()
    with stage('model_construction'):
        original_smplx = deformation_module.SMPLX
        def native_npz(*positional, **kwargs):
            kwargs['ext'] = 'npz'
            return original_smplx(*positional, **kwargs)
        deformation_module.SMPLX = native_npz
        config = OmegaConf.load(ROOT / 'configs/idol_v0.yaml')
        config.model.params.lambda_lpips = 0
        config.model.params.encoder.params.model_path = 'work_dirs/sapiens_1b_epoch_173_torchscript.pt2'
        with torch.no_grad(): model = instantiate_from_config(config.model)
        deformation_module.SMPLX = original_smplx
        model.eval().requires_grad_(False)
    with stage('checkpoint_coverage'):
        current = model.state_dict()
        # Nonlearned SMPL-X buffers must agree BEFORE loading (initialization uses them).
        body_checks = {}
        for key, value in current.items():
            if key.startswith('decoder.deformer.body_model.') and key in state:
                other = state[key]
                error = float((value.float() - other.float()).abs().max()) if value.numel() else 0.
                body_checks[key] = error
        REPORT['checkpointBodyBufferErrors'] = body_checks
        if any(error != 0 for error in body_checks.values()):
            raise ValueError('Supplied template differs from checkpoint body buffers; do not silently mix priors')
        unused = sorted(key for key in state if key.startswith('lpips.'))
        inference = {key: value for key, value in state.items() if key not in unused}
        missing, unexpected = model.load_state_dict(inference, strict=False)
        REPORT['checkpointCoverage'] = dict(missing=list(missing), unexpected=list(unexpected),
                                            ignoredTrainingMetricKeys=unused, loadedTensors=len(inference))
        if missing or unexpected: raise ValueError('Inference checkpoint coverage is incomplete')
        del current, inference, state, checkpoint
        gc.collect()
        model = model.cuda()
        REPORT['modelParameterDtypes'] = dict(Counter(str(value.dtype) for value in model.parameters()))
        REPORT['loadedAllocatedBytes'] = torch.cuda.memory_allocated()
    with torch.no_grad():
        with stage('image_to_uv'):
            code = model.forward_image_to_uv(photo.unsqueeze(0).cuda(), is_training=False)
            if not torch.isfinite(code).all(): raise ValueError('Nonfinite UV code')
            REPORT['uvCodeShape'] = list(code.shape)
        with stage('uv_to_gaussians'):
            uv = model.decoder._decode_feature(code)
            points = model.decoder._sample_feature(uv)
            if not all(torch.isfinite(x).all() for x in points): raise ValueError('Nonfinite Gaussian attributes')
            REPORT['gaussianCount'] = points[0].shape[1]
            # Release neural reconstruction layers before repeated native renders; no model math changes.
            decoder = model.decoder
            base_pose = model.get_default_smplx_params().cuda()
            base_pose[:, 70:80] = torch.tensor(betas, device='cuda')
            del uv, code, model
            gc.collect()
            torch.cuda.empty_cache()
        with stage('canonical_diagnostics'):
            sigma, rgb, radius, rotation, offset = points
            # Exactly the runner's +/-2cm hand clamp, applied once before diagnostic capture.
            mask = decoder.hands_mask[..., None].expand(-1, -1, 3)
            offset[mask] = offset[mask].clamp(-.02, .02)
            centers = decoder.init_pcd + offset
            np.savez(OUT / 'canonical-diagnostic.npz',
                     centers=centers[0].cpu().numpy(), rgb=rgb[0].cpu().numpy(),
                     opacity=np.ones((REPORT['gaussianCount'], 1), dtype=np.float32),
                     radiusMultiplier=(radius[0] + 1).cpu().numpy(), rotationAxisAngle=rotation[0].cpu().numpy())
            # Native deformer queries all 55 weights at predicted canonical centers, then overrides masks.
            deformer = decoder.deformer
            weights = deformer.deformer.query_weights(centers).clone()
            fixed = decoder.face_mask + decoder.hands_mask + decoder.outside_mask
            weights[:, fixed[0]] = deformer.init_lbsw[fixed]
            REPORT['exportFeasibility'] = dict(
                browserCompatible=False, nativeJoints=weights.shape[-1], expressionDimensions=10,
                weightSumMaxError=float((weights.sum(-1) - 1).abs().max()),
                weightsFloat32Bytes=weights.numel() * 4,
                sourceShapeBasisBytes=deformer.init_spdir.numel() * 4,
                sourcePoseBasisBytes=deformer.init_podir.numel() * 4,
                naivePerGaussianPoseBasisBytes=REPORT['gaussianCount'] * 3 * 486 * 4,
                reasons=['Pose corrective offsets (486 pose-feature dimensions)',
                         'SMPL-X full-hand poses include native hand means; 10 expression coefficients',
                         'Per-pose 3-nearest-neighbor Gaussian sizes',
                         'Full covariance transport including nonorthogonal blended transforms'])
            del weights, centers

        K, cameras = helpers['prepare_camera'](resolution_x=896, resolution_y=640, num_views=1)
        camera = helpers['construct_camera'](K, cameras).reshape(1, 1, 20)
        decoder.renderer.image_size = [640, 896]
        REPORT['camera'] = dict(nativeHelperParameters={'resolution_x': 896, 'resolution_y': 640},
                                packed=camera[0, 0].cpu().tolist(), imageSize=[640, 896])
        poses = {}
        for name in ('front', 'side', 'back', 'arms', 'elbow', 'hand', 'jaw', 'expression'):
            pose = base_pose.clone()
            if name == 'side': pose[0, 5] = math.pi / 2
            if name == 'back': pose[0, 5] = math.pi
            if name == 'arms': pose[0, 54], pose[0, 57] = 0., 0.
            if name == 'elbow': pose[0, 60] = 1.
            if name == 'hand': pose[0, 82] += .6
            if name == 'jaw': pose[0, 170] = .25
            if name == 'expression': pose[0, 179] = 1.
            poses[name] = pose
        REPORT['poses'] = []
        first_scales = first_centers = None
        with stage('native_deformation_and_render'):
            for name, pose in poses.items():
                began = time.perf_counter()
                deformed = decoder.deform_pcd(points, pose, zeros_hands_off=True, value=.02)
                output = decoder.forward_render(deformed, camera, num_imgs=1)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - began
                xyz, _, _, _, _, transforms, rot = deformed
                frame = output['image'][0, 0].float().cpu().numpy()
                if not np.isfinite(frame).all(): raise ValueError('Nonfinite rendered image')
                visible = (frame.min(axis=-1) < .97)
                if visible.sum() < 1000: raise ValueError('Empty/near-empty native render: ' + name)
                image_name = 'native-' + name + '.png'
                Image.fromarray((frame.clip(0, 1) * 255).round().astype('uint8')).save(OUT / image_name)
                rotation_base = decoder.init_rot @ batch_rodrigues(rot.reshape(-1, 3))
                rotation_deformed = transforms[0, :, :3, :3] @ rotation_base
                scales = output['scales'][0]
                covariance = get_covariance(scales, rotation_deformed)
                if not all(torch.isfinite(t).all() for t in [xyz, scales, covariance]): raise ValueError('Nonfinite deformation')
                if first_scales is None: first_scales, first_centers = scales.clone(), xyz.clone()
                row = dict(name=name, file=image_name, seconds=seconds, visiblePixels=int(visible.sum()),
                           maxCenterDisplacement=float(torch.linalg.vector_norm(xyz - first_centers, dim=-1).max()),
                           meanRelativeScaleChange=float(((scales - first_scales).abs() / first_scales.clamp_min(1e-7)).mean()),
                           maxRotationOrthonormalityError=float((rotation_deformed @ rotation_deformed.transpose(-1, -2) - torch.eye(3, device='cuda')).abs().max()))
                np.savez(OUT / ('pose-' + name + '.npz'), smplx=pose.cpu().numpy(), centers=xyz[0].cpu().numpy(),
                         scales=scales.cpu().numpy(), covariance=covariance.cpu().numpy())
                REPORT['poses'].append(row)
                save()
                print('POSE ' + json.dumps(row), flush=True)
                del output, deformed
        REPORT['artifacts'] = ['input-native.png', 'canonical-diagnostic.npz'] + [row['file'] for row in REPORT['poses']]
        REPORT['note'] = 'Native pose fixtures only. No browser, Mixamo, remote-signal or product-use validation for IDOL.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['check', 'native'], required=True)
    parser.add_argument('--image', type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        environment_and_assets()
        if args.stage == 'check':
            checkpoint_inventory()
            template_check()
            REPORT['status'] = 'preflight-passed'
        else:
            native_trial(args)
            REPORT['status'] = 'native-passed'
        return 0
    except Exception as error:
        REPORT.update(status='failed', errorType=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        import torch
        REPORT['wallSeconds'] = time.perf_counter() - started
        if torch.cuda.is_initialized():
            REPORT['peakAllocatedBytes'] = torch.cuda.max_memory_allocated()
            REPORT['peakReservedBytes'] = torch.cuda.max_memory_reserved()
        REPORT['finishedAt'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({'status': REPORT['status'], 'wallSeconds': REPORT['wallSeconds'], 'error': REPORT.get('error')}), flush=True)


if __name__ == '__main__': raise SystemExit(main())
