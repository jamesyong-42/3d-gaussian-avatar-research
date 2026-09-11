"""Freeze native FK/geometry references and export a complete deformation contract.

No reconstruction network, new model download, or main-app asset publication.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import native_trial as native
import portability_trial as correction
from corrective_codec import encode_correctives, write_package

OUT, ROOT = native.OUT, native.ROOT
RECORD = dict(status='running', stages={}, startedAt=datetime.now(timezone.utc).isoformat(),
    scope='Complete deformation export and native references. Browser validation is a separate stage.',
    reconstructionNetworkLoaded=False, browserAvatarPromoted=False,
    limits=dict(jointTransformMaxAbs=2e-6, correctiveMaxAbs=1e-5, centerMaxDistance=5e-6,
                covarianceMaxRelative=1e-3, neighborMeanMaxRelative=1e-3),
    limitsNote='Frozen before browser implementation; distances in native coordinates. Covariance uses packed-six norm, denominator floor 1e-12. Neighbor comparison clamps means to 1e-7.')
native.REPORT = correction.RECORD = RECORD


def write_blob(name, arrays, **metadata):
    specs, offset, digest = {}, 0, hashlib.sha256()
    with (OUT / (name+'.bin')).open('xb') as stream:
        for key, value in arrays.items():
            array = np.ascontiguousarray(value)
            if array.dtype.str not in ('<f4', '<u4', '<i4'): raise ValueError('Unsupported dtype')
            if array.dtype.kind == 'f' and not np.isfinite(array).all(): raise ValueError('Nonfinite export')
            data = array.tobytes(); stream.write(data); digest.update(data)
            specs[key] = dict(offset=offset,bytes=len(data),dtype=array.dtype.str,shape=list(array.shape))
            offset += len(data)
    manifest = dict(**metadata,arrays=specs,binary=name+'.bin',bytes=offset,sha256=digest.hexdigest())
    (OUT/(name+'.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def cpu(t): return t.detach().cpu().contiguous().numpy()


def run(count):
    native.environment_and_assets(verify_neural_weights=False)
    cache = ROOT/'work_dirs/cache_sub2'
    with native.stage('corrective_contract'):
        arrays, info = encode_correctives(
            np.load(cache/'init_podir_smplx_thu_newNeutral.npy',mmap_mode='r',allow_pickle=False),
            np.load(cache/'init_spdir_smplx_thu_newNeutral.npy',mmap_mode='r',allow_pickle=False),
            np.load(cache/'init_faces_smplx_newNeutral.npy',allow_pickle=False))
        manifest = write_package(OUT,arrays,info)
        RECORD['correctives'] = dict(bytes=manifest['bytes'],sha256=manifest['sha256'])
        del arrays
    with native.stage('native_deformer'):
        import lib.models.deformers.smplx_deformer_gender as module
        from lib.models.deformers.fast_snarf.lib.model.deformer_smplx import skinning
        from lib.models.renderers.gau_renderer import batch_rodrigues, get_covariance
        from simple_knn._C import distCUDA2
        original = module.SMPLX
        def template(*args,**kwargs): kwargs['ext']='npz'; return original(*args,**kwargs)
        module.SMPLX = template
        try: deformer = module.SMPLXDeformer_gender('neutral',is_sub2=True).cuda().eval()
        finally: module.SMPLX = original
        fixed = torch.from_numpy(np.load(cache/'face_mask_thu_newNeutral.npy',allow_pickle=False)|
            np.load(cache/'hands_mask_thu_newNeutral.npy',allow_pickle=False)|
            np.load(cache/'outside_mask_thu_newNeutral.npy',allow_pickle=False)).cuda()[None]
        initial_rotation = torch.from_numpy(np.load(cache/'init_rot_smplx_newNeutral.npy',allow_pickle=False)).cuda()
    with native.stage('rig_contract'):
        from lib.models.deformers.smplx.lbs import vertices2joints
        model = deformer.body_model
        # Fold the linear joint regressor in float64, retain float32 runtime values.
        # This changes evaluation order, so A is independently checked for every case.
        reg = model.J_regressor.double()
        directions = torch.cat([model.shapedirs,model.expr_dirs],dim=-1).double()
        rig = dict(deformation='idol-native-deformation-v1-lab',nBones=55,nExpressions=10,nShape=10,
            parents=cpu(model.parents).tolist(),poseMean=cpu(model.pose_mean).tolist(),
            jointBase=cpu(vertices2joints(model.J_regressor,model.v_template[None])[0]).tolist(),
            jointDirections=cpu(torch.einsum('jv,vck->jck',reg,directions).float()).tolist(),
            packedInput='189: scale(1), translation(3), root(3), body(63), betas(10), hands(45+45), jaw/eyes(3+3+3), expression(10)',
            scalePolicy='Upstream body forward ignores scale; lab adapter requires scale=1. Actor scaling is not part of this contract.',
            jointFolding='Base joints use native float32 vertices2joints; shape/expression directions fold in float64 then store float32. Analytically linear, not bitwise floating equivalence; verify A.',
            coordinateSystem='Native IDOL frame; matrices row-major; axis-angle radians; native hand means added once.',
            sourceRevision=RECORD['sourceRevision'],templateSha256=RECORD['templateSha256'])
        (OUT/'rig.json').write_text(json.dumps(rig,indent=2),encoding='utf-8')
        RECORD['rigSha256'] = native.sha(OUT/'rig.json')
    with native.stage('frozen_sources_and_bindings'):
        sources,cases = correction.load_sources(count)
        RECORD['assets'] = []
        for index,source in enumerate(sources):
            attrs=source['attributes']; centers=attrs['centers']; n=len(centers)
            weights=deformer.deformer.query_weights(centers[None]).clone()
            weights[:,fixed[0]]=deformer.init_lbsw[fixed]
            template_points,inverse=skinning(centers[None],weights,deformer.tfs_inv_t,inverse=False)
            rest=(template_points-deformer.pose_offset_cano)[0]
            basis=inverse[0,:,:3,:3] @ (initial_rotation @ batch_rodrigues(attrs['rotationAxisAngle']))
            dense=cpu(weights[0]); support=dense!=0
            offsets=np.concatenate([[0],np.cumsum(support.sum(axis=1))]).astype('<u4')
            bones=np.nonzero(support)[1].astype('<u4'); values=np.ascontiguousarray(dense[support],dtype='<f4')
            rebuilt=np.zeros_like(dense);rebuilt[support]=values
            if not np.array_equal(rebuilt,dense): raise ValueError('Skinning coefficients changed')
            data=dict(restPositions=cpu(rest),restOrientations=cpu(basis),radiusMultipliers=cpu(attrs['radiusMultiplier']),
                colors=cpu(attrs['rgb']),opacities=cpu(attrs['opacity']),skinOffsets=offsets,skinBones=bones,skinWeights=values)
            asset=write_blob('asset-'+str(index),data,deformation=rig['deformation'],nGaussians=n,nBones=55,
                nInfluences=len(values),source=index,rig='rig.json',correctives='correctives.json',
                camera=source['metrics']['camera'],defaultParameters=next(c['params'][0].tolist() for c in cases if c['name']==f'source-{index}-front'),
                limitation='Complete geometry contract, not a published main-app avatar or validated renderer.')
            RECORD['assets'].append({k:asset[k] for k in ['source','nGaussians','nInfluences','bytes','sha256']})
            del weights,inverse,template_points,basis,dense,rebuilt,data
    (OUT/'validation').mkdir()
    RECORD['cases']=[]
    with native.stage('native_geometry_references'):
        for case in cases:
            source=sources[case['source']]; attrs=source['attributes']
            params=torch.from_numpy(case['params']).cuda()
            deformer.prepare_deformer(params)
            posed,transforms=deformer(attrs['centers'][None],attrs['rotationAxisAngle'][None],mask=fixed,cano=False)
            centers=posed.reshape(info['nGaussians'],3)
            orientation=transforms[0,:,:3,:3] @ (initial_rotation @ batch_rodrigues(attrs['rotationAxisAngle']))
            mean_distance=distCUDA2(centers.contiguous())
            scales=mean_distance.clamp_min(1e-7).sqrt()[:,None]*attrs['radiusMultiplier']
            covariance=get_covariance(scales,orientation)
            fields=torch.cat([correction.pool(deformer.shape_offset,deformer),correction.pool(deformer.pose_offset,deformer)],dim=-1)
            reference=np.zeros((len(centers),20),dtype='<f4')
            reference[:,:3]=cpu(centers); reference[:,3]=cpu(mean_distance)
            reference[:,4:10]=cpu(covariance); reference[:,12:18]=cpu(fields)
            if not np.isfinite(reference).all(): raise ValueError('Nonfinite native geometry')
            name=case['name']; path=OUT/'validation'/(name+'.bin')
            with path.open('xb') as stream: stream.write(reference.tobytes())
            row=dict(name=name,source=case['source'],params=case['params'][0].tolist(),
                jointTransforms=cpu(deformer.smpl_outputs.A[0]).reshape(-1).tolist(),
                controls=cpu(torch.cat([deformer.smpl_outputs.pose_feature[0],deformer.smpl_outputs.betas[0]])).tolist(),
                file='validation/'+path.name,bytes=path.stat().st_size,sha256=native.sha(path))
            if case.get('saved'):
                with np.load(source['folder']/('pose-'+case['saved']+'.npz'),allow_pickle=False) as saved:
                    old=[torch.from_numpy(saved[k].copy()).reshape(len(centers),width).cuda() for k,width in [('centers',3),('covariance',6)]]
                row['previousNative']=correction.errors(old,[centers,covariance])
                if row['previousNative']['centerMaxDistance']>RECORD['limits']['centerMaxDistance'] or row['previousNative']['covarianceMaxRelative']>RECORD['limits']['covarianceMaxRelative']:
                    raise ValueError('Previous native parity failed')
            RECORD['cases'].append(row)
            if case.get('saved') in ('front','arms'):
                from PIL import Image
                from lib.models.renderers.gau_renderer import GRenderer
                renderer=GRenderer(image_size=[640,896],bg_color=1).cuda()
                renderer.prepare(torch.tensor(source['metrics']['camera']['packed'],device='cuda'))
                frame=renderer.render_gaussian(means3D=centers,cov3D_precomp=covariance,colors_precomp=attrs['rgb'],opacities=attrs['opacity'],rotations=None,scales=None)
                pixels=cpu(frame.permute(1,2,0)).clip(0,1)
                Image.fromarray((pixels*255).round().astype('uint8')).save(OUT/(name+'-native.png'))
            native.save();print('NATIVE '+name,flush=True)
    validation=dict(layout='N x 20 f32: center XYZ, mean3SquaredDistance, covariance 6, pad2, shapeXYZ, poseXYZ, pad2',
        nGaussians=info['nGaussians'],limits=RECORD['limits'],rigSha256=RECORD['rigSha256'],cases=RECORD['cases'])
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    RECORD['status']='native-deformation-exported'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sources',type=int,choices=[1,2],required=True)
    args=parser.parse_args();started=time.perf_counter()
    try:
        with torch.no_grad():run(args.sources)
        return 0
    except Exception as e:
        RECORD.update(status='failed',error=str(e),traceback=traceback.format_exc());traceback.print_exc();return 1
    finally:
        RECORD.update(wallSeconds=time.perf_counter()-started,finishedAt=datetime.now(timezone.utc).isoformat())
        if torch.cuda.is_initialized():RECORD.update(peakAllocatedBytes=torch.cuda.max_memory_allocated(),peakReservedBytes=torch.cuda.max_memory_reserved())
        native.save();print(json.dumps({k:RECORD.get(k) for k in ['status','wallSeconds','error']}),flush=True)


if __name__=='__main__':raise SystemExit(main())
