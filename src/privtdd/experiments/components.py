"""E2: paired component ablations with a common final safeguard and resumable rows."""
from dataclasses import asdict
import numpy as np
from . import common as C

VARIANTS=('Full','F1 + Post','w/o Beta reference','w/o Geo reference','Mass-only correction',
          'Shape-only correction','GlobalMix','w/o Post')
ESTIMATORS=()


def fit_coordinates(family,active,regularization=1.,global_mix=False):
    from privtdd.correction.optimizer import fit_joint
    from privtdd.config import JOINT_MAX_ITERATIONS,JOINT_TOLERANCE
    if global_mix:
        mapping=np.zeros((16,2))
        mapping[[0,1,4,5,15],0]=1.
        mapping[[2,3,6,7,14],1]=1.
        transformed=dict(family,delta_tables=mapping.T@family['delta_tables'],
            A=np.vstack((family['A']@mapping,[1.,1.])),b=np.r_[family['b'],1.])
        z,stats=fit_joint(transformed,(0,1),max_iterations=JOINT_MAX_ITERATIONS,
                         tolerance=JOINT_TOLERANCE,quadratic=regularization*np.eye(2))
        return mapping@z,dict(stats,global_weights=z.tolist(),regularization=regularization,
                             regularizer='0.5*lambda*(geo_weight^2+beta_weight^2)')
    x,stats=fit_joint(family,active,max_iterations=JOINT_MAX_ITERATIONS,
                     tolerance=JOINT_TOLERANCE,quadratic=regularization*np.eye(16))
    return x,dict(stats,regularization=regularization,likelihood_scale='sum over represented nodes')


def candidate(shared,family,x,cfg,*,post=True,raw_path=None):
    from privtdd.experiments.ablation import frozen_histogram,observation_score
    from privtdd.posterior.update import update_posterior_joint
    pre,audit=frozen_histogram(family,x,shared['n'])
    normalizer,score=observation_score(family,x,shared=shared,config=cfg)
    hist=update_posterior_joint(family,x,normalizer,shared['caps'],shared['report'],shared['prior'],
         shared['p'],shared['ed'],shared['el'],workers=cfg.workers,report_counts=shared['counts'],
         batch_small_groups=cfg.batch_small_groups) if post else pre
    if raw_path is not None:C.save_hist(raw_path,hist)
    guard=dict(enabled=cfg.safeguard)
    if cfg.safeguard:
        from privtdd.release.transport_guard import capacity_reference,safeguard,capacity_violation
        D=len(shared['prior'])-1
        if '_safeguard_reference' not in shared:
            shared['_safeguard_reference']=capacity_reference(shared['raw_noisy'],shared['prior'],D)
        _,reference,_,cap=shared['_safeguard_reference']
        keys=np.array(sorted(hist),np.int64);raw=(keys,np.array([hist[int(k)] for k in keys]))
        if capacity_violation(raw,cap)>2e-9:raise ArithmeticError('Ablation violates degree capacity')
        final,stats=safeguard(raw,reference,tau=cfg.safeguard_tau,D=D)
        if capacity_violation(final,cap)>2e-9:raise ArithmeticError('Guarded ablation violates degree capacity')
        hist=dict(zip(final[0],final[1]));guard.update(stats)
    return hist,pre,dict(audit,**score,safeguard=guard)


def execute(task,directory):
    from privtdd.experiments.ablation import source_family,frozen_histogram,observation_score
    from privtdd.config import JOINT_COORDINATES
    from threadpoolctl import threadpool_limits
    path=C.graph_for(task);reference,shared,info=C.ensure_joint(task,path);cfg=C.config(task)
    predictions={};extra={}
    variants=tuple(task.get('variants') or (VARIANTS+(('w/o safeguard',) if task.get('include_guard_ablation') else ())))
    for name in variants:
        out=directory/name.replace(' ','_').replace('/','_');out.mkdir(exist_ok=True)
        predictions[name]=out/'prediction.npz';extra[name]=dict(report_sha256=info['report_meta']['sha256'])
        identity=C.token(dict(reference=str(reference),report=info['report_meta']['sha256'],
                              config=asdict(cfg),variant=name,code=task['code']))
        identity_file=out/'identity.json'
        if identity_file.exists() and C.read_json(identity_file)['identity']!=identity:
            raise ValueError('Ablation row identity changed')
        C.write_json(identity_file,dict(identity=identity))
        marker=out/'complete.json'
        if marker.exists():
            state=C.read_json(marker)
            if state['identity']!=identity:raise ValueError('Ablation completion identity changed')
            for file,sha in state['artifacts'].items():
                if C.data.digest(out/file)!=sha:raise ValueError('Changed completed ablation artifact')
            continue
        family=shared['family'];fit={}
        with threadpool_limits(limits=cfg.blas_threads):
            if name in ('Full','w/o Post','w/o safeguard'):x=shared['coordinates']
            elif name=='F1 + Post':x=np.zeros(16)
            else:
                active=JOINT_COORDINATES
                if name=='w/o Beta reference':
                    family=source_family(shared,'no_beta',out/'family',cfg);active=(0,1,4,5)
                elif name=='w/o Geo reference':
                    family=source_family(shared,'no_geo',out/'family',cfg);active=(3,6,7)
                elif name=='Mass-only correction':active=(0,1,3)
                elif name=='Shape-only correction':active=(4,5,6,7)
                fit_file=out/'fit.json'
                if fit_file.exists():
                    saved=C.read_json(fit_file);x=np.asarray(saved['coordinates']);fit=saved['fit']
                else:
                    x,fit=fit_coordinates(family,active,global_mix=name=='GlobalMix')
                    C.write_json(fit_file,dict(coordinates=x,fit=fit))
            if name in ('Full','w/o safeguard'):
                # No duplicate S6: these are exact saved production artifacts.
                file=reference/('prediction.npz' if name=='Full' else 'work/prediction_raw.npz')
                k,c=C.load_hist(file);hist=dict(zip(k,c))
                pre,audit=frozen_histogram(family,x,shared['n'])
                _,score=observation_score(family,x,shared=shared,config=cfg);audit.update(score)
                audit['safeguard']=shared['original_info']['safeguard'] if name=='Full' else dict(enabled=False)
                rk,rc=C.load_hist(reference/'work/prediction_raw.npz')
                C.save_hist(out/'prediction_raw.npz',dict(zip(rk,rc)))
            else:
                hist,pre,audit=candidate(shared,family,x,cfg,post=name!='w/o Post',raw_path=out/'prediction_raw.npz')
            if name=='GlobalMix':
                weights=np.asarray(fit['global_weights'])
                audit['penalty']=.5*float(weights@weights)
                audit['penalized_minus_F1']=audit['nll_minus_F1']+audit['penalty']
        C.save_hist(out/'prediction.npz',hist);C.save_hist(out/'S5_prediction.npz',pre)
        C.write_json(out/'diagnostics.json',dict(coordinates=x,fit=fit,audit=audit,
             report_sha256=info['report_meta']['sha256'],posterior=name!='w/o Post'))
        C.write_json(marker,dict(identity=identity,artifacts={file:C.data.digest(out/file) for file in
            ('prediction.npz','prediction_raw.npz','S5_prediction.npz','diagnostics.json')}))
    return C.evaluate(task,path,predictions,extra=extra)


