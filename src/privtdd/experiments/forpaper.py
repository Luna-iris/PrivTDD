"""Fresh, resumable E1--E3 execution in the standalone package.

All outputs are new runs with a new source identity. Published records are never
used as working caches. E4 is executed by privtdd.sampling.run using these E1
controls. Internal protocol keys retain the archived stream definitions; display
names are recorded separately as canonical_method.
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
from . import common as C, components, paper_variants
from privtdd import data

DATASETS=('bio-dmela','dblp','higgs','orkut')
EPSILONS=(.5,1.,1.5,2.,2.5,3.,3.5,4.)
REPEATS={'bio-dmela':20,'dblp':20,'higgs':10,'orkut':10}
NAMES={'LDP-GDD':'PrivTDD','TriTR-Hist':'TriTR-Hist','LDPGraph-TargetHist':'LDPGraph-T','LDPGraph-SynthHist':'LDPGraph-S'}
ALLOCATION_FILE=Path(__file__).with_name('allocations.json')


def allocations(file=ALLOCATION_FILE):
    table=C.read_json(file)
    if table['applied_L']!=0 or table['safeguard_tau']!=1:
        raise ValueError('Paper profile requires L=0 and safeguard tau=1')
    result={}
    for row in table['rows']:
        ds,e=row['dataset'],row['epsilon']
        parts=[row[k] for k in ('eps_deg','eps_rr','eps_rel')]
        if (row.get('L_for_allocation')!=0 or min(parts)<=0 or not all(math.isfinite(x) for x in parts)
            or not math.isclose(2*parts[0]+parts[1]+2*parts[2],e,rel_tol=1e-12,abs_tol=1e-14)
            or row['D']!=row['n']-1 or (ds,e) in result):
            raise ValueError('Invalid or duplicate frozen allocation row')
        result[ds,e]=dict(row,allocation=[2*parts[0]/e,parts[1]/e,2*parts[2]/e])
    return result


def configure(parser):
    parser.set_defaults(root=data.ROOT/'results/runs/paper',memory_gb=16.)
    parser.add_argument('--allocation-table',type=Path,default=ALLOCATION_FILE)
    parser.add_argument('--suites',nargs='+',choices=['E1','E2','E3'],default=['E1','E2','E3'])
    parser.add_argument('--methods',nargs='+',choices=list(NAMES.values()),default=list(NAMES.values()))
    parser.add_argument('--skip-extra',action='store_true',help='Skip additional E2 supplement ablations (partial scope)')
    parser.add_argument('--external-code-root',type=Path,default=None,help='Explicit upstream LDPGraph checkout')


def make_tasks(args):
    from privtdd.config import CAP_OFFSET
    if CAP_OFFSET!=0:raise ValueError('Paper reporting mechanism requires L=0')
    table=allocations(args.allocation_table)
    datasets=args.datasets or (['bio-dmela'] if args.smoke else list(DATASETS))
    budgets=args.epsilons or ([2.] if args.smoke else list(EPSILONS))
    if any(len(v)!=len(set(v)) for v in (datasets,budgets,args.suites,args.methods)):
        raise ValueError('Duplicate datasets, budgets, suites, or methods')
    names={v:k for k,v in NAMES.items()}
    extra_file=Path(__file__).with_name('extra_cells.json')
    extra_rows=C.read_json(extra_file) if extra_file.exists() else []
    extra={(r['dataset'],float(r['epsilon']),int(r['seed'])):r['variants'] for r in extra_rows}
    stages={name:[] for name in ('E1','E2','E2-extra','E3')}
    for ds in datasets:
        for epsilon in budgets:
            row=table[ds,epsilon]
            for seed in C.seeds(args,1 if args.smoke else REPEATS[ds]):
                base=C.base_task(args,ds,epsilon,seed,allocation=row['allocation'],allocation_record=row,
                    allocation_table_sha256=data.digest(args.allocation_table),public_n=24 if args.smoke else row['n'],
                    numeric=dict(safeguard=True,safeguard_tau=1.))
                if args.external_code_root:base['external_code_root']=str(args.external_code_root.resolve())
                if args.smoke:base['synthetic']=dict(family='circle',n=24,avg_degree=6.,eta=2.,graph_seed=712)
                if 'E1' in args.suites:
                    for name in args.methods:
                        stages['E1'].append(dict(base,panel='main',method=names[name],canonical_method=name))
                if 'E2' in args.suites:
                    stages['E2'].append(dict(base,panel='paper_e2'))
                    selected=extra.get((ds,float(epsilon),int(seed)),[])
                    if args.smoke:selected=['w/o Beta reference','w/o Geo reference','Mass-only correction','Shape-only correction','GlobalMix','w/o Post','w/o safeguard']
                    if selected and not args.skip_extra:
                        stages['E2-extra'].append(dict(base,panel='ablation',variants=['Full',*selected],include_guard_ablation=True))
                if ('E3' in args.suites and epsilon in (.5,2.) and seed-args.seed_start < (1 if args.smoke else 10)):
                    stages['E3'].append(dict(base,panel='paper_e3'))
    tasks=[task for group in stages.values() for task in group]
    args.design_summary=dict(datasets=datasets,budgets=budgets,stages={k:len(v) for k,v in stages.items()},
        seed_start=args.seed_start,source='new standalone execution',L=0,safeguard_tau=1.,
        configuration='paper fast grid512 / report bins1/16 / geometry65536x4 (smoke uses64x1)',
        smoke=args.smoke,external_baselines_required=any('LDPGraph' in m for m in args.methods),
        E4='Run privtdd.sampling.run with --history-root pointing to this result root after E1')
    return tasks


def _postprocessing(task,directory):
    path=C.graph_for(task)
    reference,_,info=C.ensure_joint(task,path,load=False)
    row,=C.evaluate(task,path,{'PrivTDD':reference/'prediction.npz'})
    full=paper_variants.record(row,'new_full',info['report_meta']['sha256'])
    names=['screc',*paper_variants.FIXED] if task['panel']=='paper_e2' else list(paper_variants.BOUNDARIES)
    predictions={'PrivTDD' if task['panel']=='paper_e2' else '(0.90,0.95)':reference/'prediction.npz'}
    extra={name:dict(report_sha256=info['report_meta']['sha256']) for name in predictions}
    shared=[]
    labels={'screc':'SCRec','fixed25':'Fixed-0.25','fixed50':'Fixed-0.50','fixed75':'Fixed-0.75','q80_90':'(0.80,0.90)','q95_99':'(0.95,0.99)'}
    for name in names:
        out=directory/name;out.mkdir(parents=True,exist_ok=True)
        marker=out/'complete.json'
        identity=C.token(dict(task=task['id'],variant=name,report=full['report_sha256']))
        if marker.exists():
            result=C.read_json(marker)
            if result['identity']!=identity or data.digest(result['prediction'])!=result['prediction_sha256']:
                raise ValueError('Changed completed postprocessing output')
        else:
            result=paper_variants.variant_result(task,name,out,full,shared,None)
            result=dict(result,identity=identity)
            C.write_json(marker,result)
        predictions[labels[name]]=Path(result['prediction'])
        extra[labels[name]]=dict(report_sha256=full['report_sha256'],origin=result['origin'])
    return C.evaluate(task,path,predictions,extra=extra)


def execute(task,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if task['panel'] in ('paper_e2','paper_e3'):
        rows=_postprocessing(task,directory)
    elif task['panel']=='ablation':
        rows=components.execute(task,directory)
    else:
        path=C.graph_for(task);method=task['method']
        if method=='LDP-GDD':
            out,_,info=C.ensure_joint(task,path,load=False)
            prediction=out/'prediction.npz'
            guard=C.read_json(out/'work/result.json')['safeguard']
            extra=dict(report_sha256=info['report_meta']['sha256'],alpha=guard['alpha'],safeguard_triggered=guard['triggered'],reused=info['reused'])
        else:
            from privtdd import baselines
            from privtdd.baselines import synthesis
            synth=method=='LDPGraph-SynthHist'
            protocol='LDPGraph-TargetHist' if synth else method
            out,info=synthesis.run(baselines.run,task,path,protocol,synth=synth)
            prediction=out/('synth_prediction.npz' if synth else 'prediction.npz')
            extra=dict(protocol_seed=task['seed']+(70000001 if protocol=='TriTR-Hist' else 90000001),baseline_manifest=str(out/'complete.json'))
            if synth:extra.update(synthesis_manifest=str(out/'synthesis_complete.json'),synthetic_graph=str(out/'synthetic_graph.npz'))
            if (out/'reports.npz').exists():extra['report_sha256']=data.digest(out/'reports.npz')
            C.write_json(directory/'baseline_provenance.json',info)
        rows=C.evaluate(task,path,{method:prediction},extra={method:extra})
    for row in rows:
        row['canonical_method']=NAMES.get(row['method'],row['method'])
        row['allocation_table_sha256']=task['allocation_table_sha256']
        if task['panel']!='main' or row['method']=='LDP-GDD':
            row.update(L=0,eps_deg=task['allocation_record']['eps_deg'],eps_rr=task['allocation_record']['eps_rr'],eps_rel=task['allocation_record']['eps_rel'])
    return rows


def main(argv=None):
    C.run_cli('forpaper',make_tasks,execute,argv=argv,configure=configure)


if __name__=='__main__':main()
