"""Command-line entry points for verification and three reproduction levels."""
from __future__ import annotations
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys


def artifact_root(explicit=None):
    if explicit:
        root=Path(explicit).expanduser().resolve()
    else:
        from .data import artifact_root as locate
        root=locate()
    if not (root/'reproduction/manifest.json').is_file():
        raise FileNotFoundError('Use --repo-root to select the artifact checkout containing reproduction/manifest.json')
    os.environ['PRIVTDD_ARTIFACT_ROOT']=str(root)
    return root


def doctor(root, data_root=None, external_source=None, verify_data=False):
    from .experiments.reproduce import verify_snapshot
    from .data import manifest,verify
    from .budget import verify_allocations
    packages={}
    for name in ('numpy','scipy','numba','threadpoolctl','pandas','matplotlib','psutil','filelock','networkx'):
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]='MISSING'
    data_root=Path(data_root or root/'datasets')
    state={}
    for dataset in manifest()['datasets']:
        path=data_root/'processed'/dataset
        present=all((path/f).is_file() for f in ('indptr.npy','indices.npy','triangles.npy','graph.json'))
        state[dataset]={'present':present}
        if present and verify_data:
            state[dataset]['verification']=verify(dataset,data_root)
    baseline={'provided':bool(external_source),'required_for':'fresh LDPGraph-T/S execution only'}
    if external_source:
        from .baselines.external import validate_source
        baseline['verification']=validate_source(external_source)
    return dict(root=str(root),python=sys.version,packages=packages,data=state,baseline=baseline,
                snapshot=verify_snapshot(root),budgets=verify_allocations(root/'configs/paper/allocations.json'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path)
    sub=parser.add_subparsers(dest='command',required=True)
    check=sub.add_parser('doctor',help='Check dependencies, published evidence, and paper allocations')
    check.add_argument('--data-root',type=Path);check.add_argument('--external-code-root',type=Path)
    check.add_argument('--verify-data',action='store_true')
    datasets=sub.add_parser('data',help='Prepare or verify the four paper datasets')
    datasets.add_argument('action',choices=['prepare','verify'])
    datasets.add_argument('--datasets',nargs='+',default=['bio-dmela','dblp','higgs','orkut'])
    datasets.add_argument('--data-root',type=Path)
    budget=sub.add_parser('budget',help='Derive an allocation from public n and epsilon')
    budget.add_argument('--n',type=int,required=True);budget.add_argument('--epsilon',type=float,required=True)
    reproduce=sub.add_parser('reproduce',help='Rebuild frozen figures, run a smoke case, or execute fresh experiments')
    reproduce.add_argument('--mode',choices=['figures','smoke','full'],required=True)
    reproduce.add_argument('--suite',choices=['all','E1','E2','E3','E4'],default='all')
    reproduce.add_argument('--output',type=Path)
    reproduce.add_argument('--data-root',type=Path)
    reproduce.add_argument('--external-code-root',type=Path)
    reproduce.add_argument('--methods',nargs='+',choices=['PrivTDD','TriTR-Hist','LDPGraph-T','LDPGraph-S'])
    reproduce.add_argument('--plan',action='store_true')
    reproduce.add_argument('--retry-failed',action='store_true')
    reproduce.add_argument('--compile-supplement',action='store_true')
    reproduce.add_argument('--workers',type=int,default=8)
    reproduce.add_argument('--memory-gb',type=float,default=16.)
    args=parser.parse_args(argv)
    if args.command=='budget':
        from .budget import allocate_budget
        print(json.dumps(allocate_budget(args.n,args.epsilon),indent=2));return 0
    root=artifact_root(args.repo_root)
    if args.command=='doctor':
        result=doctor(root,args.data_root,args.external_code_root,args.verify_data)
        print(json.dumps(result,indent=2,default=str))
        bad_data=args.verify_data and any(not item['present'] or not item.get('verification',{}).get('ok',False) for item in result['data'].values())
        return int(any(v=='MISSING' for v in result['packages'].values()) or bad_data)
    if args.command=='data':
        from .data import prepare,verify
        data_root=args.data_root or root/'datasets'
        results={ds:prepare(ds,data_root,allow_download=True) if args.action=='prepare' else verify(ds,data_root) for ds in args.datasets}
        print(json.dumps(results,indent=2,default=str))
        return int(args.action=='verify' and any(not result['ok'] for result in results.values()))
    if args.mode=='figures':
        from .experiments.reproduce import render_all
        from .experiments.supplement import build_supplement
        output=args.output or root/'results/reproduced/paper-v1'
        result=render_all(root,output)
        result['supplement_source']=str(build_supplement(root,output,Path(output)/'supplement',args.compile_supplement))
        print(json.dumps(result,indent=2));return 0

    from .experiments import forpaper
    from .experiments.fresh_render import render_fresh_runs
    smoke=args.mode=='smoke'
    output=(args.output or root/'results/runs'/('quick' if smoke else 'paper')).resolve()
    effective_output=output.with_name(output.name+'_smoke') if smoke else output
    if output.is_relative_to(root/'results/published'):
        raise ValueError('Fresh runs cannot write into the immutable published snapshot')
    suites=['E1','E2','E3'] if args.suite=='all' else ([] if args.suite=='E4' else [args.suite])
    methods=args.methods or (['PrivTDD'] if smoke else ['PrivTDD','TriTR-Hist','LDPGraph-T','LDPGraph-S'])
    if not args.plan and any('LDPGraph' in m for m in methods) and 'E1' in suites:
        if not args.external_code_root:
            raise ValueError('Full LDPGraph reproduction needs --external-code-root pointing to the pinned upstream checkout; see THIRD_PARTY.md. Use --methods PrivTDD for an explicit subset.')
        from .baselines.external import validate_source
        validate_source(args.external_code_root)
    run_args=['--plan' if args.plan else '--run','--root',str(output),'--suites',*suites,
              '--methods',*methods,'--data-root',str(args.data_root or root/'datasets'),
              '--workers',str(1 if smoke else args.workers),'--memory-gb',str(args.memory_gb),'--continue-on-error']
    if smoke:run_args+=['--smoke','--skip-extra']
    if args.external_code_root:run_args+=['--external-code-root',str(args.external_code_root)]
    if args.retry_failed:run_args+=['--retry-failed']
    failed=0
    if suites:
        try:forpaper.main(run_args)
        except SystemExit as error:
            if error.code not in (0,1,None):raise
            failed=int(bool(error.code))
    e4_output=effective_output.with_name(effective_output.name+'_e4')
    if args.suite in ('all','E4'):
        if args.plan:
            print(json.dumps({'E4':{'datasets':['higgs','orkut'],'epsilon':2,'rates':[.0001,.001,.01,.1],
                'seeds':list(range(20261028,20261033)),'sampled_jobs':40,
                'requires':'completed E1 parent in --output; paper table full controls use ten runs'}},indent=2))
        else:
            from .sampling.run import main as sample
            sample_args=['--run','--root',str(e4_output),'--history-root',str(effective_output)]
            if smoke:sample_args+=['--smoke']
            if args.retry_failed:sample_args+=['--retry-failed']
            code=sample(sample_args)
            failed=max(failed,int(bool(code)))
    if not args.plan:
        result=render_fresh_runs(effective_output,sampling_root=e4_output if e4_output.exists() else None)
        print(json.dumps(result,indent=2,default=str))
    return failed
