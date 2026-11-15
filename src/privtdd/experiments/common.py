"""Experiment identity, shared transcripts, subprocess execution, and summaries.

Production joint files are never edited. Budget aliases are scoped inside a
single worker process, and the complete design is bound to every cache identity.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import threading
import numpy as np
from privtdd import data
from privtdd.evaluation import metrics

HERE=Path(__file__).resolve().parent
ROOT=data.ROOT
DEFAULT_ALLOCATION=(.30,.35,.35)
SEED_BASE=20261028
METHODS=('LDP-GDD','TriTR-Hist','LDPGraph-TargetHist')


class ResourceMeter:
    """Absolute process RSS, sampled every 20 ms; graph preparation is outside scope."""
    def __enter__(self):
        import psutil
        self.process=psutil.Process();self.peak=self.process.memory_info().rss
        self.stop=threading.Event();self.start=time.perf_counter()
        def sample():
            while not self.stop.wait(.02):self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def __exit__(self,*unused):
        self.peak=max(self.peak,self.process.memory_info().rss)
        self.stop.set();self.thread.join();self.seconds=time.perf_counter()-self.start


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def convert(x):
        if isinstance(x,Path): return str(x)
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    text=json.dumps(value,ensure_ascii=False,indent=2,default=convert,allow_nan=False)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(text,encoding='utf-8');tmp.replace(path)


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def token(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def code_digest():
    from privtdd.io.identity import source_digest
    return source_digest('package')


def config(task, *, exact=False):
    from privtdd.experiments.ablation import fast_config
    from privtdd.config import PrivTDDConfig
    settings=dict(workers=task['workers'],blas_threads=task['blas_threads'],
                  geometry_draws=64 if task.get('smoke') else 65536,
                  geometry_replicates=1 if task.get('smoke') else 4)
    cfg=PrivTDDConfig(**settings) if exact else fast_config(**settings)
    if task.get('numeric'):
        cfg=replace(cfg,**task['numeric'])
    return cfg


@contextmanager
def allocation_context(allocation):
    """Only budget constants vary; every original S1/S2/S3/S4/S5/S6 uses the same split."""
    a,b,c=map(float,allocation)
    if min(a,b,c)<=0 or abs(a+b+c-1)>1e-12: raise ValueError('Positive relationship fractions summing to one required')
    modules=[importlib.import_module(name)
             for name in ('privtdd.config','privtdd.reporting.collection','privtdd.release.pipeline','privtdd.experiments.ablation')]
    saved=[]
    for mod in modules:
        for name,value in zip(('DEGREE_FRACTION','RR_FRACTION','RELEASE_FRACTION'),(a/2,b,c/2)):
            if hasattr(mod,name):
                saved.append((mod,name,getattr(mod,name)));setattr(mod,name,value)
    try: yield
    finally:
        for mod,name,value in saved:setattr(mod,name,value)


@contextmanager
def closure_context(cells):
    modules=[importlib.import_module(name)
             for name in ('privtdd.config','privtdd.base_distribution.construction')]
    old=[m.CLOSURE_CELLS for m in modules]
    try:
        for mod in modules:mod.CLOSURE_CELLS=int(cells)
        yield
    finally:
        for mod,value in zip(modules,old):mod.CLOSURE_CELLS=value


def graph_for(task):
    if 'synthetic' in task:
        spec=task['synthetic'];path=Path(task['root'])/'graphs'/token(spec)[:20]
        return data.synthetic(path,spec)
    return data.real_graph(task['dataset'],Path(task.get('data_root',data.default_data_root())))


def graph_id(path, root):
    """Hash arrays once, then reuse only while path/size/mtime_ns remain identical."""
    paths=[Path(path)/n for n in ('indptr.npy','indices.npy')]
    stats=[(str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
    file=Path(root)/'graph_identities'/(token(stats)+'.json')
    if file.exists():return read_json(file)['graph_sha256']
    value=token([data.digest(p) for p in paths]);write_json(file,dict(files=stats,graph_sha256=value))
    return value


def graph_arrays(path):
    return tuple(np.load(Path(path)/name,mmap_mode='r') for name in ('indptr.npy','indices.npy'))


def ensure_reports(task,path):
    from privtdd.reporting.collection import collect_reports_joint,prepare_graph
    from privtdd.io.memory import atomic_npz
    from privtdd.io.runner import output_lock
    ip,ix=graph_arrays(path);n=len(ip)-1
    if task.get('public_n',n)!=n:
        raise ValueError('Prepared graph node count differs from the allocation table')
    alloc=task.get('allocation',DEFAULT_ALLOCATION)
    design=dict(graph=graph_id(path,task['root']),n=n,D=n-1,epsilon=task['epsilon'],seed=task['seed'],
                allocation=list(alloc),mechanism=f'joint_shared_original_RR_cap{importlib.import_module("privtdd.config").CAP_OFFSET}_v1',
                collector_sha256=data.digest(Path(importlib.import_module('privtdd.reporting.collection').__file__)),
                primitives_sha256=data.digest(Path(importlib.import_module('privtdd.reporting.primitives').__file__)))
    directory=Path(task['root'])/'shared_reports'/token(design)
    with output_lock(directory,'REPORT.lock'):
        marker=directory/'complete.json'
        if marker.exists():
            meta=read_json(marker)
            if data.digest(directory/'reports.npz')!=meta['sha256']:raise ValueError('Transcript changed')
            return directory,meta
        started=time.perf_counter()
        ip,ix=prepare_graph((ip,ix),n,n-1)
        with ResourceMeter() as meter,allocation_context(alloc):
            reports=collect_reports_joint(ip,ix,n-1,task['epsilon'],task['seed'],workers=task['workers'])
        atomic_npz(directory/'reports.npz',**reports)
        seconds=time.perf_counter()-started
        degree=np.diff(ip);kept=np.minimum(degree,reports['caps'])
        wedges=degree.astype(float)*(degree-1)/2;retained=kept.astype(float)*(kept-1)/2
        meta=dict(design=design,sha256=data.digest(directory/'reports.npz'),collection_seconds=seconds,collection_peak_rss_bytes=meter.peak,
                  projected_node_fraction=float(np.mean(degree>reports['caps'])),
                  lost_wedge_fraction=float((wedges-retained).sum()/wedges.sum()) if wedges.sum() else 0.,
                  actual_wedge_visits=int(retained.sum()))
        write_json(marker,meta)
    return directory,meta


def ensure_joint(task,path,*,exact=False,fresh=None,load=True):
    from privtdd.io.runner import run_case
    from privtdd.experiments.ablation import load_shared
    from threadpoolctl import threadpool_limits
    reports,report_meta=ensure_reports(task,path)
    cfg=config(task,exact=exact)
    cells=task.get('closure_cells',65)
    binding=dict(report=report_meta['sha256'],report_design=report_meta['design'],config=asdict(cfg),
                 closure_cells=cells,code=task['code'])
    directory=Path(fresh) if fresh else Path(task['root'])/'shared_joint'/token(binding)
    n=report_meta['design']['n']
    case=dict(key=token(binding),dataset=task['dataset'],mode='reports',source=str(reports/'reports.npz'),
              input_files=[str(reports/'reports.npz')],public_cap=n-1,epsilon=task['epsilon'],seed=None,
              truth=None,degree_prior=None,output=str(directory))
    started=time.perf_counter()
    with allocation_context(task.get('allocation',DEFAULT_ALLOCATION)),closure_context(cells),threadpool_limits(limits=cfg.blas_threads):
        row=run_case(case,cfg,token(binding))
        # Loader expects a local transcript; create only after production identity is established.
        local=directory/'reports.npz'
        if not local.exists():
            import shutil
            shutil.copyfile(reports/'reports.npz',local)
        shared=load_shared(directory,cfg) if load else None
    return directory,shared,dict(report_meta=report_meta,reused=row['reused'],
         invocation_seconds=time.perf_counter()-started,postprocessing_seconds=row['postprocessing_seconds'],
         end_to_end_seconds=report_meta['collection_seconds']+row['postprocessing_seconds'])


def save_hist(path,hist):
    from privtdd.io.memory import atomic_npz
    k=np.array(sorted(hist),np.int64);c=np.array([hist[int(x)] for x in k],float)
    metrics.law(k,c)
    atomic_npz(path,keys=k,counts=c)


def load_hist(path):
    with np.load(path) as z:return z['keys'],z['counts']


def evaluate(task,path,predictions,*,extra=None):
    """Truth opens only after every candidate in this task has been saved."""
    for file in predictions.values():
        if not Path(file).is_file():raise FileNotFoundError(file)
    true=data.truth(path);tk,tc=np.unique(true,return_counts=True)
    truth_file=Path(path)/'triangles.npy';truth_digest=data.digest(truth_file)
    rows=[]
    for method,file in predictions.items():
        k,c=load_hist(file)
        if abs(c.sum()/len(true)-1)>1e-8 or np.any(k>(len(true)-1)*(len(true)-2)//2):
            raise ValueError('Prediction population/domain mismatch')
        values=metrics.compare(tk,tc,k,c,n=len(true),public_cap=len(true)-1)
        row={key:task[key] for key in ('dataset','epsilon','seed','panel','parameter','value','cluster','avg_degree','eta','n') if key in task}
        row.update(method=method,prediction=str(file),truth=str(truth_file),truth_sha256=truth_digest,**values)
        if extra:row.update(extra.get(method,{}))
        rows.append(row)
    return rows


def base_task(args,dataset,epsilon,seed,**kwargs):
    return dict(dataset=dataset,epsilon=float(epsilon),seed=int(seed),workers=args.workers,
                blas_threads=args.blas_threads,smoke=args.smoke,root=str(args.root.resolve()),
                data_root=str(args.data_root.resolve()),**kwargs)


def seeds(args,n): return list(range(args.seed_start,args.seed_start+(args.repeats or n)))


def parser(suite):
    p=argparse.ArgumentParser(description=f'{suite}: paper experiments; --plan is the default, --run executes.')
    p.add_argument('--root',type=Path,default=ROOT/'results/runs')
    p.add_argument('--data-root',type=Path,default=data.default_data_root(),help='Shared prepared real graphs, independent of result root')
    p.add_argument('--datasets',nargs='+',choices=data.DATASETS)
    p.add_argument('--epsilons',nargs='+',type=float)
    p.add_argument('--repeats',type=int)
    p.add_argument('--seed-start',type=int,default=SEED_BASE)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--blas-threads',type=int,default=2)
    p.add_argument('--plan',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--summarize',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--limit',type=int,help='Execute only the first N planned jobs; coverage remains partial')
    p.add_argument('--timeout',type=float,default=86400,help='Wall seconds per worker including setup; failures remain explicit')
    p.add_argument('--memory-gb',type=float,default=48,help='Worker RSS limit, recorded as memory_limit rather than OOM')
    p.add_argument('--continue-on-error',action='store_true');p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    return p


def write_csv(path,rows):
    fields=sorted(set().union(*(r.keys() for r in rows))) if rows else ['status']
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def validate_completion(directory,state):
    if data.digest(directory/'records.json')!=state['records_sha256']:raise ValueError('Changed records')
    for name,sha in state['artifacts'].items():
        if data.digest(directory/name)!=sha:raise ValueError(f'Changed artifact: {name}')
    for name,sha in state.get('external_artifacts',{}).items():
        if data.digest(name)!=sha:raise ValueError(f'Changed external artifact: {name}')


def summarize(folder):
    from collections import defaultdict
    folder=Path(folder);plan=read_json(folder/'plan.json');rows=[];states=[]
    for job in plan['jobs']:
        d=folder/'jobs'/job['id'];file=d/'status.json'
        state=read_json(file) if file.exists() else dict(status='pending')
        states.append(dict(id=job['id'],status=state['status'],dataset=job['dataset'],epsilon=job['epsilon'],
                           method=job.get('method','multiple'),panel=job.get('panel',''),seed=job['seed']))
        if state['status']=='complete':
            validate_completion(d,state)
            result=read_json(d/'records.json')
            rows.extend(result)
    write_csv(folder/'results.csv',rows);write_csv(folder/'coverage.csv',states)
    grouped=defaultdict(list)
    columns=('dataset','epsilon','method','panel','parameter','value','avg_degree','eta','n')
    for row in rows:grouped[tuple((k,row[k]) for k in columns if k in row)].append(row)
    summaries=[]
    measured=('W1','KS','TV','TV_bin','MSE','Recall_at_5','regret','end_to_end_seconds',
              'postprocessing_seconds','peak_rss_bytes','speedup','W1_delta','projected_node_fraction','lost_wedge_fraction',
              'cross_bucket_mass')
    for key,group in grouped.items():
        for metric in measured:
            values=[r for r in group if r.get(metric) is not None]
            if not values:continue
            if all('cluster' in r for r in values):
                clusters=defaultdict(list)
                for r in values:clusters[r['cluster']].append(r[metric])
                stats=metrics.hierarchical(list(clusters.values()))
            else:stats=metrics.bootstrap([r[metric] for r in values])
            if 'n' in stats:stats['publications']=stats.pop('n')
            summaries.append(dict(key,metric=metric,**stats))
    write_csv(folder/'summary.csv',summaries)
    # Pair only alternatives inside the same job with the same report hash.
    paired=[]
    for job in plan['jobs']:
        d=folder/'jobs'/job['id']
        if not (d/'records.json').exists() or not (d/'status.json').exists() or read_json(d/'status.json')['status']!='complete':continue
        group=read_json(d/'records.json');base=next((r for r in group if r['method']=='Full'),None)
        if base is None or not base.get('report_sha256'):continue
        for r in group:
            if r['method']=='Full' or r.get('report_sha256')!=base['report_sha256']:continue
            for m in ('W1','KS','TV','TV_bin'):
                if m in r and m in base:paired.append(dict(dataset=r['dataset'],epsilon=r['epsilon'],seed=r['seed'],method=r['method'],metric=m,delta=r[m]-base[m]))
    write_csv(folder/'paired_deltas.csv',paired)
    pg=defaultdict(list)
    for r in paired:pg[(r['dataset'],r['epsilon'],r['method'],r['metric'])].append(r['delta'])
    write_csv(folder/'paired_summary.csv',[dict(zip(('dataset','epsilon','method','metric'),key),**metrics.bootstrap(v)) for key,v in pg.items()])
    write_json(folder/'coverage.json',dict(expected=len(states),completed=sum(r['status']=='complete' for r in states),
               states={s:sum(r['status']==s for r in states) for s in sorted(set(r['status'] for r in states))},
               smoke=plan['smoke'],bootstrap_draws=2000))
    if summaries:
        # Tables are emitted here; paper plots use the explicit reproduction pipeline.
        pass
    return rows,summaries


def run_cli(suite,make_tasks,execute,argv=None,configure=None):
    p=parser(suite)
    if configure:configure(p)
    args=p.parse_args(argv)
    if args.worker:
        task=read_json(args.worker);directory=args.worker.parent
        try:
            if code_digest()!=task['code']:raise ValueError('Code changed after plan; use a new root')
            rows=execute(task,directory)
            write_json(directory/'records.json',rows)
            artifacts={str(f.relative_to(directory)):data.digest(f) for f in directory.rglob('*.npz')}
            external={r['prediction']:data.digest(r['prediction']) for r in rows}
            external.update({r['truth']:r['truth_sha256'] for r in rows})
            for row in rows:
                report=Path(row['prediction']).parent/'reports.npz'
                if report.exists():external[str(report)]=data.digest(report)
                for field in ('baseline_manifest','synthesis_manifest','synthetic_graph'):
                    if row.get(field):external[row[field]]=data.digest(row[field])
            provenance=directory/'baseline_provenance.json'
            if provenance.exists():artifacts[provenance.name]=data.digest(provenance)
            write_json(directory/'status.json',dict(status='complete',artifacts=artifacts,external_artifacts=external,
                       records_sha256=data.digest(directory/'records.json')))
        except BaseException as error:
            write_json(directory/'status.json',dict(status='oom' if isinstance(error,MemoryError) else 'failed',
                       error=repr(error),traceback=traceback.format_exc()))
            raise
        return
    if args.smoke:
        args.root=args.root.with_name(args.root.name+'_smoke')
        args.workers=min(args.workers,2);args.blas_threads=1
    if args.repeats is not None and args.repeats<1:p.error('repeats must be positive')
    if args.limit is not None and args.limit<1:p.error('limit must be positive')
    if args.timeout<=0 or args.memory_gb<=0:p.error('positive resource limits required')
    folder=args.root/suite
    if args.summarize:
        summarize(folder);return
    tasks=make_tasks(args)
    if tasks is None:return
    code=code_digest()
    for t in tasks:
        t['code']=code;t['suite']=suite;t['id']=token(t)[:24]
    if len({t['id'] for t in tasks})!=len(tasks):raise ValueError('Duplicate jobs cannot count as repetitions')
    plan=dict(suite=suite,code=code,smoke=args.smoke,jobs=tasks,timeout=args.timeout,memory_gb=args.memory_gb)
    if hasattr(args,'design_summary'):plan['design_summary']=args.design_summary
    if not args.run:
        print(json.dumps(dict(suite=suite,jobs=len(tasks),root=str(args.root),smoke=args.smoke,
             design_summary=getattr(args,'design_summary',None),
             first_jobs=tasks[:2],command_hint='Add --run to execute; --limit 1 for a full-precision pilot'),indent=2));return
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'plan.json').exists():
        old=read_json(folder/'plan.json')
        design=lambda x:{k:v for k,v in x.items() if k not in ('timeout','memory_gb')}
        if design(old)!=design(plan):raise ValueError('Experimental design/code changed; use another --root')
        if old!=plan:
            history=folder/'resource_policy_history.json'
            records=read_json(history) if history.exists() else []
            records.append(dict(timestamp=time.time(),old={k:old[k] for k in ('timeout','memory_gb')},
                                new={k:plan[k] for k in ('timeout','memory_gb')}))
            write_json(history,records)
    write_json(folder/'plan.json',plan)
    import psutil
    from privtdd.io.runner import output_lock
    failures=0
    with output_lock(folder,'SUITE.lock'):
        for i,task in enumerate(tasks[:args.limit] if args.limit else tasks):
            directory=folder/'jobs'/task['id'];directory.mkdir(parents=True,exist_ok=True)
            state=directory/'status.json'
            if state.exists():
                old=read_json(state)
                if old['status']=='complete':
                    validate_completion(directory,old)
                    continue
                if not args.retry_failed:
                    failures+=1
                    if not args.continue_on_error:break
                    continue
            write_json(directory/'task.json',task)
            print(f'{suite} {i+1}/{len(tasks)} {task["dataset"]} eps={task["epsilon"]} seed={task["seed"]}',flush=True)
            started=time.monotonic();peak=0;stop_reason=None
            with (directory/'worker.log').open('a',encoding='utf-8') as log:
                proc=subprocess.Popen([sys.executable,'-u','-m',f'privtdd.experiments.{suite}',
                    '--worker',str(directory/'task.json')],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                monitor=psutil.Process(proc.pid)
                while proc.poll() is None:
                    try:rss=monitor.memory_info().rss;peak=max(peak,rss)
                    except psutil.NoSuchProcess:break
                    if rss>args.memory_gb*1024**3:stop_reason='memory_limit'
                    elif time.monotonic()-started>args.timeout:stop_reason='timeout'
                    if stop_reason:
                        proc.terminate();proc.wait();break
                    time.sleep(.25)
                proc.wait()
            # Only clear locks owned by this confirmed-exited worker, within this result root.
            if proc.returncode:
                for lock in args.root.rglob('*.lock'):
                    if lock.read_text(encoding='utf-8').strip()==str(proc.pid):lock.unlink()
                for lock in args.root.rglob('*.LOCK'):
                    if lock.read_text(encoding='utf-8').strip()==str(proc.pid):lock.unlink()
            cost=dict(worker_wall_seconds=time.monotonic()-started,worker_peak_rss_bytes=peak,
                      timeout_scope='entire worker including graph setup and evaluation',exit_code=proc.returncode)
            write_json(directory/'worker_cost.json',cost)
            if stop_reason or proc.returncode or not state.exists():
                failures+=1
                if stop_reason or not state.exists():write_json(state,dict(status=stop_reason or 'failed',**cost))
                if not args.continue_on_error:break
        summarize(folder)
    if failures:raise SystemExit(1)
