"""Audited GDD adapters; LDPGraph retains the author's squared consistency weight.

The original repository is immutable. Shared RR is implemented by the existing
pair-indexed simulator, fixing the author's `== True` cache assignment bug.
The only transmitted second-round statistic is the single centered noisy count
used in source code (not the two-noise pseudocode in the paper).
"""
from __future__ import annotations
import ast
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from numba import njit, prange, set_num_threads, config as numba_config
from privtdd.experiments import common as C
from privtdd import data

from .external import source_directory, validate_source
VENDOR_COMMIT='b903ddc7d93aa85dae951c536de3eededad69c5d'


@njit(cache=True,parallel=True)
def consistency(d_est,t_est,eps_degree,eps_triangle,n):
    """Same discrete minimum and tie order as source, bounded to public d<n.

For an infeasible pair, the objective on t=d(d-1)/2 has one stationary
minimum to the right of d_est. Bisection plus neighboring integers avoids
the source's potentially very long integer enumeration. No weight changes.
"""
    ds=np.empty(len(d_est),np.int64);ts=np.empty(len(d_est),np.int64)
    for i in prange(len(ds)):
        a=min(n-1,max(1,int(d_est[i])));t=min((n-1)*(n-2)//2,max(0,int(t_est[i])))
        if a*(a-1)//2>=t:ds[i]=a;ts[i]=t;continue
        upper=min(n-1,int(math.ceil((1+math.sqrt(1+8*t))/2)))
        weight=(a*eps_degree/eps_triangle)**2
        lo=float(a);hi=float(upper)
        for _ in range(64):
            x=(lo+hi)/2
            slope=2*weight*(x-a)+(x*(x-1)/2-t)*(2*x-1)
            if slope<0:lo=x
            else:hi=x
        best_d=upper;best_t=min(t,upper*(upper-1)//2)
        best=weight*(best_d-a)**2+(best_t-t)**2
        # Author initializes upper, then scans ascending and changes only on strict improvement.
        candidates=(a,max(a,min(upper,int(math.floor((lo+hi)/2)))),max(a,min(upper,int(math.ceil((lo+hi)/2)))))
        for d in candidates:
            tr=min(t,d*(d-1)//2);cost=weight*(d-a)**2+(tr-t)**2
            if cost<best:best=cost;best_d=d;best_t=tr
        ds[i]=best_d;ts[i]=best_t
    return ds,ts


def source_thresholds(degree,epsilon,seed):
    """Source ratios .2:.4:.4, rescaled from local-row to 2e1+e2+2e3=epsilon."""
    n=len(degree);e1,e2,e3=np.array([.2,.4,.4])*epsilon/1.6
    rng=np.random.default_rng(seed)
    noisy=np.clip(np.rint(np.maximum(1.,degree+rng.laplace(0,1/e1,n))),1,n-1).astype(np.int64)
    max_tau=int(math.log(5)/e1)
    tau=np.full(n,max_tau,np.int64)
    def objective(x):return (noisy-.5)**2/e1**2*np.exp(-x*e1)+2/e3**2*(noisy**2+2/e1**2+2*noisy*x+x*x)
    best=objective(max_tau)
    for j in range(max_tau):
        value=objective(j);mask=value<best;best[mask]=value[mask];tau[mask]=j
    return noisy,np.minimum(noisy+tau,n-1),rng,(float(e1),float(e2),float(e3))


def ldpgraph_collect(ip,ix,epsilon,seed,workers):
    from privtdd.baselines.tritr_hist import project_csr
    from privtdd.reporting.primitives import _exact_local_wedge_reports_parallel_numba
    n=len(ip)-1
    degree,caps,rng,(e1,e2,e3)=source_thresholds(np.diff(ip),epsilon,seed)
    p=1/(1+math.exp(-e2));q=1-p;gap=p-q
    set_num_threads(min(workers,numba_config.NUMBA_NUM_THREADS))
    secret=rng.integers(0,2**63,dtype=np.uint64)
    ptr,idx=project_csr(ip,ix,caps,secret)
    rr_secret=int(rng.integers(0,2**63,dtype=np.int64))
    B,W,_,_=_exact_local_wedge_reports_parallel_numba(ip,ix,ptr,idx,p,q,rr_secret)
    # Source int() truncation, lower clipping, and squared consistency weighting are retained.
    noisy=(B-q*W+rng.laplace(0,1,n)*caps/e3)/gap
    triangle=np.clip(np.trunc(noisy),0,(n-1)*(n-2)//2).astype(np.int64)
    ds,ts=consistency(degree,triangle,e1,e3,n)
    meta=dict(method='LDPGraph source-consistency relationship adaptation',vendor_commit=VENDOR_COMMIT,
              eps_degree=e1,eps_rr=e2,eps_triangle=e3,relationship_cost=2*e1+e2+2*e3,
              delta=0.,rr_input='original unordered edges; shared bits',
              consistency_weight='(noisy_degree*eps_degree/eps_triangle)^2',
              changes=['shared RR cache fixed','stable-priority uniform neighbor projection',
                       'relationship budget rescaled','public n-1 domain enforced','int64 targets',
                       'discrete source objective solved by stationary-point search'],
              actual_wedge_visits=int(W.sum()),projected_node_fraction=float(np.mean(np.diff(ip)>caps)))
    return dict(noisy_degree=degree,caps=caps,noisy_triangle=triangle,target_degree=ds,target_triangle=ts),meta


def synthesis_function():
    """Extract source synthesis statements, remove only the redundant dense matrix.

All grouping, degree-demand checks, within-hash and cross-hash edge requests,
the 100-iteration limit and .001 stopping rule remain from the author's code.
The original files are not patched or imported with evaluation side effects.
"""
    VENDOR=source_directory()
    validate_source(VENDOR)
    source=ast.parse((VENDOR/'LDPGraph_main.py').read_text(encoding='utf-8-sig'))
    fn=next(x for x in source.body if isinstance(x,ast.FunctionDef) and x.name=='LDPGraphSyn')
    loop=next(x for x in fn.body if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='ei')
    release=next(x for x in loop.body if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='exper')
    start=next(i for i,x in enumerate(release.body) if isinstance(x,ast.ClassDef) and x.name=='MyCommunity')
    end=next(i for i,x in enumerate(release.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='t6_cost' for t in x.targets))
    class RemoveDense(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(x,ast.Name) and x.id=='mat2' for target in node.targets for x in ast.walk(target)):
                return None
            return self.generic_visit(node)
    body=[RemoveDense().visit(x) for x in release.body[start:end]]
    body=[x for x in body if x is not None]
    # Avoid division by zero once degree demand is completely satisfied.
    for node in ast.walk(ast.Module(body=body,type_ignores=[])):
        if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='super_iter':
            node.body.insert(0,ast.parse('if all(EdgesDemand(u) <= 0 for u in range(new_graph_node_num)): break').body[0])
    new=ast.FunctionDef(name='synthesize',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in ('new_node_degrees','new_node_triangles','seed')],kwonlyargs=[],kw_defaults=[],defaults=[]),
        body=ast.parse('random.seed(seed)\nnp.random.seed(seed % (2**32))\nnew_graph_node_num=len(new_node_degrees)').body+body+[ast.Return(value=ast.Name(id='newG',ctx=ast.Load()))],decorator_list=[])
    util=ast.parse((VENDOR/'utils.py').read_text(encoding='utf-8-sig'))
    fenwick=next(x for x in util.body if isinstance(x,ast.ClassDef) and x.name=='FenwickTree')
    namespace=dict(np=np,math=math,random=random,time=time,defaultdict=defaultdict)
    import networkx as nx
    namespace['nx']=nx
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fenwick,new],type_ignores=[])),str(VENDOR/'LDPGraph_main.py'),'exec'),namespace)
    return namespace['synthesize']


def _run_reports(task,path,method,*,synth=False,fresh=None):
    from privtdd.io.runner import output_lock
    from privtdd.io.memory import atomic_npz
    ip,ix=C.graph_arrays(path);n=len(ip)-1
    seed=int(task['seed'])+(70000001 if method=='TriTR-Hist' else 90000001)
    binding=dict(graph=C.graph_id(path,task['root']),method=method,epsilon=task['epsilon'],seed=seed,
                 D=n-1,code=task['code'])
    directory=Path(fresh) if fresh else Path(task['root'])/'shared_baselines'/C.token(binding)
    with output_lock(directory,'BASELINE.lock'):
        marker=directory/'complete.json'
        if marker.exists():
            meta=C.read_json(marker)
            if meta['binding'] != binding:
                raise ValueError('Baseline task differs from saved binding')
            for name,sha in meta['artifacts'].items():
                if data.digest(directory/name)!=sha:raise ValueError('Changed baseline artifact')
        else:
            started=time.perf_counter()
            if method=='TriTR-Hist':
                from privtdd.baselines.tritr_hist import run_csr
                set_num_threads(min(task['workers'],numba_config.NUMBA_NUM_THREADS))
                hist,diag=run_csr(ip,ix,n-1,task['epsilon'],'tritr_gdd',seed)
                C.save_hist(directory/'prediction.npz',hist)
            elif method=='LDPGraph-TargetHist':
                reports,diag=ldpgraph_collect(ip,ix,task['epsilon'],seed,task['workers'])
                atomic_npz(directory/'reports.npz',**reports)
                keys,counts=np.unique(reports['target_triangle'],return_counts=True)
                atomic_npz(directory/'prediction.npz',keys=keys,counts=counts)
            else:raise ValueError(method)
            meta=dict(binding=binding,diagnostics=diag,seconds=time.perf_counter()-started,
                      artifacts={f.name:data.digest(f) for f in directory.glob('*.npz')})
            C.write_json(marker,meta)
        if synth:
            sm=directory/'synthesis_complete.json'
            if sm.exists():
                for name,sha in C.read_json(sm)['artifacts'].items():
                    if data.digest(directory/name)!=sha:raise ValueError('Changed synthesis artifact')
            else:
                started=time.perf_counter()
                with np.load(directory/'reports.npz') as z:degrees,triangles=z['target_degree'],z['target_triangle']
                graph=synthesis_function()(degrees,triangles,seed+101)
                edges=np.asarray(list(graph.edges()),np.int64).reshape(-1,2)
                sp,si=data.csr_from_edges(n,edges[:,0],edges[:,1])
                atomic_npz(directory/'synthetic_graph.npz',indptr=sp,indices=si)
                from privtdd.data.triangles import validate_and_triangles
                values=validate_and_triangles(sp,si);keys,counts=np.unique(values,return_counts=True)
                atomic_npz(directory/'synth_prediction.npz',keys=keys,counts=counts)
                C.write_json(sm,dict(seconds=time.perf_counter()-started,
                    source_report_sha256=data.digest(directory/'reports.npz'),
                    artifacts={f:data.digest(directory/f) for f in ('synthetic_graph.npz','synth_prediction.npz')},
                    remaining_degree_deficit=int(np.maximum(degrees-np.diff(sp),0).sum()),
                    changes=['redundant dense adjacency removed','stop on zero remaining demand']))
    return directory,meta


def run(task, path, method, *, synth=False, fresh=None):
    """Run the paper adapter; synthesis always uses the compact graph backend."""
    if synth:
        validate_source(source_directory(task.get('ldpgraph_source')))
        if task.get('ldpgraph_source'):
            from .external import configure_source
            configure_source(task['ldpgraph_source'])
        from .synthesis import run as synthesize_run
        return synthesize_run(_run_reports, task, path, method, synth=True, fresh=fresh)
    return _run_reports(task, path, method, synth=False, fresh=fresh)
