"""Original public synthetic graph families, preserving the random stream."""
from pathlib import Path
import json
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
from . import csr_from_edges, save_graph

def synthetic(path, spec):
    """Independent edges for logistic families; sparse G(n,p) for scalability."""
    path = Path(path)
    if (path/'graph.json').exists():
        existing=json.loads((path/'graph.json').read_text())
        if existing['spec'] != spec: raise ValueError('Synthetic graph identity mismatch')
        return path
    n=int(spec['n']); avg=float(spec['avg_degree']); rng=np.random.default_rng(spec['graph_seed'])
    family=spec['family']
    if family == 'er':
        total=n*(n-1)//2; p=avg/(n-1); chunks=[]; last=-1
        while last < total:
            ids=last+np.cumsum(rng.geometric(p,size=min(1000000,max(1024,int(n*avg/2)))))
            chunks.append(ids[ids<total]); last=int(ids[-1])
        k=np.concatenate(chunks)
        # Lower-triangle packed index k = v(v-1)/2 + u, u<v.
        v=np.floor((1+np.sqrt(1+8*k))/2).astype(np.int64)
        u=k-v*(v-1)//2
    elif family == 'small_world':
        import networkx as nx
        k=int(round(avg/2))*2
        g=nx.watts_strogatz_graph(n,k,float(spec.get('rewire',.1)),seed=int(spec['graph_seed']))
        edges=np.asarray(list(g.edges()),np.int64); u,v=edges.T
    else:
        if n > 20000: raise ValueError('Dense pair enumeration is restricted to mechanism/utility graphs')
        u,v=np.triu_indices(n,1)
        # Degree rank and spatial/block labels are independently randomized.
        rank=rng.permutation(np.arange(1,n+1))
        theta=2*np.pi*rng.permutation(n)/n
        offset=-.6*np.log(rank[u])-.6*np.log(rank[v])
        eta=float(spec.get('eta',0.))
        if family == 'circle':
            offset+=eta*np.cos(theta[u]-theta[v])
        elif family == 'block':
            group=rng.permutation(n)%4
            offset+=float(spec.get('assortativity',3.))*(group[u]==group[v])
        elif family != 'beta':
            raise ValueError(family)
        target=n*avg/2
        c=brentq(lambda x: float(expit(2*x+offset).sum())-target,-50.,50.,xtol=1e-10)
        keep=rng.random(len(u)) < expit(2*c+offset)
        u,v=u[keep],v[keep]
    ip,ix=csr_from_edges(n,u,v)
    return save_graph(path,ip,ix,spec)
