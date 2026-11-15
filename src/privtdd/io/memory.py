"""Memory and array storage utilities for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from privtdd.degree.groups import groups_by_degree

import numpy as np

from privtdd.base_distribution.cells import component_loglikelihood as original_component

import os



# [PrivTDD] memory across stages; memory and parallel execution; Compute each closure-rate component likelihood for report Q; the memory module uses blocks of 2048 reports.
def component_loglikelihood(report,degree,p,scale,edges):
    report=np.asarray(report);scale=np.asarray(scale)
    if len(report)<=2048:return original_component(report,degree,p,scale,edges)
    output=np.empty((len(report),len(edges)+1))
    for start in range(0,len(report),2048):
        stop=min(start+2048,len(report))
        output[start:stop]=original_component(report[start:stop],degree,p,scale if scale.ndim==0 else scale[start:stop],edges)
    return output



class LazyRows:
    # [PrivTDD] memory across stages; memory and parallel execution; Store only the cap-group-to-degree index and node normalizers.
    def __init__(self,groups,normalizer,n):
        self.entries=groups_by_degree(groups);self.R=normalizer;self.n=n
    # [PrivTDD] memory across stages; memory and parallel execution; Construct node indices and w_id/(n R_i) on demand for the requested degree, avoiding all node-degree pairs in memory.
    def get(self,degree):
        entries=self.entries.get(degree,[])
        if not entries:return np.array([],np.int64),np.array([],float)
        size=sum(len(ids) for cap,ids,w in entries)
        nodes=np.empty(size,np.int64);weights=np.empty(size);start=0
        for cap,ids,w in entries:
            end=start+len(ids);nodes[start:end]=ids;weights[start:end]=w/(self.n*self.R[ids]);start=end
        return nodes,weights



# [PrivTDD] memory across stages; memory and parallel execution; Write a temporary NPZ and atomically replace the target to avoid partially written intermediate results.
def atomic_npz(path,**values):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**values)
    os.replace(temporary,path)
