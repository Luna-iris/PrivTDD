"""Correction observation operator for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from privtdd.correction.feasible_domain import coefficient_domain

from privtdd.correction.feasible_domain import degree_gate_basis

import numpy as np

from privtdd.base_distribution.weights import prepare_groups

from privtdd.correction.source_mapping import source_map

from privtdd.correction.sparse_channel import sparse_component_channel, use_batched_channel, batched_integer_channel
from privtdd.correction.projected_sparse_channel import projected_sparse_channel

import privtdd.correction.basis as stable



# [PrivTDD] S5 forward scoring; core method or probability/structure primitive (not necessarily a separate contribution); Invert cap-group indices into a list of (cap, node indices, weights) for each d.
def groups_by_degree(groups):
    result={}
    for group in groups:
        for d,weight in zip(group['degree'],group['posterior']):
            if weight>0:result.setdefault(int(d),[]).append((group['cap'],group['nodes'],float(weight)))
    return result



# [PrivTDD] S5 forward scoring; core method or probability/structure primitive (not necessarily a separate contribution); Pass the three conditional sources for the same degree and region through the shared reporting kernel to build the feasible joint family and forward direction tables.
def build_family(sources,caps,report,prior,p,eps_degree,eps_release,boundaries,*,component_directory=None,workers=8,batch_small_groups=False):
    """No graph name, truth metric, or FULL snapshot is an inference input.
    
    batch_small_groups: Implementation switch; batch the same kernel over sparse report groups while retaining each report's actual scale.
    """
    caps=np.asarray(caps,np.int64);report=np.asarray(report,float);prior=np.asarray(prior,float)
    support=np.flatnonzero(prior>0);pi=prior[support];n=len(report)
    for source in sources:
        if not np.array_equal(source.support,support):raise ValueError('Common fixed-degree support required')
    groups=prepare_groups(caps,prior,eps_degree);by_degree=groups_by_degree(groups)
    basis,gate_audit=degree_gate_basis(support,pi)
    bases=np.zeros((len(support),12));directions=np.zeros((len(support),12,16));masses=np.zeros((len(support),3,4))
    predictive=np.zeros(n);delta=np.zeros((16,n));source_prediction=np.zeros((n,2))
    kernel_error=np.zeros(n);kernel_direction_error=np.zeros((16,n));omission=np.zeros(n)
    endpoint_error=0.;normalization_error=0.;retained_pairs=0;component_files=[]
    if component_directory is not None:
        component_directory=Path(component_directory);component_directory.mkdir(parents=True,exist_ok=True)
    for group in groups:
        cap=group['cap'];ids=group['nodes'];missing=group['omitted_numerical']+group['omitted_projected']
        if cap<=1:
            if np.any(report[ids]!=0):raise ArithmeticError('Deterministic report must be zero')
            omission[ids]=missing
        else:omission[ids]=missing/(2*((cap-1)/eps_release))
    pool=ThreadPoolExecutor(max_workers=workers)
    for j,d in enumerate(support):
        d=int(d);keys,shapes,base,D,mass=stable.aligned_row(sources,j,boundaries,basis[j])
        bases[j]=base;directions[j]=D;masses[j]=mass
        shape_mass=shapes.sum(axis=1);direction_l1=np.abs(D).T@shape_mass
        original=np.zeros(len(keys));fk,fp=sources[0].row(j);original[np.searchsorted(keys,fk)]=fp
        normalization_error=max(normalization_error,float(np.max(abs(base@shapes-original))))
        mapping=source_map()
        for column,source in enumerate((sources[2],sources[1])):
            expected=np.zeros(len(keys));k,v=source.row(j);expected[np.searchsorted(keys,k)]=v
            endpoint_error=max(endpoint_error,float(np.max(abs((base+D@mapping[:,column])@shapes-expected))))
        saved_nodes=[];saved_weight=[];saved_values=[];saved_error=[]
        entries=by_degree.get(d,[])
        # [S5 projection model] Only unprojected states may use the old batched
        # kernel. Keep every projected state's native degree posterior weight.
        ordinary = [entry for entry in entries if entry[0] >= d or entry[0] <= 1]
        batched = {}
        if batch_small_groups and use_batched_channel(ordinary, d, len(keys)):
            sizes = [len(entry[1]) for entry in ordinary]
            ids = np.concatenate([entry[1] for entry in ordinary])
            values = batched_integer_channel(report[ids], np.maximum(caps[ids]-1, 0)/eps_release,
                                             d, keys, p, shapes.T)
            offsets = np.r_[0, np.cumsum(sizes)]
            batched = {entry[0]: values[a:b] for entry, a, b in zip(ordinary, offsets[:-1], offsets[1:])}
        def cap_task(entry):
            cap,ids,weight=entry
            if cap<=1:
                values=np.tile(shape_mass,(len(ids),1));err=np.zeros(12)
                base_values=np.ones(len(ids));delta_values=np.zeros((len(ids),16));raw=np.ones((len(ids),2))
            else:
                b=(cap-1)/eps_release
                if cap in batched:
                    values,err=batched[cap],np.zeros(12)
                else:
                    values,err=projected_sparse_channel(keys,shapes.T,d,cap,p,b,report[ids])
                base_values=values@base;delta_values=values@D
                raw=np.column_stack((values[:,2::3]@mass[2],values[:,1::3]@mass[1])) # Beta, Geo
            predictive[ids]+=weight*base_values;delta[:,ids]+=weight*delta_values.T
            source_prediction[ids]+=weight*raw
            bound=float(np.max(err));kernel_error[ids]+=weight*bound
            kernel_direction_error[:,ids]+=weight*direction_l1[:,None]*bound
            if component_directory is not None:
                saved_nodes.append(ids);saved_weight.append(np.full(len(ids),weight));saved_values.append(values)
                saved_error.append(np.full(len(ids),bound))
        tasks = map(cap_task, entries) if batched else pool.map(cap_task,entries)
        for unused in tasks:pass
        retained_pairs+=sum(len(entry[1]) for entry in entries)
        if component_directory is not None and saved_nodes:
            path=component_directory/f'd{d}.npz'
            np.savez_compressed(path,degree=d,node_ids=np.concatenate(saved_nodes),degree_weight=np.concatenate(saved_weight),
                component_density=np.concatenate(saved_values),kernel_error=np.concatenate(saved_error))
            component_files.append(path.name)
    pool.shutdown()
    if endpoint_error>3e-12 or normalization_error>3e-12:raise ArithmeticError('Lost F1 or complete-source identity')
    if predictive.min()<=0 or not np.isfinite(predictive).all():raise ArithmeticError('F1 forward prediction unresolved')
    A,b=coefficient_domain(bases,directions)
    maximum_direction_l1=np.max(np.abs(directions).sum(axis=1),axis=0)
    return dict(support=support,pi=pi,n=n,counts=np.ones(n),sources=sources,boundaries=tuple(boundaries),
        gate_basis=basis,gate_audit=gate_audit,base_coefficients=bases,coefficient_directions=directions,
        partition_mass=masses,A=A,b=b,predictive_table=predictive,delta_tables=delta,
        source_prediction=source_prediction,numerical_prediction_error=kernel_error+omission,
        numerical_direction_error=kernel_direction_error+maximum_direction_l1[:,None]*omission[None,:],
        kernel_error=kernel_error,omission_density_bound=omission,
        channel_audit=dict(endpoint_error=endpoint_error,F1_base_error=normalization_error,
            node_degree_pairs=retained_pairs,component_files=component_files,
            observed_scales_used=True,private_report_binning=False,source_order=['F1','Beta','Geo'],
            RR_exact_when_W_at_most_64=True,large_degree_RR_is_working_normal=True,
            projected_domain_complete=True,projection_model='conditional_uniform_edges_small_mean_thinning_large',
            truth_used=False,additional_epsilon=0.))
