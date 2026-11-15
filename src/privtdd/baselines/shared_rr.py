"""Numba-accelerated exact joint-law simulation of shared GNAM / ARR wedge sums.
Uses the exact same pair-indexed RR randomness as external_gdd.py.
No dense n*n adjacency; per-pair PRF hash guarantees repeated reports match.
"""
import numpy as np
try:
    from numba import njit, prange
    HAS_NUMBA=True
except ImportError:
    HAS_NUMBA=False


def csr_sorted(sets):
    n=len(sets)
    indptr=np.empty(n+1,dtype=np.int64)
    indptr[0]=0
    for i,s in enumerate(sets):indptr[i+1]=indptr[i]+len(s)
    vals=np.empty(indptr[-1],dtype=np.int64)
    for i,s in enumerate(sets):
        vals[indptr[i]:indptr[i+1]]=sorted(s)
    return indptr,vals

if HAS_NUMBA:
    @njit(inline='always')
    def binary_contains(indptr,neighbors,u,v):
        lo=indptr[u];hi=indptr[u+1]
        while lo<hi:
            mid=(lo+hi)//2
            w=neighbors[mid]
            if w<v:lo=mid+1
            else:hi=mid
        return lo<indptr[u+1] and neighbors[lo]==v

    @njit(inline='always')
    def randpair(u,v,n,secret):
        x=(np.uint64(u)*np.uint64(n)+np.uint64(v))^np.uint64(secret)
        x=x+np.uint64(0x9E3779B97F4A7C15)
        x=(x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        x=x^(x>>np.uint64(31))
        return float(x>>np.uint64(11))*(1.0/(1<<53))

    @njit(parallel=True)
    def _compute_raw(proj_ptr,proj_idx,center_ptr,center_idx,n,secret,p,q,normalize):
        out=np.zeros(n,dtype=np.float64)
        for center in prange(n):
            st=center_ptr[center];ed=center_ptr[center+1]
            accum=0.0
            for a in range(st,ed):
                u=center_idx[a]
                for b in range(a+1,ed):
                    v=center_idx[b] # center_idx is sorted, so u<v
                    bitp=p if binary_contains(proj_ptr,proj_idx,u,v) else q
                    resp=1.0 if randpair(u,v,n,secret)<bitp else 0.0
                    accum+=resp-q
            if normalize:
                accum/=(p-q)
            out[center]=accum
        return out


def compute(rr, local_adj, normalize=True, max_wedge_visits=None):
    if not HAS_NUMBA:
        raise RuntimeError('pip install numba to use the compiled sparse RR backend')
    visits=int(sum(len(s)*(len(s)-1)//2 for s in local_adj))
    if max_wedge_visits is not None and visits > max_wedge_visits:
        raise RuntimeError(f'Need {visits} wedge accesses > max_wedge_visits={max_wedge_visits}')
    proj_ptr,proj_idx=csr_sorted(rr.proj)
    center_ptr,center_idx=csr_sorted(local_adj)
    ret=_compute_raw(proj_ptr,proj_idx,center_ptr,center_idx,len(local_adj),
                     np.uint64(rr.secret),float(rr.p),float(rr.q),bool(normalize))
    return ret,visits
