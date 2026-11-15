"""Base distribution quadrature for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from privtdd.base_distribution.degree_posterior import DegreePosterior

import math

from privtdd.base_distribution.cells import normal_laplace_loginterval

import numpy as np



# [PrivTDD] S3 F1; numerical integration and stable computation; Select precision for far-tail normal-Laplace integration; this is a local numerical control.
def tail_cdf_error(L, a):
    # For x<=-L and a=sigma/b<L, replace F_NL by .5*exp(x*a+a*a/2).
    # The three omitted normal-tail terms have sum <=2*phi(L)/(L-a).
    if L<=a:raise ValueError('Tail threshold must exceed sigma/b')
    return 2*math.exp(-L*L/2)/math.sqrt(2*math.pi)/(L-a)



# [PrivTDD] S3 F1; numerical integration and stable computation; Stably compute an index-interval weight sum from logarithmic prefix sums.
def logrange(prefix, start, stop):
    answer=np.full(len(start),-np.inf);active=stop>start
    ids=np.flatnonzero(active)
    if len(ids):
        high=prefix[stop[ids]-1]
        low=np.full(len(ids),-np.inf);positive=start[ids]>0
        low[positive]=prefix[start[ids][positive]-1]
        with np.errstate(divide='ignore',invalid='ignore'):
            answer[ids]=high+np.log(-np.expm1(np.minimum(low-high,0.)))
    return answer



class IntervalMixture:
    # [PrivTDD] S3 F1; numerical integration and stable computation; Stably group equal noise scales and precompute sorted prefix/suffix moments.
    def __init__(self,y,b,a,sigma,L,*,batch_small_groups=False):
        self.sigma=float(sigma);self.L=float(L);self.groups=[];self.direct=None
        direct_ids=[]
        positive=np.flatnonzero(a>0)
        order=positive[np.argsort(b[positive],kind='stable')]
        scales,first,counts=np.unique(b[order],return_index=True,return_counts=True)
        for scale,start,count in zip(scales,first,counts):
            if batch_small_groups and count <= 8:
                direct_ids.append(order[start:start+count])
                continue
            ids=order[start:start+count];ids=ids[np.argsort(y[ids],kind='stable')]
            yy=y[ids];aa=a[ids];logaa=np.log(aa)
            if not len(ids):continue
            self.groups.append((float(scale),yy,aa,np.r_[0.,np.cumsum(aa)],
                np.logaddexp.accumulate(logaa+yy/scale),
                np.logaddexp.accumulate(logaa-yy/scale),
                np.logaddexp.accumulate((logaa-yy/scale)[::-1])[::-1]))
        if direct_ids:
            ids=np.concatenate(direct_ids)
            self.direct=(np.asarray(y)[ids],np.asarray(b)[ids],np.asarray(a)[ids])
        self.absolute_error_bound=sum(2*tail_cdf_error(L,sigma/bb)*float(aa.sum())
                                      for bb,yy,aa,unused,up,down,suffix in self.groups)

    # [PrivTDD] S3 F1; numerical integration and stable computation; Compute weighted normal-Laplace interval integrals by scale group: directly near the center and with prefix/suffix sums in distant tails.
    def integral(self,left,right):
        left=np.asarray(left,float);right=np.asarray(right,float)
        if left.shape!=right.shape or np.any(right<=left):raise ValueError('Ordered target intervals')
        answer=np.zeros(left.shape);band=self.L*self.sigma
        if self.direct is not None:
            y,b,a=self.direct
            # [Common-kernel implementation] Batch small scale groups and integrate the original kernel directly; 150000 only limits temporary matrix elements.
            step=max(1,150000//len(y))
            for offset in range(0,len(left),step):
                ell=left[offset:offset+step];r=right[offset:offset+step]
                values=np.exp(normal_laplace_loginterval(y[:,None]-r[None,:],y[:,None]-ell[None,:],self.sigma,b[:,None]))
                answer[offset:offset+step]=a@values
        for b,y,a,psum,up,down,suffix in self.groups:
            step=max(1,150000//max(1,len(y)))
            for offset in range(0,len(left),step):
                ell=left[offset:offset+step];r=right[offset:offset+step];M=len(ell)
                i0=np.searchsorted(y,ell-band,side='left')
                i1=np.searchsorted(y,ell+band,side='right')
                i2=np.searchsorted(y,r-band,side='left')
                i3=np.searchsorted(y,r+band,side='right')
                value=np.zeros(M);common=.5*(self.sigma/b)**2-math.log(2.)
                logwidth=np.log(-np.expm1(-(r-ell)/b))
                valid=i0>0
                value[valid]+=np.exp(up[i0[valid]-1]-ell[valid]/b+common+logwidth[valid])
                valid=i3<len(y)
                value[valid]+=np.exp(suffix[i3[valid]]+r[valid]/b+common+logwidth[valid])
                wide=i2>i1
                middle_start=i1.copy();middle_stop=np.maximum(i2,i1)
                if wide.any():
                    middle=psum[middle_stop]-psum[middle_start]
                    middle-=np.exp(logrange(up,middle_start,middle_stop)-r/b+common)
                    middle-=np.exp(logrange(suffix[::-1],len(y)-middle_stop,len(y)-middle_start)+ell/b+common)
                    value+=middle
                starts=np.r_[i0,np.where(wide,i2,i3)]
                stops=np.r_[np.where(wide,i1,i3),i3]
                counts=stops-starts
                query=np.repeat(np.tile(np.arange(M),2),counts)
                if len(query):
                    cumulative=np.r_[0,np.cumsum(counts)]
                    nodes=np.repeat(starts,counts)+np.arange(len(query))-np.repeat(cumulative[:-1],counts)
                    likelihood=np.exp(normal_laplace_loginterval(y[nodes]-r[query],y[nodes]-ell[query],self.sigma,b))
                    value+=np.bincount(query,weights=a[nodes]*likelihood,minlength=M)
                if np.min(value)<-1e-10*max(1.,float(a.sum())):
                    raise ArithmeticError('Negative integrated mixture')
                answer[offset:offset+step]+=np.maximum(value,0.)
        return answer



class FastDegreePosterior(DegreePosterior):
    # [PrivTDD] S3 F1; numerical integration and stable computation; Build a fast CDF with local error control on the blocked posterior base class.
    def __init__(self,*args,numerical_W1_limit=1e-6,batch_small_groups=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.fast_integral=None;self.numeric_W1_bound=0.;self.cdf_error_bound=0.;self.fallback_queries=0
        if self.W<=64:return
        active=self.b>0
        if not active.any():return
        sigma=math.sqrt(self.W*self.p*(1-self.p));gap=2*self.p-1
        ratio=float(np.max(self.theta[1:-1]/np.diff(self.edges)))/(gap*self.W*self.mass)
        if ratio==0:return
        amax=float(np.max(sigma/self.b[active]));total=float(self.a[active].sum());L=max(8.,amax+1.)
        # Two CDF endpoints give interval error 2e; output transport costs <=2W*CDFerror.
        while 2*self.W*ratio*2*tail_cdf_error(L,amax)*total>numerical_W1_limit:
            L+=.5
            if L>40:raise ArithmeticError('Unresolved analytic tail precision')
        self.fast_integral=IntervalMixture(self.y[active],self.b[active],self.a[active],sigma,L,batch_small_groups=batch_small_groups)
        self.cdf_error_bound=ratio*self.fast_integral.absolute_error_bound
        self.numeric_W1_bound=2*self.W*self.cdf_error_bound

    # [PrivTDD] S3 F1; numerical integration and stable computation; Return the CDF of this degree-conditional posterior at integer keys; return 0/1 outside its support.
    def cdf(self,keys):
        if self.fast_integral is None:return super().cdf(keys)
        keys=np.asarray(keys,dtype=np.int64);answer=np.zeros(keys.shape);answer[keys>=self.W]=1.
        valid=(keys>=0)&(keys<self.W)
        if not valid.any():return answer
        x=(keys[valid]+.5)/self.W
        cells=np.minimum(np.searchsorted(self.edges,x,side='right')-1,len(self.edges)-2)
        out=self.component_mass[0]+self.prefix[cells]
        left=self.edges[cells];width=np.diff(self.edges)[cells]
        selected=(x>left)&(self.theta[cells+1]>0)
        gap=2*self.p-1
        if selected.any():
            mass=self.fast_integral.integral(gap*self.W*left[selected],gap*self.W*x[selected])
            partial=mass/(gap*self.W*width[selected])
            partial+=float(self.a[self.b==0].sum())*(x[selected]-left[selected])/width[selected]
            out[selected]+=self.theta[cells[selected]+1]*partial
        out/=self.mass
        bad=(out<-1e-10)|(out>1+1e-10)|~np.isfinite(out)
        if bad.any():
            self.fallback_queries+=int(bad.sum());out[bad]=super().cdf(keys[valid][bad])
        answer[valid]=np.clip(out,0.,1.)
        return answer
