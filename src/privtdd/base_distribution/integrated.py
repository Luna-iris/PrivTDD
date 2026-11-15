"""Integrated base distribution likelihoods for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from privtdd.base_distribution.degree_posterior import DegreePosterior

from privtdd.base_distribution.integral import IntervalMixture

from privtdd.base_distribution.integral import FastDegreePosterior as Previous

from privtdd.base_distribution.count_channel import logpdf_q

import math

import numpy as np



class IntegratedDegreePosterior(Previous):
    # [PrivTDD] S3 F1; numerical integration and stable computation; Accumulate continuous-cell masses directly using common-kernel interval integration, avoiding another reports x 65 table expansion.
    def __init__(self,degree,report,scale,coefficient,theta,edges,p,*,numerical_W1_limit=1e-6,batch_small_groups=False):
        self.degree=int(degree);self.W=self.degree*(self.degree-1)//2
        if self.W<=64:
            super().__init__(degree,report,scale,coefficient,theta,edges,p,numerical_W1_limit=numerical_W1_limit,batch_small_groups=batch_small_groups)
            return
        self.y=np.asarray(report,float);self.b=np.asarray(scale,float);self.a=np.asarray(coefficient,float)
        self.theta=np.asarray(theta,float);self.edges=np.asarray(edges,float);self.p=p
        if not(self.y.shape==self.b.shape==self.a.shape) or np.any(self.b<0):raise ValueError('Posterior inputs')
        active=self.b>0
        if not active.any():
            super().__init__(degree,report,scale,coefficient,theta,edges,p,numerical_W1_limit=numerical_W1_limit,batch_small_groups=batch_small_groups)
            return
        if np.any(self.y[~active]!=0):raise ValueError('Nonzero deterministic report')
        self.finite_cdf=None;self.fallback_queries=0
        deterministic=float(self.a[~active].sum())
        y=self.y[active];b=self.b[active];a=self.a[active]
        atoms=np.full(2,deterministic)
        for start in range(0,len(y),16384):
            stop=start+16384
            density=np.exp(logpdf_q(y[start:stop,None],self.degree,np.array([0,self.W])[None,:],p,b[start:stop,None]))
            atoms+=a[start:stop]@density
        sigma=math.sqrt(self.W*p*(1-p));gap=2*p-1
        widths=gap*self.W*np.diff(self.edges)
        L=max(8.,float(np.max(sigma/b))+1.)
        while True:
            mixture=IntervalMixture(y,b,a,sigma,L,batch_small_groups=batch_small_groups)
            cells=mixture.integral(gap*self.W*self.edges[:-1],gap*self.W*self.edges[1:])/widths+deterministic
            self.component_mass=self.theta*np.r_[atoms[0],cells,atoms[1]]
            self.mass=float(self.component_mass.sum())
            errors=self.theta[1:-1]*mixture.absolute_error_bound/widths
            total_error=float(errors.sum());partial_error=float(errors.max(initial=0.))
            lower=self.mass-total_error
            bound=(2*total_error+partial_error)/lower if lower>0 else math.inf
            if 2*self.W*bound<=numerical_W1_limit:break
            L+=.5
            if L>40:raise ArithmeticError('Unresolved integrated posterior precision')
        self.fast_integral=mixture
        self.prefix=np.r_[0.,np.cumsum(self.component_mass[1:-1])]
        self.cdf_error_bound=bound;self.numeric_W1_bound=2*self.W*bound
        self.component_mass_error_bound=total_error

    # [PrivTDD] S3 F1; numerical integration and stable computation; Return the CDF of this degree-conditional posterior at integer keys; return 0/1 outside its support.
    def cdf(self,keys):
        # The inherited fast CDF uses precisely these component integrals.
        # Any rare direct fallback must recompute exact component normalizers,
        # rather than combining an exact partial integral with approximate ones.
        before=self.fallback_queries
        values=super().cdf(keys)
        if self.fallback_queries>before:
            exact=DegreePosterior(self.degree,self.y,self.b,self.a,self.theta,self.edges,self.p)
            return exact.cdf(keys)
        return values
