"""Degree-conditional posterior using the same projected forward working law.

Large retained capacities integrate continuous closure analytically; small
projected capacities integrate the original rounded count exactly. Output
keys are always original counts, never the truncated neighborhood counts.
"""
import math
import numpy as np
from privtdd.base_distribution.integral import IntervalMixture,tail_cdf_error
from privtdd.base_distribution.integral import FastDegreePosterior
from privtdd.base_distribution.cells import integer_cell_weights,normal_laplace_loginterval
from privtdd.base_distribution.count_channel import normal_laplace_logpdf,logpdf_q
from privtdd.base_distribution.projected_count_channel import projected_logpdf_q,projected_component_loglikelihood,projected_cell_count_probabilities,projected_partial_cell_mass


class ProjectedDegreePosterior:
    def __init__(self,degree,report,caps,scale,coefficient,theta,edges,p):
        self.degree=int(degree);self.W=self.degree*(self.degree-1)//2
        self.y=np.asarray(report,float);self.caps=np.asarray(caps,int);self.b=np.asarray(scale,float)
        self.a=np.asarray(coefficient,float);self.theta=np.asarray(theta,float);self.edges=np.asarray(edges,float);self.p=p
        if not(self.y.shape==self.caps.shape==self.b.shape==self.a.shape) or self.y.ndim!=1:raise ValueError('Posterior rows')
        if np.any(self.a<0) or np.any(self.b<0):raise ValueError('Nonnegative coefficients/scales')
        deterministic=self.caps<=1
        if np.any(self.y[deterministic]!=0):raise ValueError('Deterministic report')
        retained=np.minimum(self.degree,self.caps);w=retained*(retained-1)//2
        self.deterministic=deterministic;self.continuous=(~deterministic)&(w>64)
        self.small=(~deterministic)&(~self.continuous)
        self.slope=(2*p-1)*w[self.continuous];self.sigma=np.sqrt(w[self.continuous]*p*(1-p))
        K=np.ones((len(self.y),len(theta)))
        ids=np.flatnonzero(self.continuous)
        if len(ids):
            y=self.y[ids,None];b=self.b[ids,None];h=self.slope[:,None];sigma=self.sigma[:,None]
            K[ids,1:-1]=np.exp(normal_laplace_loginterval(y-h*self.edges[1:],y-h*self.edges[:-1],sigma,b))/(h*np.diff(self.edges))
            K[ids,0]=np.exp(normal_laplace_logpdf(self.y[ids],self.sigma,self.b[ids]))
            K[ids,-1]=np.exp(normal_laplace_logpdf(self.y[ids]-self.slope,self.sigma,self.b[ids]))
        self.small_likelihood=[]
        for cap in np.unique(self.caps[self.small]):
            ids=np.flatnonzero(self.small&(self.caps==cap))
            K[ids]=np.exp(projected_component_loglikelihood(self.y[ids],self.degree,int(cap),p,self.b[ids],self.edges))
            # Same linear mixture: sum weighted report likelihoods once per
            # cap, before querying many CDF endpoints. This is not Q binning.
            retained=min(self.degree,int(cap));capacity=retained*(retained-1)//2
            likelihood=np.exp(logpdf_q(self.y[ids,None],retained,np.arange(capacity+1),p,self.b[ids,None]))
            self.small_likelihood.append((capacity,self.a[ids]@likelihood))
        self.component_mass=(self.a@K)*self.theta;self.mass=float(self.component_mass.sum())
        if not self.mass>0:raise ArithmeticError('Empty projected posterior')
        self.prefix=np.r_[0.,np.cumsum(self.component_mass[1:-1])];self.finite_cdf=None
        if self.W<=64:
            T=np.arange(self.W+1);prior=integer_cell_weights(self.W,self.edges)@self.theta[1:-1]
            prior[0]+=self.theta[0];prior[-1]+=self.theta[-1]
            likelihood=np.ones((len(self.y),len(T)))
            for cap in np.unique(self.caps[~deterministic]):
                ids=np.flatnonzero((self.caps==cap)&(~deterministic))
                likelihood[ids]=np.exp(projected_logpdf_q(self.y[ids,None],self.degree,int(cap),T[None,:],p,self.b[ids,None]))
            mass=(self.a@likelihood)*prior
            if not np.isclose(mass.sum(),self.mass,rtol=2e-11,atol=0.):raise ArithmeticError('Projected posterior mass mismatch')
            self.finite_cdf=np.cumsum(mass)/self.mass

    def cdf(self,keys):
        keys=np.asarray(keys,np.int64);answer=np.zeros(keys.shape);answer[keys>=self.W]=1.
        valid=(keys>=0)&(keys<self.W)
        if not valid.any():return answer
        k=keys[valid]
        if self.finite_cdf is not None:
            answer[valid]=self.finite_cdf[k];return answer
        x=(k+.5)/self.W;cells=np.minimum(np.searchsorted(self.edges,x,side='right')-1,len(self.edges)-2)
        out=self.component_mass[0]+self.prefix[cells]
        ia=np.flatnonzero(self.continuous);det=float(self.a[self.deterministic].sum())
        for cell in np.unique(cells):
            selected=np.flatnonzero((cells==cell)&(x>self.edges[cell]))
            if self.theta[cell+1]==0:continue
            left,right=self.edges[cell:cell+2];step=max(1,150000//max(1,len(ia)))
            for start in range(0,len(selected),step):
                ids=selected[start:start+step];xx=x[ids]
                value=det*(xx-left)/(right-left)
                if len(ia):
                    logmass=normal_laplace_loginterval(self.y[ia,None]-self.slope[:,None]*xx,
                        self.y[ia,None]-self.slope[:,None]*left,self.sigma[:,None],self.b[ia,None])
                    value+=(self.a[ia]/self.slope)@np.exp(logmass)/(right-left)
                for capacity,likelihood in self.small_likelihood:
                    probability=projected_partial_cell_mass(self.W,capacity,left,k[ids])
                    value+=(probability@likelihood)/(right-left)
                out[ids]+=self.theta[cell+1]*value
        answer[valid]=out/self.mass
        if np.any(answer< -1e-11) or np.any(answer>1+1e-11):raise ArithmeticError('Projected CDF range')
        return np.clip(answer,0.,1.)


class IntegratedProjectedPosterior(ProjectedDegreePosterior):
    def __init__(self,*args,numerical_W1_limit=1e-6,**kwargs):
        super().__init__(*args,**kwargs)
        self.integrals=[];self.cdf_error_bound=0.;self.numeric_W1_bound=0.
        if self.finite_cdf is not None:return
        caps=np.unique(self.caps[self.continuous]);ratio=float(np.max(self.theta[1:-1]/np.diff(self.edges)))
        for cap in caps:
            ids=np.flatnonzero(self.continuous&(self.caps==cap));retained=min(int(cap),self.degree);w=retained*(retained-1)//2
            h=(2*self.p-1)*w;sigma=math.sqrt(w*self.p*(1-self.p))
            amax=float(np.max(sigma/self.b[ids]));L=max(8.,amax+1.)
            # Same tail error formula and global W1 target as original Joint.
            multiplier=ratio/(h*self.mass);total=float(self.a[ids].sum())
            while 2*self.W*multiplier*2*tail_cdf_error(L,amax)*total>numerical_W1_limit/max(1,len(caps)):
                L+=.5
                if L>40:raise ArithmeticError('Projected interval precision unresolved')
            mixture=IntervalMixture(self.y[ids],self.b[ids],self.a[ids],sigma,L,batch_small_groups=True)
            self.integrals.append((h,mixture))
            self.cdf_error_bound+=multiplier*mixture.absolute_error_bound
        self.numeric_W1_bound=2*self.W*self.cdf_error_bound

    def cdf(self,keys):
        if self.finite_cdf is not None:return super().cdf(keys)
        keys=np.asarray(keys,np.int64);answer=np.zeros(keys.shape);answer[keys>=self.W]=1.
        valid=(keys>=0)&(keys<self.W)
        if not valid.any():return answer
        k=keys[valid];x=(k+.5)/self.W
        cells=np.minimum(np.searchsorted(self.edges,x,side='right')-1,len(self.edges)-2)
        out=self.component_mass[0]+self.prefix[cells]
        left=self.edges[cells];width=np.diff(self.edges)[cells]
        selected=(x>left)&(self.theta[cells+1]>0)
        ids=np.flatnonzero(selected)
        if len(ids):
            partial=float(self.a[self.deterministic].sum())*(x[ids]-left[ids])
            for h,mixture in self.integrals:
                partial+=mixture.integral(h*left[ids],h*x[ids])/h
            for cell in np.unique(cells[ids]):
                selected_ids=np.flatnonzero(cells[ids]==cell)
                for capacity,likelihood in self.small_likelihood:
                    partial[selected_ids]+=projected_partial_cell_mass(self.W,capacity,self.edges[cell],k[ids[selected_ids]])@likelihood
            out[ids]+=self.theta[cells[ids]+1]*partial/width[ids]
        answer[valid]=out/self.mass
        if np.any(answer< -1e-11) or np.any(answer>1+1e-11):raise ArithmeticError('Projected CDF range')
        return np.clip(answer,0.,1.)




class FastProjectedDegreePosterior:
    """Reuse the certified fast integral on unprojected observations.

    Projected observations use direct analytic integration. Mixing the two
    posterior measures is by their native likelihood mass, not a fitted weight.
    """
    def __init__(self,degree,report,caps,scale,coefficient,theta,edges,p,*,numerical_W1_limit=1e-6,batch_small_groups=False,posterior_backend='integrated'):
        self.degree=int(degree);self.W=self.degree*(self.degree-1)//2
        y,c,b,a=[np.asarray(v) for v in [report,caps,scale,coefficient]]
        unprojected=(c>=degree)|(c<=1);self.parts=[]
        if unprojected.any():
            s=unprojected
            from privtdd.base_distribution.integrated import IntegratedDegreePosterior
            constructor=IntegratedDegreePosterior if posterior_backend=='integrated' else FastDegreePosterior
            self.parts.append(constructor(int(degree),y[s],b[s],a[s],theta,edges,p,
                numerical_W1_limit=numerical_W1_limit,batch_small_groups=batch_small_groups))
        if (~unprojected).any():
            s=~unprojected
            if posterior_backend=='integrated':
                self.parts.append(IntegratedProjectedPosterior(int(degree),y[s],c[s],b[s],a[s],theta,edges,p,
                    numerical_W1_limit=numerical_W1_limit))
            else:
                self.parts.append(ProjectedDegreePosterior(int(degree),y[s],c[s],b[s],a[s],theta,edges,p))
        if not self.parts:raise ValueError('Nonempty posterior rows')
        self.component_mass=sum((part.component_mass for part in self.parts),np.zeros(len(theta)))
        self.mass=float(self.component_mass.sum())
        self.cdf_error_bound=sum(part.mass*getattr(part,'cdf_error_bound',0.) for part in self.parts)/self.mass
        self.numeric_W1_bound=sum(part.mass*getattr(part,'numeric_W1_bound',0.) for part in self.parts)/self.mass

    @property
    def fallback_queries(self):return sum(getattr(part,'fallback_queries',0) for part in self.parts)

    def cdf(self,keys):
        return sum((part.mass*part.cdf(keys) for part in self.parts),np.zeros(np.asarray(keys).shape))/self.mass
