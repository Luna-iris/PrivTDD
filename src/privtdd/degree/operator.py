"""Capped degree likelihood operator for PrivTDD.

Method phases and paper settings: docs/method.md.
Source and migration checksums: provenance/module_mapping.json.
Statistical formulas and numerical constants are preserved.
"""

from __future__ import annotations

from scipy.signal import lfilter

from privtdd.degree.channel import log_cap_probability

import math

import numpy as np
from privtdd.config import CAP_OFFSET



class CapDegreeOperator:
    # [PrivTDD] S2 degree prior; core method or probability/structure primitive (not necessarily a separate contribution); Initialize boundary and exponential-scan state for the linear-time cap-channel operator.
    def __init__(self,public_cap,eps_degree,offset=CAP_OFFSET):
        if not isinstance(offset,(int,np.integer)) or offset<0:raise ValueError('Nonnegative integer offset required')
        self.D=int(public_cap);self.ed=float(eps_degree);self.offset=int(offset)
        if self.D<1 or not np.isfinite(self.ed) or self.ed<=0:raise ValueError('Invalid channel parameters')
        self.r=math.exp(-self.ed);self.h=.5*(-math.expm1(-self.ed))
        self.support=np.arange(self.D+1);self.t=np.arange(1,self.D)-self.offset
        self.endpoint0=np.exp(log_cap_probability(0,self.support,self.D,self.ed,self.offset))
        self.endpointD=np.exp(log_cap_probability(self.D,self.support,self.D,self.ed,self.offset))
        self.lo=min(0,1-self.offset);self.hi=self.D
        self.positions=np.arange(self.D+1)-self.lo
        self.interior_positions=self.t-self.lo

    # [PrivTDD] S2 degree prior; core method or probability/structure primitive (not necessarily a separate contribution); Forward/backward geometric filtering in linear time avoids explicitly constructing the D x D channel matrix.
    def scans(self,x):
        left=lfilter([1.],[1.,-self.r],x)
        right=lfilter([1.],[1.,-self.r],x[::-1])[::-1]
        return left,right

    # [PrivTDD] S2 degree prior; core method or probability/structure primitive (not necessarily a separate contribution); Compute the cap channel times the degree prior, K*pi.
    def forward(self,prior):
        p=np.asarray(prior,float)
        if p.shape!=(self.D+1,):raise ValueError('Prior shape')
        left,right=self.scans(p);t=self.t
        y=np.empty(self.D+1);y[0]=self.endpoint0@p;y[-1]=self.endpointD@p
        values=np.empty(len(t));below=t<0;above=t>=self.D;middle=~(below|above)
        values[below]=self.h*np.exp(self.ed*(t[below]+1))*right[0]
        values[above]=self.h*np.exp(-self.ed*(t[above]-self.D))*left[-1]
        values[middle]=self.h*(left[t[middle]]+right[t[middle]+1])
        y[1:-1]=values
        return y

    # [PrivTDD] S2 degree prior; core method or probability/structure primitive (not necessarily a separate contribution); Compute the transposed cap channel times observation weights, K^T*w.
    def adjoint(self,weights):
        w=np.asarray(weights,float)
        if w.shape!=(self.D+1,):raise ValueError('Observation shape')
        extended=np.zeros(self.hi-self.lo+1)
        extended[self.interior_positions]=w[1:-1]
        left,right=self.scans(extended);positions=self.positions
        prev=np.zeros(len(positions));valid=positions>0;prev[valid]=left[positions[valid]-1]
        return self.h*(right[positions]+prev)+w[0]*self.endpoint0+w[-1]*self.endpointD
