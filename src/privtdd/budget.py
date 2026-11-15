"""Public-input budget minimization from the manuscript's allocation equation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, root
from scipy.special import softmax


def allocate_budget(n: int, epsilon: float):
    if isinstance(n,bool) or int(n)!=n or n<3 or not np.isfinite(epsilon) or epsilon<=0:
        raise ValueError('n >= 3 and a positive finite epsilon are required')
    degree_max = int(n)-1
    coefficient=np.sqrt(degree_max/(2*(degree_max-1)))

    def evaluate(z):
        shares=softmax(np.r_[z,0.])
        x=epsilon*shares[0]/2; y=epsilon*shares[1]; rel=epsilon*shares[2]/2
        sx,sy=np.sinh(x/2),np.sinh(y/2)
        cy=1/np.tanh(y/2)
        value=coefficient/(2*sy)+cy/rel+1.25/np.tanh(x/2)+1/x
        gx=cy/rel**2-.625/sx**2-1/x**2
        gy=-coefficient*np.cosh(y/2)/(4*sy**2)-.5/(sy**2*rel)+.5*cy/rel**2
        dx=x*np.array([1-shares[0],-shares[1]])
        dy=y*np.array([-shares[0],1-shares[1]])
        return float(value),gx*dx+gy*dy,(x,y,rel)

    solution=minimize(lambda z:evaluate(z)[:2],np.log([.30/.35,1.]),jac=True,method='BFGS',options={'gtol':1e-9})
    refinement=root(lambda z:evaluate(z)[1],solution.x)
    candidates=[solution.x,refinement.x]
    z=min(candidates,key=lambda z:np.linalg.norm(evaluate(z)[1],ord=np.inf))
    value,gradient,allocation=evaluate(z)
    if not np.isfinite(value) or np.linalg.norm(gradient,ord=np.inf)>1e-6:
        raise RuntimeError('Budget minimization failed the stationarity check')
    d,r,l=map(float,allocation)
    return dict(n=int(n),D=degree_max,epsilon=float(epsilon),eps_deg=d,eps_rr=r,eps_rel=l,
                L_for_allocation=0,objective=value,stationarity_residual=float(np.linalg.norm(gradient,ord=np.inf)))


def verify_allocations(table_path, tolerance=1e-7):
    table=json.loads(Path(table_path).read_text(encoding='utf-8'))
    checked=[]
    for row in table['rows']:
        actual=allocate_budget(row['n'],row['epsilon'])
        difference=max(abs(actual[k]-row[k]) for k in ('eps_deg','eps_rr','eps_rel'))
        if difference>tolerance:
            raise ValueError(f"Allocation differs from frozen paper table: {row['dataset']}, epsilon={row['epsilon']}")
        checked.append(dict(dataset=row['dataset'],epsilon=row['epsilon'],max_absolute_difference=difference,
                            stationarity_residual=actual['stationarity_residual']))
    return dict(rows=len(checked),tolerance=tolerance,max_absolute_difference=max(r['max_absolute_difference'] for r in checked),details=checked)
