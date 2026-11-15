"""Projection completion and matched-channel regressions, no dataset fitting."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from privtdd.base_distribution.channel import prepare_groups, build_matrix
from privtdd.correction.projected_sparse_channel import projected_sparse_channel, projected_sparse_adjoint
from privtdd.base_distribution.projected_count_channel import retained_count_probabilities
from privtdd.base_distribution.projected_posterior import FastProjectedDegreePosterior
from privtdd.base_distribution.cells import closure_edges


class ProjectionJointRegression(unittest.TestCase):
    def test_mixed_family_and_posterior_use_matched_projected_kernel(self):
        from concurrent.futures import ThreadPoolExecutor
        from privtdd.distributions.conditional import ConditionalDistribution
        from privtdd.correction.forward import build_family
        from privtdd.posterior.update import posterior_evidence, update_posterior_joint
        from privtdd.correction.sparse_channel import sparse_component_channel
        d=30;W=d*(d-1)//2;keys=np.array([0,3,70,200,W])
        prior=np.zeros(d+1);prior[d]=1.;support=np.array([d]);pi=np.ones(1)
        probabilities=[np.array([.2,.1,.3,.2,.2]),np.array([.1,.2,.1,.3,.3]),np.array([.3,.2,.2,.2,.1])]
        sources=tuple(ConditionalDistribution(str(i),support,(keys,),(mass,),pi) for i,mass in enumerate(probabilities))
        caps=np.array([0,1,2,8,15,30]);q=np.array([0.,0.,-4.,20.,60.,120.]);p=.7;el=.2
        direct=build_family(sources,caps,q,prior,p,.3,el,(70,200),workers=2,batch_small_groups=False)
        batch=build_family(sources,caps,q,prior,p,.3,el,(70,200),workers=2,batch_small_groups=True)
        for field in ['predictive_table','delta_tables','A','b']:
            np.testing.assert_allclose(batch[field],direct[field],rtol=2e-11,atol=1e-13)
        expected=np.array([projected_sparse_channel(keys,probabilities[0],d,int(c),p,max(c-1,0)/el,np.array([y]))[0][0,0] for c,y in zip(caps,q)])
        np.testing.assert_allclose(batch['predictive_table'],expected,rtol=2e-11,atol=1e-14)
        # No projection reduces to the previous ordinary kernel exactly.
        ordinary,_=sparse_component_channel(keys,probabilities[0],d,p,(d-1)/el,q[-1:])
        np.testing.assert_allclose(expected[-1],ordinary[0,0],rtol=1e-13)
        entries=[(int(c),np.array([i]),1.) for i,c in enumerate(caps)]
        inverse=np.arange(1,7,dtype=float)[:,None]/21
        with ThreadPoolExecutor(max_workers=2) as pool:
            a=posterior_evidence(keys,d,entries,q,p,el,inverse,pool,batch_small_groups=True)
            b=posterior_evidence(keys,d,entries,q,p,el,inverse,pool,batch_small_groups=False)
        np.testing.assert_allclose(a,b,rtol=2e-11,atol=1e-14)
        self.assertAlmostEqual(float(probabilities[0]@a[:,0]),float(expected@inverse[:,0]),places=12)
        hist=update_posterior_joint(batch,np.zeros(16),expected[:,None],caps,q,prior,p,.3,el,workers=2,batch_small_groups=True)
        self.assertAlmostEqual(sum(hist.values()),len(caps),places=10)
        self.assertGreater(hist[W],0.)

    def test_large_projected_integral_matches_direct_cdf(self):
        from privtdd.base_distribution.projected_posterior import ProjectedDegreePosterior, IntegratedProjectedPosterior
        for d in [50,500,5000]:
            W=d*(d-1)//2;caps=np.array([12,20]);y=np.array([-4.,33.]);b=(caps-1)/.2
            theta=np.arange(1.,66.);theta/=theta.sum()
            args=(d,y,caps,b,np.array([.4,.6]),theta,closure_edges(),.7)
            direct=ProjectedDegreePosterior(*args)
            integrated=IntegratedProjectedPosterior(*args,numerical_W1_limit=1e-6)
            keys=np.unique(np.r_[0,1,64,np.linspace(0,W,31).astype(int),W])
            np.testing.assert_allclose(integrated.cdf(keys),direct.cdf(keys),atol=2e-10,rtol=2e-10)
            self.assertLessEqual(integrated.numeric_W1_bound,1e-6)

    def test_report_space_exposure_weights(self):
        from privtdd.references.calibration import report_degree_weights
        prior=np.zeros(31);prior[30]=1.
        caps=np.array([0,1,3,15,30])
        support,weights,_=report_degree_weights(caps,prior,.3)
        np.testing.assert_array_equal(support,[30])
        self.assertAlmostEqual(weights[0],np.mean(caps*(caps-1)/2)/(30*29/2))

    def test_batched_projected_cdf_matches_dense_original_count_law(self):
        from privtdd.base_distribution.projected_posterior import ProjectedDegreePosterior
        from privtdd.base_distribution.projected_count_channel import projected_logpdf_q
        from privtdd.base_distribution.cells import integer_cell_weights
        degree=30;W=degree*(degree-1)//2;keys=np.arange(W+1)
        caps=np.array([0,1,2,8,11]);q=np.array([0.,0.,5.,-30.,10.])
        b=np.maximum(caps-1,0)/.2;a=np.array([1.,2.,3.,4.,5.])/15
        theta=np.arange(1.,66.);theta/=theta.sum();edges=closure_edges()
        row=ProjectedDegreePosterior(degree,q,caps,b,a,theta,edges,.7)
        prior=integer_cell_weights(W,edges)@theta[1:-1]
        prior[0]+=theta[0];prior[-1]+=theta[-1]
        likelihood=np.array([np.exp(projected_logpdf_q(q[j],degree,int(cap),keys,.7,b[j])) for j,cap in enumerate(caps)])
        measure=prior*(a@likelihood)
        np.testing.assert_allclose(row.cdf(keys),np.cumsum(measure)/measure.sum(),rtol=2e-11,atol=2e-13)

    def test_batched_hypergeom_partial_intervals(self):
        from privtdd.base_distribution.projected_count_channel import projected_partial_cell_mass,projected_cell_count_probabilities
        for degree in [12,30,200,10000]:
            W=degree*(degree-1)//2
            for cap in [2,5,11]:
                for left in [0.,.001,.1,.73]:
                    lo=int(np.floor(W*left+.5));keys=np.unique(np.minimum(W-1,lo+np.array([0,1,63,64,65,2000])))
                    result=projected_partial_cell_mass(W,cap*(cap-1)//2,left,keys)
                    for j,key in enumerate(keys):
                        end=(key+.5)/W;partition=np.unique([0.,left,end,1.])
                        index=np.flatnonzero(partition==left)[0]+1
                        reference,_=projected_cell_count_probabilities(degree,cap,partition)
                        np.testing.assert_allclose(result[j],reference[:,index]*(end-left),rtol=2e-8,atol=2e-13)

    def test_vectorized_cells_match_direct_integer_sum(self):
        from privtdd.base_distribution.projected_count_channel import projected_cell_count_probabilities
        from privtdd.base_distribution.cells import integer_cell_weights
        for d in [12,30,50]:
            W=d*(d-1)//2
            for cap in [2,5,11]:
                expected=retained_count_probabilities(W,np.arange(W+1),cap*(cap-1)//2).T@integer_cell_weights(W,closure_edges())
                actual,_=projected_cell_count_probabilities(d,cap)
                np.testing.assert_allclose(actual[:,1:-1],expected,rtol=1e-11,atol=2e-13)

    def test_degree_above_cap_is_retained(self):
        prior=np.zeros(31);prior[30]=1.
        groups=prepare_groups(np.array([0,1,2,8,15]),prior,.3)
        for g in groups:
            np.testing.assert_array_equal(g['degree'],[30])
            self.assertEqual(g['omitted_projected'],0.)
            self.assertAlmostEqual(g['posterior'].sum(),1.)

    def test_hypergeometric_mean_and_forward_adjoint(self):
        for degree,cap in [(6,3),(30,15),(6,8),(30,1)]:
            W=degree*(degree-1)//2;keys=np.array([0,W//3,W])
            y=np.zeros(4) if cap<=1 else np.array([-10.,0.,5.,30.])
            mass=np.array([[.1,.2],[.3,.5],[.6,.3]])
            weights=np.array([[.1,.2],[.3,.5],[.7,.8],[.2,.1]])
            value,_=projected_sparse_channel(keys,mass,degree,cap,.7,max(cap-1,0)/.2,y)
            adjoint,_=projected_sparse_adjoint(y,weights,degree,cap,keys,.7,max(cap-1,0)/.2)
            np.testing.assert_allclose(value.T@weights,mass.T@adjoint,rtol=2e-11,atol=1e-14)
        pmf=retained_count_probabilities(15,np.array([0,5,15]),3)
        np.testing.assert_allclose(pmf@np.arange(4),[0,1,3],atol=1e-12)

    def test_projected_f1_mass_matches_forward_and_batch(self):
        degree=30;prior=np.zeros(degree+1);prior[degree]=1.
        caps=np.array([0,1,2,8,15,31]);q=np.array([0.,0.,-.5,4.,30.,80.])
        args=(q,caps,prior,.7,.3,.2)
        # Public cap is degree: keep last observation within its valid domain.
        caps[-1]=30
        A,_,groups,edges,_=build_matrix(*args,workers=2,batch_degrees=True)
        B,*_=build_matrix(*args,workers=2,batch_degrees=False)
        np.testing.assert_allclose(A,B,rtol=1e-12,atol=1e-14)
        theta=np.full(65,1/65);coefficient=np.arange(1,7)/21
        row=FastProjectedDegreePosterior(degree,q,caps,np.maximum(caps-1,0)/.2,coefficient,theta,edges,.7,batch_small_groups=True)
        self.assertAlmostEqual(row.mass,float(coefficient@(A@theta)),places=12)
        cdf=row.cdf(np.array([-1,0,100,degree*(degree-1)//2]))
        self.assertEqual(cdf[0],0.);self.assertEqual(cdf[-1],1.)
        self.assertTrue(np.all(np.diff(cdf)>=-1e-12))

    def test_complete_pipeline_all_informative_states_projected(self):
        from privtdd.release.pipeline import estimate_tdd_from_reports
        from privtdd.config import PrivTDDConfig
        caps=np.tile(np.array([0,1,2,3,4]),8)
        noisy=np.tile(np.array([0.,0.,-1.,2.,3.]),8)
        prior=np.zeros(9);prior[6]=1.
        config=PrivTDDConfig(workers=2,blas_threads=1,geometry_draws=32,geometry_replicates=1,
                           report_bin_width=1/16,triangle_bin_width=1/16,
                           batch_f1=True,batch_small_groups=True,structural_lazy_endpoints=True)
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
            hist,info=estimate_tdd_from_reports(caps,noisy,8,2.,config=config,
                    degree_prior=prior,workdir=Path(directory)/'work',return_debug=True)
            with np.load(Path(directory)/'work/degree.npz') as z:
                np.testing.assert_array_equal(z['prior'], prior)
        self.assertAlmostEqual(sum(hist.values()),40.,places=8)
        # L0 preserves the degree-6 prior exactly. All reports have cap<=4,
        # yet the complete projected pipeline must retain this latent degree.
        # Its realized T support need not extend beyond choose(cap,2).
        self.assertLessEqual(max(hist),6*5//2)
        self.assertGreater(max(hist),0)
        self.assertEqual(info['posterior_updates'],1)
        self.assertEqual(info['F1']['channel']['max_projected_omission'],0.)


if __name__=='__main__':unittest.main()
