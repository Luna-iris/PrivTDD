"""Validate the sampling channel using exhaustive public small graphs and discrete densities; tests read no paper graphs or experiment results."""
import itertools
import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import binom, hypergeom, laplace

from .sampled_channel import SampledChannel, exposure_distribution, _sampled_exposure_moments


def _enumerate_exposure(n,k,h):
    N = n+n%2
    m = N//2
    # For odd n, the public dummy is fixed on the left and excluded from enumeration; do not use the Fenwick counter under test here.
    population = list(range(N)) if n==N else list(range(1,N))
    outcomes = []
    for subset in itertools.combinations(population,h):
        edges = sum((v-m-u)%m < k for u in subset for v in subset if u<m<=v)
        outcomes.append(edges)
    return np.bincount(outcomes)/len(outcomes)


def _reference(channel,d,cap,t,y,cdf=False):
    law = _enumerate_exposure(channel.n,channel.k,min(d,cap))
    W,b = math.comb(d,2),float(channel.scale(cap))
    evaluate = laplace.cdf if cdf else laplace.pdf
    weighted = np.zeros_like(y,dtype=float)
    for a,pa in enumerate(law):
        if pa==0:
            continue
        for j in range(max(0,a+t-W),min(t,a)+1):
            pj = hypergeom.pmf(j,W,t,a)
            p1 = binom.pmf(np.arange(j+1),j,channel.p)
            p0 = binom.pmf(np.arange(a-j+1),a-j,channel.q)
            for z,pz in enumerate(np.convolve(p1,p0)):
                weighted += pa*pj*pz*evaluate(y,loc=z-channel.q*a,scale=b)
    return weighted


class TestSampledChannel(unittest.TestCase):
    def test_odd_even_rotations_and_exposure_moments(self):
        for n in (7,8,9):
            N = n+n%2
            for k in (1,2,N//2):
                for h in range(2,min(n,6)):
                    expected = _enumerate_exposure(n,k,h)
                    probs = exposure_distribution(N,k,h)
                    np.testing.assert_allclose(probs[:len(expected)],expected,atol=1e-13)
                    self.assertAlmostEqual(probs[len(expected):].sum(),0.,places=13)
                    a = np.arange(len(expected))
                    mean,var = _sampled_exposure_moments(h,k,N-1)
                    self.assertAlmostEqual(mean,expected@a,places=12)
                    self.assertAlmostEqual(var,expected@((a-mean)**2),places=12)

    def test_projected_likelihood_matches_full_enumeration(self):
        channel = SampledChannel(9,2,.7,.31)
        d,cap = 6,4
        y = np.array([-40.,-3.,0.,2.,38.])
        for t in (0,1,7,15):
            np.testing.assert_allclose(channel.density(d,cap,[t],y)[:,0],
                                       _reference(channel,d,cap,t,y),rtol=2e-13)
        # Caps 0 and 1 give deterministic points; d=0 with cap>=2 still has a positive release scale.
        np.testing.assert_array_equal(channel.density(3,1,[1],[-1,0,1])[:,0],[0,1,0])
        np.testing.assert_allclose(channel.density(0,3,[0],y)[:,0],laplace.pdf(y,scale=channel.scale(3)))
        self.assertTrue(np.isfinite(channel.logdensity(d,cap,[1,2],[-1e6,1e6])).all())
        self.assertTrue((channel.density(d,cap,[1,2],[-1e6,1e6])>0).all())

    def test_general_public_draws_calibrate_first_moment(self):
        # Outside the enumerable and special-k cases; exercise size-biased and ordinary public simulation separately.
        N,k = 1200,5
        for h in (11,23):
            mean,var = _sampled_exposure_moments(h,k,N-1)
            prob = exposure_distribution(N,k,h)
            a = np.arange(len(prob))
            self.assertAlmostEqual(prob.sum(),1,places=12)
            self.assertAlmostEqual(prob@a,mean,places=10)
            self.assertLess(abs(prob@((a-mean)**2)-var),.12*var)

    def test_public_discrete_kernel_against_enumerated_reference(self):
        # C(16,8)>8192, so the production channel uses 4096 public draws; the reference enumerates all 12870 subsets.
        channel = SampledChannel(16,2,.7,.32)
        y = np.linspace(-22,22,27)
        for t in (0,14,28):
            ref = _reference(channel,8,8,t,y)
            actual = channel.density(8,8,[t],y)[:,0]
            self.assertLess(np.max(abs(actual/ref-1)),.005)

    def test_high_exposure_against_exact_reference(self):
        channel = SampledChannel(8,4,.7,.32)
        d,cap,t = 6,5,8
        y = np.array([-22.,-1.,3.,30.])
        # Both the approximate and exact-mixture CDFs undergo Laplace smoothing; use an independent discrete kernel to measure the actual discrepancy.
        exact = _reference(channel,d,cap,t,y,cdf=True)
        predicted = [quad(lambda z: channel.density(d,cap,[t],[z])[0,0],-np.inf,x)[0] for x in y]
        self.assertLess(np.max(abs(predicted-exact)),.005)
        mass = quad(lambda z: channel.density(d,cap,[t],[z])[0,0],-np.inf,np.inf)[0]
        self.assertAlmostEqual(mass,1.,places=7)

    def test_million_vertex_numerics(self):
        ch = SampledChannel(3072441,308,.67,.32)
        d,cap = 3072440,1000
        W = math.comb(d,2)
        density = ch.density(d,cap,[0,W//2,W],[-1e6,-12,0.,10,1e6])
        self.assertTrue(np.isfinite(density).all())
        self.assertTrue((density>0).all())
        self.assertAlmostEqual(float(ch.scale(cap)),308/.32)


if __name__ == '__main__':
    unittest.main()
