import itertools
import unittest
import numpy as np
from scipy.stats import hypergeom
from titraselect import remaining_error_bound, contrast_scores, calibrate, invariant_ratios


class Guarantees(unittest.TestCase):
    def test_hypergeometric_coverage_exhaustive(self):
        cases = 0
        for M in range(2, 19):
            for n in range(1, M):
                for total_errors in range(M + 1):
                    for delta in [.01, .05, .2]:
                        failure = 0.
                        for k in range(max(0, n-(M-total_errors)), min(n,total_errors)+1):
                            bound = remaining_error_bound(M,n,k,delta)
                            actual = (total_errors-k)/(M-n)
                            if actual > bound + 1e-12:
                                failure += hypergeom.pmf(k,M,total_errors,n)
                        self.assertLessEqual(failure,delta+1e-10)
                        cases += 1
        print('Exhaustive finite-population coverage cases:', cases)

    def test_contrast_sign_and_affine_invariance(self):
        rng=np.random.default_rng(914)
        c=np.r_[np.logspace(-6,-11,11),0]
        tau=1e-8
        K=10**rng.uniform(-11,-6,500)
        A=10**rng.uniform(1,5,500)
        B=10**rng.uniform(0,3,500)
        F=B[:,None]+A[:,None]*c/(K[:,None]+c)
        E=np.full(F.shape,.1)
        pairs=list(itertools.combinations(range(11),2))
        scores=contrast_scores(np.log10(F),E,c,tau,pairs)
        self.assertTrue(np.all((scores>=0)==(K[:,None]<=tau)))
        F2=7.3*F+123.
        scores2=contrast_scores(np.log10(F2),E,c,tau,pairs)
        self.assertTrue(np.array_equal(scores>=0,scores2>=0))
        print('Noiseless contrast signs checked:',scores.size*2)

    def test_no_test_label_leakage(self):
        rng=np.random.default_rng(81)
        s=rng.normal(size=1000)
        y=(s+rng.normal(size=1000)*.15>=0).astype(int)
        train=np.arange(200);cal=np.arange(200,500);test=np.arange(500,1000)
        for protected in [False,True]:
            a,accept,pred=calibrate(s,y,train,cal,test,protect_hits=protected)
            y2=y.copy();y2[test]=1-y2[test]
            b,accept2,pred2=calibrate(s,y2,train,cal,test,protect_hits=protected)
            np.testing.assert_array_equal(accept,accept2)
            np.testing.assert_array_equal(pred,pred2)
            self.assertEqual(a['gates'],b['gates'])

    def test_ratio_robustness(self):
        rng=np.random.default_rng(72)
        F=rng.uniform(1,100,(100,12))
        F[:,-1]=1
        gain=10**rng.uniform(-2,2,(100,1));offset=rng.uniform(0,100,(100,1))
        a=invariant_ratios(np.log10(F),[0,3])
        b=invariant_ratios(np.log10(gain*F+offset),[0,3])
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)



if __name__=='__main__': unittest.main()
