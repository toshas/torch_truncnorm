import unittest

import torch
from scipy.stats import truncnorm
from TruncatedNormal import TruncatedNormal as TruncatedNormalPT


class TruncatedNormalSC:
    def __init__(self, loc, scale, a, b):
        self.loc = loc
        self.scale = scale
        self.alpha = (a - loc) / scale
        self.beta = (b - loc) / scale

    @property
    def mean(self):
        return truncnorm.stats(self.alpha, self.beta, loc=self.loc, scale=self.scale, moments='m')

    @property
    def variance(self):
        return truncnorm.stats(self.alpha, self.beta, loc=self.loc, scale=self.scale, moments='v')

    def cdf(self, value):
        return truncnorm.cdf(value, self.alpha, self.beta, loc=self.loc, scale=self.scale)

    def icdf(self, value):
        return truncnorm.ppf(value, self.alpha, self.beta, loc=self.loc, scale=self.scale)

    def log_prob(self, value):
        return truncnorm.logpdf(value, self.alpha, self.beta, loc=self.loc, scale=self.scale)

    @property
    def entropy(self):
        return truncnorm.entropy(self.alpha, self.beta, loc=self.loc, scale=self.scale)


class Tests(unittest.TestCase):

    def _test_numerical(self, loc, scale, a, b, do_icdf=True, places_logprob=4):
        pt = TruncatedNormalPT(loc, scale, a, b, validate_args=None)
        sc = TruncatedNormalSC(loc, scale, a, b)

        mean_sc = sc.mean
        mean_pt = pt.mean.numpy()
        self.assertAlmostEqual(mean_sc, mean_pt, places=4)

        var_sc = sc.variance
        var_pt = pt.variance.numpy()
        self.assertAlmostEqual(var_sc, var_pt, places=4)

        entropy_sc = sc.entropy
        entropy_pt = pt.entropy.numpy()
        self.assertAlmostEqual(entropy_sc, entropy_pt, places=4)

        N = 2
        for i in range(N):
            p = i / (N - 1)
            x = a + (b - a) * p

            cdf_sc = sc.cdf(x)
            cdf_pt = float(pt.cdf(torch.tensor(x)))
            self.assertAlmostEqual(cdf_sc, cdf_pt, places=4)

            log_prob_sc = sc.log_prob(x)
            log_prob_pt = float(pt.log_prob(torch.tensor(x)))
            self.assertAlmostEqual(log_prob_sc, log_prob_pt, places=places_logprob)

            if do_icdf:
                icdf_sc = sc.icdf(p)
                icdf_pt = float(pt.icdf(torch.tensor(p)))
                self.assertAlmostEqual(icdf_sc, icdf_pt, places=3)

    def test_simple(self):
        self._test_numerical(0., 1., -2., 0.)
        self._test_numerical(0., 1., -2., 1.)
        self._test_numerical(0., 1., -2., 2.)
        self._test_numerical(0., 1., -1., 0.)
        self._test_numerical(0., 1., -1., 1.)
        self._test_numerical(0., 1., -1., 2.)
        self._test_numerical(0., 1., 0., 1.)
        self._test_numerical(0., 1., 0., 2.)
        self._test_numerical(1., 2., 1., 2.)
        self._test_numerical(1., 2., 2., 4.)

    def test_precision(self):
        self._test_numerical(0., 1., 2., 3.)
        self._test_numerical(0., 1., 2., 4.)
        # self._test_numerical(0., 1., 2., 8.)  # fails due to .icdf returning inf
        self._test_numerical(0., 1., 2., 8., do_icdf=False)
        self._test_numerical(0., 1., 2., 16., do_icdf=False)
        self._test_numerical(0., 1., 2., 32., do_icdf=False)
        self._test_numerical(0., 1., 2., 64., do_icdf=False)
        self._test_numerical(0., 1., 2., 128., do_icdf=False)
        self._test_numerical(0., 1., 2., 256., do_icdf=False, places_logprob=2)
        self._test_numerical(0., 1., 2., 512., do_icdf=False, places_logprob=2)


if __name__ == '__main__':
    unittest.main()
