import unittest
from unittest.util import safe_repr
from warnings import warn

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

    def assertRelativelyEqual(self, first, second, tol=1e-6, error=1e-5, msg=None):
        if first == second:
            return
        diff = abs(first - second)
        rel = diff / max(abs(first), abs(second))
        if rel <= tol or diff <= error:
            return
        standardMsg = '%s != %s within tol=%s abs=%s (rel=%s diff=%s)' % (safe_repr(first), safe_repr(second),
            safe_repr(tol), safe_repr(error), safe_repr(rel), safe_repr(diff))
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

    def _test_numerical(self, loc, scale, a, b, do_icdf=True):
        pt = TruncatedNormalPT(loc, scale, a, b, validate_args=None)
        sc = TruncatedNormalSC(loc, scale, a, b)

        mean_sc = sc.mean
        mean_pt = pt.mean.numpy()
        self.assertRelativelyEqual(mean_sc, mean_pt)

        var_sc = sc.variance
        var_pt = pt.variance.numpy()
        self.assertRelativelyEqual(var_sc, var_pt)

        entropy_sc = sc.entropy
        entropy_pt = pt.entropy.numpy()
        self.assertRelativelyEqual(entropy_sc, entropy_pt)

        N = 10
        for i in range(N):
            p = i / (N - 1)
            if torch.isinf(torch.tensor(a)):
                x = b - scale * i / N
            elif torch.isinf(torch.tensor(b)):
                x = a + scale * i / N
            else:
                x = a + (b - a) * p

            cdf_sc = sc.cdf(x)
            cdf_pt = float(pt.cdf(torch.tensor(x)))
            self.assertRelativelyEqual(cdf_sc, cdf_pt)

            log_prob_sc = sc.log_prob(x)
            log_prob_pt = float(pt.log_prob(torch.tensor(x)))
            self.assertRelativelyEqual(log_prob_sc, log_prob_pt)

            if do_icdf:
                icdf_sc = sc.icdf(p)
                icdf_pt = float(pt.icdf(torch.tensor(p)))
                self.assertRelativelyEqual(icdf_sc, icdf_pt, tol=1e-4, error=1e-3)

    def _test_grad(self, loc, scale, a, b, grad_point):
        loc = torch.nn.parameter.Parameter(torch.tensor(loc))
        scale = torch.nn.parameter.Parameter(torch.tensor(scale))
        pt = TruncatedNormalPT(loc, scale, a, b)
        grads = torch.autograd.grad(pt.log_prob(grad_point), [loc, scale])
        self.assertFalse(any([grad.isnan() for grad in grads]))

    def test_grad(self):
        self._test_grad(0., 1., -2., 0., -1.)
        self._test_grad(0., 1., -2., torch.inf, -1.)
        self._test_grad(0., 1., -torch.inf, 0., -1.)

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
        self._test_numerical(0., 1., -2., torch.inf)
        self._test_numerical(0., 1., -torch.inf, 0.)
        self._test_numerical(1., 2., 2., torch.inf)
        self._test_numerical(1., 2., -torch.inf, 4.)

    def test_precision(self):
        self._test_numerical(0., 1., 2., 3.)
        self._test_numerical(0., 1., 2., 4.)
        # self._test_numerical(0., 1., 2., 8.)  # fails due to .icdf returning inf
        self._test_numerical(0., 1., 2., 8., do_icdf=False)
        self._test_numerical(0., 1., 2., 16., do_icdf=False)
        self._test_numerical(0., 1., 2., 32., do_icdf=False)
        self._test_numerical(0., 1., 2., 64., do_icdf=False)
        self._test_numerical(0., 1., 2., 128., do_icdf=False)
        self._test_numerical(0., 1., 2., 256., do_icdf=False)
        self._test_numerical(0., 1., 2., 512., do_icdf=False)

    def test_support(self):
        pt = TruncatedNormalPT(0., 1., -1., 2., validate_args=None)
        with self.assertRaises(ValueError) as e:
            pt.log_prob(torch.tensor(-10))

        self.assertFalse(str(e.exception) != 'The value argument must be within the support' and
                         str(e.exception) != 'Expected value argument (Tensor of shape ()) to be within the support (Interval(lower_bound=-1.0, upper_bound=2.0)) of the distribution TruncatedNormal(a: -1.0, b: 2.0), but found invalid values:\n-10.0')

    def test_cuda(self):
        if not torch.cuda.is_available():
            warn('Skipping CUDA tests')
            return
        loc = torch.tensor([0., 1.]).cuda()
        scale = torch.tensor([1., 2.]).cuda()
        a = torch.tensor([-1., -10.]).cuda()
        b = torch.tensor([0., 100.]).cuda()
        pt = TruncatedNormalPT(loc, scale, a, b, validate_args=None)
        s = pt.rsample()
        self.assertTrue(s.is_cuda)


if __name__ == '__main__':
    unittest.main()
