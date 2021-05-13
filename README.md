# torch_truncnorm
Truncated Normal distribution in PyTorch. The module provides:
- `TruncatedStandardNormal` class - zero mean unit variance of the parent Normal distribution, parameterized by the 
  cut-off range `[a, b]` (similar to `scipy.stats.truncnorm`);
- `TruncatedNormal` class - a wrapper with extra `loc` and `scale` parameters of the parent Normal distribution;
- Differentiability wrt parameters of the distribution;
- Batching support.

# Why
I just needed differentiation with respect to the parameters of the distribution and found that this distribution is not 
bundled in torch.distributions as of 1.6.0.

# Known issues
`icdf` is rather unstable and most of the times will produce infinity.

# Links
https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
