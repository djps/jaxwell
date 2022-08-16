# Jaxwell: GPU-accelerated, differentiable 3D iterative FDFD electromagnetic solver

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](LICENSE)
[![Continous Integration](https://github.com/djps/jaxwell/actions/workflows/main.yml/badge.svg)](https://github.com/djps/jaxwell/actions/workflows/main.yml) 
[![Coverage Status](https://coveralls.io/repos/github/djps/jaxwell/badge.svg)](https://coveralls.io/github/djps/jaxwell) 


Jaxwell is [JAX](https://github.com/google/jax) +
[Maxwell](https://github.com/stanfordnqp/maxwell-b):
an iterative solver for solving the finite-difference frequency-domain
Maxwell equations on NVIDIA GPUs.
Jaxwell is differentiable and fits seamlessly in the JAX ecosystem,
enabling nanophotonic inverse design problems to be cast as ML training jobs
and take advantage of the tsunami of innovations
in ML-specific hardware, software, and algorithms.

Jaxwell is a finite-difference frequency-domain solver that finds solutions to the time-harmonic form of Maxwell's equations, specifically:

$$
\left( \nabla \times \nabla \times - \omega^2 \varepsilon \right) \boldsymbol{E} = -i \omega \boldsymbol{J}
$$

for the electric field `E` via the API

```python
x, err = jaxwell.solve(params, z, b)
```

where `E → x`, `ω²ε → z`, `-iωJ → b`, `params` controls how the solve proceeds iteratively, and `err` is the error in the solution.

Following [meep](https://meep.readthedocs.io/en/latest/), Jaxwell uses
[dimensionless units](https://meep.readthedocs.io/en/latest/Introduction/#units-in-meep),
and assumes `μ = 1` everywhere,
and implements Shin's stretched-coordinate perfectly matched layers (SC-PML)
for absorbing boundary conditions.

You can install this version of Jaxwell with `pip install git+https://github.com/djps/jaxwell.git` 
but the easiest way to get started is to go straight to the example 
[colaboratory notebook](https://colab.research.google.com/gist/JesseLu/1e030fd8ca3fcbca7148cef315bc2ba7/differentiable-jaxwell.ipynb).

References:

- PMLs and diagonalization: [Shin2012] W. Shin and S. Fan. “Choice of the perfectly matched layer boundary condition for frequency-domain Maxwell's equations solvers.” Journal of Computational Physics 231 (2012): 3406–31
- COCG algorithm: [Gu2014] X. Gu, T. Huang, L. Li, H. Li, T. Sogabe and M. Clemens, "Quasi-Minimal Residual Variants of the COCG and COCR Methods for Complex Symmetric Linear Systems in Electromagnetic Simulations," in IEEE Transactions on Microwave Theory and Techniques, vol. 62, no. 12, pp. 2859-2867, Dec. 2014
 - BiCGStabL algorithm: [Sleijpen1993] G. Sleijpen and D. R. Fokkema. "BiCGstab(l) for linear equations involving unsymmetric matrices with complex spectrum."  Electronic Transactions on Numerical Analysis vol 1, pp. 11-32 (1993).

