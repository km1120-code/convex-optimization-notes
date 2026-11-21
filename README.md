# Convex Optimization Notes

Educational example created for an academic homepage prototype. This repository
uses synthetic data and makes no claim of a published research contribution.
The example is not evidence of any person's education, employment, or authorship.

## Gradient descent and acceleration on a quadratic

We minimize `f(x) = 0.5 x^T A x - b^T x`, where `A` is symmetric positive
definite. Its eigenvalues range from `mu = 1` to `L = 50`; the exact optimizer
is obtained with `numpy.linalg.solve(A, b)`.

Gradient descent uses step size `1/L`. The accelerated method uses
`y_k = x_k + beta (x_k - x_{k-1})`, then `x_{k+1} = y_k - grad f(y_k)/L`,
where `beta = (sqrt(L) - sqrt(mu))/(sqrt(L) + sqrt(mu))`. This constant momentum
assumes the strong-convexity and smoothness constants are known.

Both methods start at zero and use the same fixed 24-dimensional objective.
The plot shows `f(x_k) - f(x*)`, including iteration zero. Values below machine
precision are clipped only for log-scale visualization. Each CSV retains raw
nonnegative gaps computed from `0.5 (x - x*)^T A (x - x*)`.

This one quadratic illustrates convergence behavior. It is not a benchmark of
optimizer speed on neural networks.

## Links

- Repository: https://github.com/km1120-code/convex-optimization-notes
- Interactive homepage: https://km1120-code.github.io/
- Project explanation: https://km1120-code.github.io/projects/convex-optimization-notes.html

## Run

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py
python -m unittest discover -s tests -v
```

The demo writes a PNG figure, a CSV of the plotted observations, and a JSON
summary to `results/`. Random seeds and experimental settings are stored in the
summary. Results depend slightly on numerical-library versions.

## Files

- `core.py`: mathematical functions and algorithms.
- `demo.py`: reproducible experiment and plotting.
- `tests/test_core.py`: numerical and edge-case checks.
- `results/`: outputs from the provided experiment.

## Scope

These small CPU experiments illustrate specified models. Their observations do
not establish behavior on real data, general-purpose optimizers, or deployed
communication systems. Code is available under the MIT license.
