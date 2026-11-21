import numpy as np

def validate_quadratic(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != (a.shape[0],):
        raise ValueError('Incompatible quadratic dimensions.')
    if not np.allclose(a, a.T):
        raise ValueError('A must be symmetric.')
    values = np.linalg.eigvalsh(a)
    if values[0] <= 0: raise ValueError('A must be positive definite.')
    return a, b, float(values[0]), float(values[-1])

def minimize(a, b, steps=200, accelerated=False):
    a, b, mu, smoothness = validate_quadratic(a, b)
    if steps < 0: raise ValueError('steps must be nonnegative.')
    beta = (np.sqrt(smoothness) - np.sqrt(mu)) / (np.sqrt(smoothness) + np.sqrt(mu)) if accelerated else 0.
    x = np.zeros_like(b); previous = x.copy(); trajectory = [x.copy()]
    for _ in range(steps):
        y = x + beta * (x - previous)
        next_x = y - (a @ y - b) / smoothness
        previous, x = x, next_x
        trajectory.append(x.copy())
    return np.asarray(trajectory)

def objective_gaps(a, b, trajectory):
    optimum = np.linalg.solve(a, b)
    difference = np.asarray(trajectory) - optimum
    return .5 * np.einsum('ni,ij,nj->n', difference, a, difference)
