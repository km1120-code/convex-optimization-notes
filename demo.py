from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import *

OUT = Path(__file__).parent / 'results'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#34433e', 'text.color': '#34433e',
                     'axes.edgecolor': '#b6beb8', 'xtick.color': '#64726b',
                     'ytick.color': '#64726b', 'figure.facecolor': '#faf9f5',
                     'axes.facecolor': '#faf9f5', 'savefig.facecolor': '#faf9f5'})
GREEN, SAND, BLUE = '#235b48', '#b77c45', '#6b889a'

rng = np.random.default_rng(2026)
dimension, condition_number, steps = 24, 50, 200
q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
a = q @ np.diag(np.geomspace(1, condition_number, dimension)) @ q.T
b = rng.normal(size=dimension)
gd = objective_gaps(a, b, minimize(a, b, steps))
ag = objective_gaps(a, b, minimize(a, b, steps, accelerated=True))
with (OUT / 'convergence.csv').open('w') as f:
    writer = csv.writer(f); writer.writerow(['iteration', 'gradient_descent_gap', 'accelerated_gap'])
    writer.writerows(zip(range(steps + 1), gd, ag))
fig, ax = plt.subplots(figsize=(7.4, 4.4), layout='constrained')
ax.semilogy(np.maximum(gd, 1e-16), color=SAND, label='Gradient descent')
ax.semilogy(np.maximum(ag, 1e-16), color=GREEN, label='Nesterov acceleration')
ax.set(xlabel='Gradient steps', ylabel='Objective gap', title='The effect of acceleration')
ax.grid(alpha=.15); ax.legend(frameon=False, fontsize=9)
fig.savefig(OUT / 'convergence.png', dpi=180)
summary = {'experiment': 'positive-definite quadratic minimization', 'seed': 2026,
           'dimension': dimension, 'condition_number': condition_number, 'steps': steps,
           'gradient_descent_final_gap': float(gd[-1]), 'accelerated_final_gap': float(ag[-1]),
           'interpretation': 'One synthetic quadratic with known curvature constants; no neural-network claim.'}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + chr(10))
print(json.dumps(summary))
