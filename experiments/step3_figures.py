"""TechQuantum, step 3: figures for the pitch (outputs/figures/*.png) from saved results.

Requires step2 and step3_learning_curve to have run. The bandwidth sweep (figure 1) is
recomputed here over a wide scale range using training CV only. Run from the repo root:

    .venv/Scripts/python.exe -m experiments.step3_figures
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from challenge.coverage import LABELS, metric_vector
from challenge.data import ROOT
from challenge.quantum_kernel import encode, feature_map, fidelity
from experiments.step2_quantum_kernel import CS, cv_f1

OUT = ROOT / 'outputs/figures'
S2, S3 = ROOT / 'outputs/step2', ROOT / 'outputs/step3'
# Reference categorical palette (light): slots 1-3, neutral gray for context models.
COLOR = {'quantum_zz': '#2a78d6', 'rbf_pca8': '#eb6834', 'linear_pca8': '#1baf7a',
         'reference_svm13': '#8a8985', 'qnn_starter': '#8a8985'}
NAME = {'quantum_zz': 'Quantum kernel (8 qubits)', 'rbf_pca8': 'RBF SVM (pca8)',
        'linear_pca8': 'Linear SVM (pca8)', 'reference_svm13': 'Reference SVM (13 feat.)',
        'qnn_starter': 'Starter QNN (4 qubits)'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .8,
                     'axes.axisbelow': True, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / f'{name}.png', dpi=170, bbox_inches='tight'); plt.close(fig)
    print('saved', name)


def fig_bandwidth():
    with np.load(ROOT / 'data/views/pca8.npz', allow_pickle=False) as f:
        X, y = f['X_train'], f['y_train']
    c = feature_map(8, 2); rows = []
    for s in [0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]:
        S = encode(X, s, c); K = fidelity(S, S); off = K[~np.eye(len(K), dtype=bool)]
        best = max(cv_f1(K, y, C) for C in CS)
        rows.append(dict(scale=s, offdiag_mean=off.mean(), offdiag_std=off.std(), cv_f1=best[0], cv_sd=best[1]))
    d = pd.DataFrame(rows); d.to_csv(OUT / 'bandwidth_sweep.csv', index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].fill_between(d.scale, d.offdiag_mean - d.offdiag_std, d.offdiag_mean + d.offdiag_std,
                       color=COLOR['quantum_zz'], alpha=.15, lw=0)
    ax[0].plot(d.scale, d.offdiag_mean, color=COLOR['quantum_zz'], lw=2, marker='o', ms=6)
    ax[0].axhline(1 / 256, color=MUTED, lw=1, ls='--')
    ax[0].text(.8, 1 / 256 + .03, '1/256 (all states look alike)', ha='right', color=MUTED, fontsize=8)
    ax[0].set(xscale='log', xlabel='Encoding scale (log)', ylabel='Kernel value between different fields',
              title='Kernel concentration: similarity collapses')
    ax[1].errorbar(d.scale, d.cv_f1, yerr=d.cv_sd, color=COLOR['quantum_zz'], lw=2, marker='o', ms=6, capsize=3)
    sel = d.loc[d.cv_f1.idxmax()]
    ax[1].annotate(f'selected: scale {sel.scale:g}, CV F1 {sel.cv_f1:.3f}', (sel.scale, sel.cv_f1),
                   xytext=(0.025, 0.62), color=INK, fontsize=8, arrowprops=dict(arrowstyle='-', color=MUTED))
    ax[1].set(xscale='log', xlabel='Encoding scale (log)', ylabel='5-fold CV macro-F1 (training)',
              title='Accuracy follows the kernel spread')
    ticks = [0.02, 0.05, 0.1, 0.2, 0.5, 0.8]
    for a in ax:
        a.set_xticks(ticks, [f'{t:g}' for t in ticks]); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    save(fig, 'fig1_kernel_bandwidth')


def fig_validation():
    res = json.loads((S2 / 'results.json').read_text())['results']
    summary = {r['model']: r for r in json.loads((ROOT / 'results/summary.json').read_text())['metrics']}
    groups = json.loads((ROOT / 'data/validation_groups.json').read_text())
    enc = {c: i for i, c in enumerate(LABELS)}
    rows = []
    for m in ['quantum_zz', 'rbf_pca8', 'linear_pca8', 'reference_svm13']:
        p = pd.read_csv(S2 / f'{m}_predictions.csv')
        y, pr = p.target.map(enc).to_numpy(), p.prediction.map(enc).to_numpy()
        g = np.array([groups[i] for i in p.sample_id]); u, idx = np.unique(g, return_inverse=True)
        rng = np.random.default_rng(2026); boots = []
        while len(boots) < 2000:
            w = np.bincount(rng.integers(len(u), size=len(u)), minlength=len(u))[idx]
            if all(w[y == k].sum() > 0 for k in range(3)): boots.append(metric_vector(y, pr[None], w)[0])
        rows.append((m, res[m]['metrics']['macro_f1'], *np.quantile(boots, [.025, .975])))
    q = summary['qnn']; rows.append(('qnn_starter', q['macro_f1_mean'], np.nan, np.nan))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, (m, f1, lo, hi) in enumerate(rows[::-1]):
        ax.barh(i, f1, height=.55, color=COLOR[m], edgecolor='#fcfcfb', linewidth=2)
        if not np.isnan(lo): ax.plot([lo, hi], [i, i], color=INK, lw=1.2)
        ax.text(0.505, i, f'{f1:.3f}', va='center', ha='left', color='white', fontsize=9, fontweight='bold')
    ax.set_yticks(range(len(rows)), [NAME[r[0]] for r in rows[::-1]])
    ax.set(xlim=(.5, 1), xlabel='Validation macro-F1 (150 fields)',
           title='Quantum kernel beats the reference and the starter QNN, not the matched SVMs')
    ax.text(0, -.3, 'Lines: 95% spatial-block bootstrap interval. Starter QNN: mean of 3 seeds (supplied results).',
            color=MUTED, fontsize=7.5, transform=ax.transAxes)
    ax.grid(axis='y', visible=False)
    save(fig, 'fig2_validation_scores')


def fig_confusion():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for a, m in zip(ax, ['quantum_zz', 'rbf_pca8']):
        p = pd.read_csv(S2 / f'{m}_predictions.csv')
        cm = confusion_matrix(p.target, p.prediction, labels=LABELS); n = cm / cm.sum(1, keepdims=True)
        a.imshow(n, vmin=0, vmax=1, cmap='Blues'); a.grid(False)
        for i in range(3):
            for j in range(3):
                a.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='white' if n[i, j] > .55 else INK)
        a.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
              xlabel='Predicted', ylabel='True', title=NAME[m])
    save(fig, 'fig3_confusion')


def fig_learning():
    d = pd.read_csv(S3 / 'learning_curve_summary.csv')
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m in ['quantum_zz', 'rbf_pca8', 'linear_pca8']:
        s = d[d.model == m]
        ax.errorbar(s.n_train, s.macro_f1_mean, yerr=s.macro_f1_sd, color=COLOR[m], lw=2, marker='o',
                    ms=7, capsize=3, label=NAME[m])
        ax.text(s.n_train.iloc[-1] + 15, s.macro_f1_mean.iloc[-1], NAME[m], va='center', color=INK, fontsize=8)
    ax.set(xlabel='Training fields (stratified subsets; preprocessing refit each time)',
           ylabel='Validation macro-F1', xticks=sorted(d.n_train.unique()), xlim=(180, 1180),
           title='Learning curve: quantum kernel gains most from more data, then plateaus')
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    ax.text(0, -.22, 'Error bars: SD over 3 subset draws (seeds 0/1/2); at 914 all seeds use every row.',
            transform=ax.transAxes, color=MUTED, fontsize=7.5)
    save(fig, 'fig4_learning_curve')


def fig_runtime():
    res = json.loads((S2 / 'results.json').read_text())['results']
    t = {m: res[m]['fit_seconds'] + res[m]['predict_seconds'] for m in ['quantum_zz', 'rbf_pca8', 'reference_svm13']}
    t['qnn_starter'] = 4124.0  # measured on this machine in step 1 (seed 0, 914 fields)
    order = ['qnn_starter', 'quantum_zz', 'reference_svm13', 'rbf_pca8']
    fig, ax = plt.subplots(figsize=(7.5, 3))
    for i, m in enumerate(order[::-1]):
        ax.barh(i, t[m], height=.55, color=COLOR[m], edgecolor='#fcfcfb', linewidth=2)
        ax.text(t[m] * 1.15, i, f'{t[m]:.2f} s' if t[m] < 100 else f'{t[m] / 60:.0f} min', va='center', color=INK, fontsize=9)
    ax.set_yticks(range(len(order)), [NAME[m] for m in order[::-1]])
    ax.set(xscale='log', xlim=(.01, 2e4), xlabel='Fit + predict time, 914 training fields (s, log)',
           title='Exact simulation: one state per field instead of one per optimizer step')
    ax.grid(axis='y', visible=False)
    ax.text(0, -.38, 'All timings on the same Windows laptop. The supplied Linux run of the starter QNN took ~55 s per seed.',
            color=MUTED, fontsize=7.5, transform=ax.transAxes)
    save(fig, 'fig5_runtime')


def fig_circuit():
    c = feature_map(8, 1)  # one repetition shown; the model stacks this block twice (reps=2)
    fig = c.draw('mpl', fold=-1, scale=.6, style={'backgroundcolor': '#fcfcfb'})
    fig.savefig(OUT / 'fig0_circuit.png', dpi=170, bbox_inches='tight'); plt.close(fig)
    print('saved fig0_circuit')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig_circuit(); fig_validation(); fig_confusion(); fig_learning(); fig_runtime(); fig_bandwidth()
