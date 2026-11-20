"""TechQuantum, step 2: quantum fidelity kernel vs matched classical kernels on pca8.

Question: with the input view, training rows, SVM learner and tuning budget held
fixed, does replacing a classical kernel (RBF, linear) by an 8-qubit ZZ fidelity
kernel change validation macro-F1?

Selection uses stratified 5-fold CV on the training set only (same folds, same
number of configurations for the quantum kernel and the RBF control). Validation
is touched once per selected model. Run from the repository root:

    .venv/Scripts/python.exe -m experiments.step2_quantum_kernel
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from challenge.coverage import LABELS, metric_vector
from challenge.data import ROOT, load_data
from challenge.classical import reference
from challenge.metrics import score
from challenge.quantum_kernel import encode, feature_map, fidelity

OUT = ROOT / 'outputs/step2'
# Quantum bandwidth (input multiplier). Round 1 used [0.1, 0.2, 0.3, 0.5, 0.8]: CV picked the
# edge value 0.1 and scales >= 0.2 concentrated the kernel (off-diagonal mean ~1/256). Round 2
# shifts the same five-value budget downward. Both rounds used training CV only.
SCALES = [0.02, 0.04, 0.07, 0.1, 0.15]
GAMMAS = [0.01, 0.03, 0.1, 0.3, 1.0]         # RBF bandwidth, same grid size
CS = [0.1, 1, 10, 100, 1000]
FOLDS = StratifiedKFold(5, shuffle=True, random_state=0)


def cv_f1(K, y, C):
    scores = []
    for tr, te in FOLDS.split(K, y):
        svm = SVC(kernel='precomputed', C=C).fit(K[np.ix_(tr, tr)], y[tr])
        scores.append(f1_score(y[te], svm.predict(K[np.ix_(te, tr)]), average='macro'))
    return float(np.mean(scores)), float(np.std(scores))


def rbf(A, B, gamma):
    d = (A**2).sum(1)[:, None] + (B**2).sum(1)[None] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(d, 0))


def bootstrap(y, pred_a, pred_b, groups, samples=2000):
    """Paired spatial-block bootstrap of (b - a) on validation, as in challenge.coverage."""
    unique, index = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(2026); deltas = []
    while len(deltas) < samples:
        w = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[index]
        if all(w[y == k].sum() > 0 for k in range(3)):
            deltas.append(metric_vector(y, pred_b[None], w)[0] - metric_vector(y, pred_a[None], w)[0])
    d = metric_vector(y, pred_b[None], np.ones(len(y)))[0] - metric_vector(y, pred_a[None], np.ones(len(y)))[0]
    return dict(delta_macro_f1=float(d), ci95=[float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables, _, manifest, _ = load_data()
    with np.load(ROOT / 'data/views/pca8.npz', allow_pickle=False) as f:
        v = {k: f[k] for k in f.files}
    Xtr, Xva, ytr, yva = v['X_train'], v['X_validation'], v['y_train'], v['y_validation']
    circuit = feature_map(8, reps=2)
    grid, timing = [], {}

    # Quantum kernel: encode every field once per scale, then all CV fits reuse the matrix.
    states = {}
    for s in SCALES:
        t = perf_counter()
        Str, Sva = encode(Xtr, s, circuit), encode(Xva, s, circuit)
        Ktr = fidelity(Str, Str); timing[f'quantum_encode_scale_{s}'] = perf_counter() - t
        off = Ktr[~np.eye(len(Ktr), dtype=bool)]
        states[s] = (Str, Sva, Ktr)
        for C in CS:
            m, sd = cv_f1(Ktr, ytr, C)
            grid.append(dict(model='quantum_zz', bandwidth=s, C=C, cv_f1=m, cv_sd=sd,
                             offdiag_mean=float(off.mean()), offdiag_std=float(off.std())))
        print(f'scale {s}: off-diagonal kernel mean {off.mean():.3f} std {off.std():.3f}')
    for g in GAMMAS:
        Ktr = rbf(Xtr, Xtr, g); off = Ktr[~np.eye(len(Ktr), dtype=bool)]
        for C in CS:
            m, sd = cv_f1(Ktr, ytr, C)
            grid.append(dict(model='rbf_pca8', bandwidth=g, C=C, cv_f1=m, cv_sd=sd,
                             offdiag_mean=float(off.mean()), offdiag_std=float(off.std())))
    Klin = Xtr @ Xtr.T
    for C in CS:
        m, sd = cv_f1(Klin, ytr, C)
        grid.append(dict(model='linear_pca8', bandwidth=None, C=C, cv_f1=m, cv_sd=sd))
    grid = pd.DataFrame(grid); grid.to_csv(OUT / 'cv_grid.csv', index=False)
    best = grid.loc[grid.groupby('model').cv_f1.idxmax()].set_index('model')
    print(best[['bandwidth', 'C', 'cv_f1', 'cv_sd']])

    # Refit each selected model on all training rows and evaluate once on validation.
    groups = json.loads((ROOT / 'data/validation_groups.json').read_text())
    gva = np.array([groups[i] for i in v['ids_validation']])
    preds, results = {}, {}

    def record(name, pred, fit_s, pred_s, resources):
        preds[name] = pred
        results[name] = dict(metrics=score(yva, pred, LABELS), fit_seconds=fit_s, predict_seconds=pred_s,
                             resources=resources, selected=best.loc[name][['bandwidth', 'C', 'cv_f1']].to_dict()
                             if name in best.index else 'frozen reference (13 features)')
        pd.DataFrame(dict(sample_id=v['ids_validation'], target=yva, prediction=pred)).to_csv(
            OUT / f'{name}_predictions.csv', index=False)

    s, C = best.loc['quantum_zz', 'bandwidth'], best.loc['quantum_zz', 'C']
    t = perf_counter(); Str = encode(Xtr, s, circuit); svm = SVC(kernel='precomputed', C=C).fit(fidelity(Str, Str), ytr)
    fit_s = perf_counter() - t
    t = perf_counter(); p = svm.predict(fidelity(encode(Xva, s, circuit), Str)); pred_s = perf_counter() - t
    flat = circuit.decompose()
    record('quantum_zz', p, fit_s, pred_s, dict(qubits=8, depth=flat.depth(),
           two_qubit_gates=sum(len(i.qubits) == 2 for i in flat.data), trainable_quantum=0,
           support_vectors=int(svm.n_support_.sum()), simulated_states=len(Xtr) + len(Xva),
           kernel_entries=len(Xtr) ** 2 + len(Xva) * len(Xtr), simulation='exact Statevector'))

    g, C = best.loc['rbf_pca8', 'bandwidth'], best.loc['rbf_pca8', 'C']
    t = perf_counter(); svm = SVC(kernel='rbf', gamma=g, C=C).fit(Xtr, ytr); fit_s = perf_counter() - t
    t = perf_counter(); p = svm.predict(Xva); pred_s = perf_counter() - t
    record('rbf_pca8', p, fit_s, pred_s, dict(support_vectors=int(svm.n_support_.sum())))

    C = best.loc['linear_pca8', 'C']
    t = perf_counter(); svm = SVC(kernel='linear', C=C).fit(Xtr, ytr); fit_s = perf_counter() - t
    t = perf_counter(); p = svm.predict(Xva); pred_s = perf_counter() - t
    record('linear_pca8', p, fit_s, pred_s, dict(support_vectors=int(svm.n_support_.sum())))

    F = manifest['features']; tr, va = tables['train'], tables['validation']
    t = perf_counter(); ref = reference().fit(tr[F], tr.target); fit_s = perf_counter() - t
    t = perf_counter(); p = ref.predict(va[F]); pred_s = perf_counter() - t
    record('reference_svm13', p, fit_s, pred_s, dict(support_vectors=int(ref[-1].n_support_.sum())))

    enc = {c: i for i, c in enumerate(LABELS)}; y = np.array([enc[c] for c in yva])
    P = {k: np.array([enc[c] for c in p]) for k, p in preds.items()}
    comparisons = {f'quantum_zz - {k}': bootstrap(y, P[k], P['quantum_zz'], gva)
                   for k in ['rbf_pca8', 'linear_pca8', 'reference_svm13']}

    report = dict(question=__doc__.split('\n\n')[1], selection='stratified 5-fold CV on training only',
                  configurations_tried={'quantum_zz': len(SCALES) * len(CS), 'rbf_pca8': len(GAMMAS) * len(CS),
                                        'linear_pca8': len(CS)},
                  determinism='All four models are deterministic; seeds 0/1/2 give identical results.',
                  timing=timing, results=results, paired_block_bootstrap=comparisons)
    (OUT / 'results.json').write_text(json.dumps(report, indent=2, default=float))
    print('\nValidation (150 fields):')
    for k, r in results.items():
        m = r['metrics']
        print(f"{k:16s} macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}  recall {np.round(m['recall'], 2)}"
              f"  fit {r['fit_seconds']:.2f}s")
    for k, c in comparisons.items():
        print(f"{k:32s} dF1 {c['delta_macro_f1']:+.4f}  95% CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")


if __name__ == '__main__':
    main()
