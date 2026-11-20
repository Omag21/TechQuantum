"""TechQuantum, step 3: learning curve, quantum kernel vs matched classical kernels.

For each training fraction (25%, 50%, 100%) and seed 0/1/2, draw one stratified subset of
the 914 training fields (seed = subset draw; at 100% all seeds use every row). On that
subset, refit StandardScaler -> PCA(8) -> StandardScaler from the 13 raw features, then fit
the quantum-kernel SVM and the RBF/linear controls on the identical rows and inputs.
Hyperparameters stay fixed at their full-data CV selections (step 2), for all models.
Validation (150 fields) is unchanged. Run from the repository root:

    .venv/Scripts/python.exe -m experiments.step3_learning_curve
"""
import json
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from challenge.coverage import LABELS
from challenge.data import ROOT, load_data
from challenge.metrics import score
from challenge.quantum_kernel import QuantumKernelSVC

OUT = ROOT / 'outputs/step3'
FRACTIONS = [0.25, 0.5, 1.0]
SEEDS = [0, 1, 2]
SEL = json.loads((ROOT / 'outputs/step2/results.json').read_text())['results']


def models():
    q, r, l = SEL['quantum_zz']['selected'], SEL['rbf_pca8']['selected'], SEL['linear_pca8']['selected']
    return {
        'quantum_zz': lambda: QuantumKernelSVC(LABELS, scale=q['bandwidth'], C=q['C']),
        'rbf_pca8': lambda: SVC(kernel='rbf', gamma=r['bandwidth'], C=r['C']),
        'linear_pca8': lambda: SVC(kernel='linear', C=l['C']),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables, _, manifest, _ = load_data()
    F = manifest['features']; tr, va = tables['train'], tables['validation']
    X, y, Xva, yva = tr[F].to_numpy(), tr.target.to_numpy(), va[F].to_numpy(), va.target.to_numpy()
    rows = []
    for frac in FRACTIONS:
        for seed in SEEDS:
            idx = np.arange(len(X)) if frac == 1.0 else train_test_split(
                np.arange(len(X)), train_size=frac, stratify=y, random_state=seed)[0]
            prep = make_pipeline(StandardScaler(), PCA(8, svd_solver='full'), StandardScaler())
            A, B = prep.fit_transform(X[idx]), prep.transform(Xva)
            for name, make in models().items():
                t = perf_counter(); m = make().fit(A, y[idx]); fit_s = perf_counter() - t
                t = perf_counter(); p = m.predict(B); pred_s = perf_counter() - t
                s = score(yva, p, LABELS)
                rows.append(dict(model=name, fraction=frac, n_train=len(idx), seed=seed,
                                 macro_f1=s['macro_f1'], accuracy=s['accuracy'],
                                 recall_alfalfa=s['recall'][0], recall_maize=s['recall'][1],
                                 recall_wheat=s['recall'][2], fit_seconds=fit_s, predict_seconds=pred_s))
                pd.DataFrame(dict(sample_id=va.sample_id, target=yva, prediction=p)).to_csv(
                    OUT / f'{name}_n{len(idx)}_seed{seed}_predictions.csv', index=False)
                print(f'{name:12s} n={len(idx):3d} seed={seed}  macro-F1 {s["macro_f1"]:.4f}  fit {fit_s:.1f}s')
    df = pd.DataFrame(rows); df.to_csv(OUT / 'learning_curve.csv', index=False)
    summary = df.groupby(['model', 'n_train']).agg(
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_sd=('macro_f1', lambda v: v.std(ddof=0)),
        accuracy_mean=('accuracy', 'mean'), fit_seconds_mean=('fit_seconds', 'mean')).reset_index()
    summary.to_csv(OUT / 'learning_curve_summary.csv', index=False)
    print('\n', summary.to_string(index=False))


if __name__ == '__main__':
    main()
