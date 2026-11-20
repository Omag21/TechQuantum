"""TechQuantum: final workflow. Fit Submission on the 914 training fields with seeds 0/1/2,
check it reproduces step 2 on validation, and write test predictions.

    .venv/Scripts/python.exe -m experiments.predict_test
"""
import json
from time import perf_counter

import numpy as np
import pandas as pd

from challenge.coverage import LABELS
from challenge.data import ROOT, load_data
from challenge.metrics import score
from challenge.submission import Submission

OUT = ROOT / 'outputs/final'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables, _, _, _ = load_data()
    tr, va, te = tables['train'], tables['validation'], tables['test']
    runs, test_preds = [], []
    for seed in [0, 1, 2]:
        t = perf_counter(); m = Submission(LABELS).fit(tr, tr.target, seed=seed); fit_s = perf_counter() - t
        t = perf_counter(); pv = m.predict(va); pt = m.predict(te); pred_s = perf_counter() - t
        s = score(va.target, pv, LABELS)
        runs.append(dict(seed=seed, validation=s, fit_seconds=fit_s, predict_seconds_val_and_test=pred_s,
                         resources=m.resources()))
        test_preds.append(pt)
        print(f'seed {seed}: validation macro-F1 {s["macro_f1"]:.4f}  fit {fit_s:.1f}s  predict {pred_s:.1f}s')
    assert all((p == test_preds[0]).all() for p in test_preds), 'Deterministic model must agree across seeds'
    step2 = pd.read_csv(ROOT / 'outputs/step2/quantum_zz_predictions.csv')
    assert (step2.prediction.to_numpy() == Submission(LABELS).fit(tr, tr.target).predict(va)).all(), \
        'Submission must reproduce the step 2 validation predictions'
    pd.DataFrame(dict(sample_id=te.sample_id, prediction=test_preds[0])).to_csv(OUT / 'test_predictions.csv', index=False)
    (OUT / 'runs.json').write_text(json.dumps(runs, indent=2, default=float))
    print('test class counts:', dict(zip(*np.unique(test_preds[0], return_counts=True))))


if __name__ == '__main__':
    main()
