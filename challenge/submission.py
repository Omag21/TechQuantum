"""TechQuantum submission: 8-qubit quantum fidelity kernel + SVM.

Input: the 13 raw features in manifest order (B1 ... B12, NDVI), as an (n, 13) array or a
DataFrame with those columns; IDs are never inputs. fit() learns the training-only
StandardScaler -> PCA(8) -> StandardScaler transform (identical to data/views/pca8.npz),
then trains the quantum-kernel SVM with the settings in configs/techquantum.json.
The model is deterministic: `seed` is accepted for interface parity and does not change results.
"""
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ROOT
from .quantum_kernel import QuantumKernelSVC

FEATURES = json.loads((ROOT / 'data/manifest.json').read_text())['features']
CONFIG = json.loads((ROOT / 'configs/techquantum.json').read_text())


def _raw(X):
    if hasattr(X, 'columns'):
        X = X[FEATURES]
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(FEATURES):
        raise ValueError(f'Expected (n,{len(FEATURES)}) raw features in order {FEATURES}')
    return X


class Submission:
    def __init__(self, labels=('alfalfa', 'maize', 'wheat')):
        fm = CONFIG['feature_map']
        self.prep = make_pipeline(StandardScaler(), PCA(fm['qubits'], svd_solver='full'), StandardScaler())
        self.model = QuantumKernelSVC(labels, reps=fm['reps'], entanglement=fm['entanglement'], **CONFIG['params'])

    def fit(self, X_train, y_train, seed=0):
        self.model.fit(self.prep.fit_transform(_raw(X_train)), np.asarray(y_train), seed=seed)
        return self

    def predict(self, X):
        return self.model.predict(self.prep.transform(_raw(X)))

    def resources(self):
        return self.model.resources()
