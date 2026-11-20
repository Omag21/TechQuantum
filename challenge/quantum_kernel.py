"""TechQuantum: fidelity quantum-kernel SVM on the eight-feature PCA view.

Each field x is encoded once into an 8-qubit state |psi(x)> = U_ZZ(scale * x)|0>,
simulated exactly with Statevector (no shots). The kernel is the state fidelity
K(x, x') = |<psi(x)|psi(x')>|^2, and a classical SVM is trained on that kernel.
Computing states once and taking inner products avoids one circuit run per pair,
which keeps the exact simulation fast on ordinary laptops.
"""
from time import perf_counter
import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC


def feature_map(n_qubits=8, reps=2, entanglement='linear'):
    return zz_feature_map(n_qubits, reps=reps, entanglement=entanglement, parameter_prefix='x')


def encode(X, scale, circuit):
    """Exact statevectors, shape (n_rows, 2**n_qubits)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != circuit.num_parameters or not np.isfinite(X).all():
        raise ValueError(f'Expected a finite (n,{circuit.num_parameters}) prepared view')
    params = list(circuit.parameters)  # sorted x[0]..x[n-1]
    return np.array([Statevector(circuit.assign_parameters(dict(zip(params, scale * row)))).data
                     for row in X])


def fidelity(states_a, states_b):
    return np.abs(states_a.conj() @ states_b.T) ** 2


class QuantumKernelSVC:
    def __init__(self, labels=('alfalfa', 'maize', 'wheat'), scale=0.5, C=10.0, reps=2,
                 entanglement='linear', class_weight=None):
        self.labels = np.asarray(labels); self.scale = scale; self.C = C
        self.reps = reps; self.entanglement = entanglement; self.class_weight = class_weight
        self.circuit = feature_map(8, reps, entanglement)

    def fit(self, X_train, y_train, seed=0):
        # The SVM is deterministic; seed is accepted for interface parity only.
        start = perf_counter()
        self.train_states = encode(X_train, self.scale, self.circuit)
        self.encode_seconds = perf_counter() - start
        K = fidelity(self.train_states, self.train_states)
        self.svm = SVC(kernel='precomputed', C=self.C, class_weight=self.class_weight).fit(K, y_train)
        self.fit_seconds = perf_counter() - start
        return self

    def kernel(self, X):
        return fidelity(encode(X, self.scale, self.circuit), self.train_states)

    def predict(self, X):
        return self.svm.predict(self.kernel(X))

    def resources(self):
        flat = self.circuit.decompose()
        n = len(self.train_states)
        return dict(qubits=self.circuit.num_qubits, trainable_quantum=0, depth=flat.depth(),
                    two_qubit_gates=sum(len(i.qubits) == 2 for i in flat.data),
                    feature_map=f'ZZ reps={self.reps} {self.entanglement}', scale=self.scale, C=self.C,
                    circuit_simulations_fit=n, support_vectors=int(self.svm.n_support_.sum()),
                    kernel_entries=n * n, encode_seconds=self.encode_seconds)
