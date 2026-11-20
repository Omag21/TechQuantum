"""TechQuantum app: turn a Sentinel-2 field image (or band table) into a crop prediction.

The model needs the 13 spectral values the challenge uses (B1...B12 + NDVI, reflectance x 10000).
A multispectral image of ONE field is reduced to those values by a per-band median over valid
pixels, the same summary the dataset uses. An ordinary RGB photo has 3 bands and cannot be used.
"""
import io
import numpy as np
import pandas as pd

from challenge.coverage import LABELS
from challenge.data import ROOT, load_data
from challenge.submission import FEATURES, Submission

BANDS = FEATURES[:-1]  # 12 spectral bands, NDVI computed
LABELS_FR = {'alfalfa': 'Luzerne', 'maize': 'Maïs', 'wheat': 'Blé'}


class CropPredictor:
    """Trains the submitted quantum-kernel model once on the 914 training fields."""

    def __init__(self):
        tables, _, _, _ = load_data()
        self.train, self.validation = tables['train'], tables['validation']
        self.model = Submission(LABELS).fit(self.train, self.train.target)
        self.class_spectra = self.train.groupby('target')[FEATURES].median()
        # Familiarity reference: how close validation fields are to their nearest training field.
        self.familiar_cutoff = float(np.quantile(self._nearest_fidelity(self.validation[FEATURES]), 0.02))

    def _kernel(self, raw):
        m = self.model
        return m.model.kernel(m.prep.transform(np.asarray(raw, dtype=float)))

    def _nearest_fidelity(self, raw):
        return self._kernel(raw).max(axis=1)

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw[FEATURES]
        K = self._kernel(raw)
        svm = self.model.model.svm
        scores = svm.decision_function(K)
        relative = np.exp(scores - scores.max(1, keepdims=True)); relative /= relative.sum(1, keepdims=True)
        pred = svm.classes_[scores.argmax(1)]
        out = pd.DataFrame({'prediction': pred, 'culture': [LABELS_FR[p] for p in pred],
                            'similarite_max': K.max(1)}, index=raw.index)
        out['familier'] = out.similarite_max >= self.familiar_cutoff
        for i, c in enumerate(svm.classes_):
            out[f'score_{c}'] = relative[:, i]
        return out


def read_image(data: bytes, name: str) -> np.ndarray:
    """Return an (H, W, bands) float array from a GeoTIFF/TIFF or .npy file."""
    name = name.lower()
    if name.endswith('.npy'):
        arr = np.load(io.BytesIO(data), allow_pickle=False)
    elif name.endswith(('.tif', '.tiff')):
        import tifffile
        arr = tifffile.imread(io.BytesIO(data))
    else:
        raise ValueError('Format non pris en charge. Utilisez une image GeoTIFF (.tif) multispectrale ou un .npy.')
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f'Image à {arr.ndim} dimension(s) : il faut une image multibande (hauteur × largeur × bandes).')
    axis = next((a for a in (2, 0, 1) if arr.shape[a] in (12, 13)), None)
    if axis is None:
        raise ValueError(f'Image de forme {arr.shape} : il faut 12 bandes Sentinel-2 ({", ".join(BANDS)}) '
                         f'ou 13 avec le NDVI. Une photo couleur (3 bandes) ne suffit pas.')
    return np.moveaxis(arr, axis, 2)


def summarize_field(img: np.ndarray):
    """Per-band median over valid pixels -> one row of the 13 model features, plus notes."""
    notes = []
    bands = img[..., :12]
    valid = np.isfinite(bands).all(-1) & (bands > 0).all(-1)
    if valid.sum() == 0:
        raise ValueError('Aucun pixel valide (toutes les valeurs sont nulles, négatives ou manquantes).')
    if valid.mean() < 1:
        notes.append(f'{(~valid).sum()} pixel(s) ignoré(s) (valeurs nulles, négatives ou manquantes).')
    px = bands[valid]
    if np.median(px) < 2:  # reflectance given in 0-1: convert to the dataset scale
        px = px * 10000; notes.append('Réflectance détectée entre 0 et 1 : convertie à l’échelle ×10 000 du jeu de données.')
    ndvi = (px[:, 7] - px[:, 3]) / (px[:, 7] + px[:, 3])
    row = dict(zip(BANDS, np.median(px, axis=0)))
    row['NDVI'] = float(np.median(ndvi))
    return pd.DataFrame([row])[FEATURES], int(valid.sum()), notes


def rgb_preview(img: np.ndarray) -> np.ndarray:
    """True-colour composite (B4, B3, B2) stretched to 0-1 for display."""
    rgb = img[..., [3, 2, 1]].astype(float)
    lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    return np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
