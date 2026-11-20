"""Build SYNTHETIC test images for the app from real validation fields.

Each image is 40x40 pixels x 12 bands: every pixel is the field's real band medians times a
small random texture, with a few invalid (zero) pixels. They let you exercise the upload path;
they are not real satellite scenes. Also writes a small CSV of validation fields (no labels).

    .venv/Scripts/python.exe -m app.make_samples
"""
import numpy as np
import tifffile

from app.inference import BANDS, LABELS_FR
from challenge.data import ROOT, load_data

OUT = ROOT / 'app/samples'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    va = load_data()[0]['validation']
    rng = np.random.default_rng(7)
    for crop in ['alfalfa', 'maize', 'wheat']:
        row = va[va.target == crop].iloc[0]
        base = row[BANDS].to_numpy(float)
        img = base[None, None, :] * rng.normal(1, 0.04, (40, 40, 1)) * rng.normal(1, 0.01, (40, 40, 12))
        img[rng.random((40, 40)) < 0.02] = 0  # a few invalid pixels
        name = f'exemple_synthetique_{LABELS_FR[crop].lower().replace("ï", "i").replace("é", "e")}.tif'
        tifffile.imwrite(OUT / name, img.astype('float32'), photometric='minisblack', planarconfig='contig')
        print('wrote', name, 'from', row.sample_id)
    va.sample(12, random_state=0).drop(columns='target').to_csv(OUT / 'exemple_parcelles.csv', index=False)
    print('wrote exemple_parcelles.csv')


if __name__ == '__main__':
    main()
