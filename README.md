# Moroccan Crop Classification Challenge

Build a quantum or hybrid model to classify **alfalfa, maize and wheat** from satellite features. You receive **914 training, 150 validation and 221 unlabeled test fields**. Start with the [student brief (PDF)](Student_Challenge_Brief.pdf), then work through [the notebook](notebooks/00_start_here.ipynb).

The simple starter QNN performs worse than the classical baseline. Your task is to improve the approach **or replace it with another quantum or hybrid model**. You do not have to extend this QNN. A well-supported comparison can earn full analysis credit even if it does not beat the baseline.

## Install and run

Use Python 3.12 from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name crop-challenge --display-name "Crop challenge"
```

Open `notebooks/00_start_here.ipynb` in a notebook editor, select **Crop challenge**, and run all cells. The pinned environment uses Qiskit 2.4.2 and Qiskit Machine Learning 0.9.1. All data is included; local simulation needs no IBM token. To execute and export without an editor:

```bash
.venv/bin/python -m jupyter nbconvert --to html --execute notebooks/00_start_here.ipynb --output-dir outputs
```

The default fit uses 100 fixed training fields. Set `USE_FULL_TRAINING = True` for the main 914-field comparison. Later plots read saved full-training results, so changing the quick fit does not replace those plots. Your generated files go in `outputs/`.

## Starting results

Validation results using all 914 training fields and seeds 0, 1 and 2:

| Model | Mean macro-F1 | Mean accuracy |
| --- | ---: | ---: |
| Classical baseline: RBF SVM | 0.8797 | 88.00% |
| Simple four-qubit QNN | 0.8557 | 85.56% |
| Small classical replacement control | 0.5814 | 65.11% |

The SVM is deterministic. The QNN has eight trainable angles; the control has eleven parameters, a disclosed 37.5% mismatch. Saved predictions, histories and scores are in `results/`. Keep them as reference results and save your own runs separately. These results do not demonstrate a quantum advantage.

## Where to work

- **Notebook:** explains each step and output, with a table mapping every Python file to the changes you can make.
- **`challenge/models.py`:** encoding, QNN and classical control; **`challenge/submission.py`:** your final model's `fit`/`predict` interface.
- **`challenge/coverage.py`:** reproducible training-size comparisons; see [coverage commands](docs/COVERAGE.md).
- **`data/`:** supplied tables and PCA views; read [feature definitions and limitations](data/README.md). Keep these files unchanged.
- **`configs/`:** baseline configuration and recorded starter settings. `reference.json` is loaded by the baseline; `pilot.json` records settings and is not an automatic configuration loader.

Fit transformations on training data only. Use validation data for model selection and test data only for final predictions. “Development data” in the original brief means validation data. Keep all three classes and the supplied spatial splits; IDs are never model inputs. Larger datasets require a supplied, checked extension: adding rows alone cannot guarantee better accuracy.

## Submit

Submit one runnable notebook or code package with configurations and results, plus **three to five slides**. Run selected stochastic models and controls with seeds **0, 1 and 2**. Report macro-F1, accuracy, each class's precision/recall, confusion matrices, runtime and model resources.

Judging: predictive performance **30**, experimental quality and analysis **40**, model design and reproducibility **30**. Optional verified IBM inference may earn **+5**. See the [evaluation guide](docs/EVALUATION.md) and [method ideas](docs/HINTS.md).

Use exact local simulation with at most **eight qubits**. Plan for up to **15 minutes per starter seed** and **35 minutes per final seed**, including preprocessing and prediction; final judging-machine details will be announced. **Submission deadline, timezone and route: to be announced.**

Hardware inference is optional and disabled by default. Only that section needs `requirements-hardware.txt`, credentials in your environment and permission to use the chosen account/backend. Train locally first.

## Data source

Adapted from Ouassanouan et al., [Moroccan Crop and Irrigation Dataset v2](https://doi.org/10.6084/m9.figshare.28486022.v2), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Filtering, daily medians, spatial splitting and training-only PCA were applied. Labels surveyed on 23–27 April are **assumed** to apply to imagery from 25 April 2024; crop persistence is not independently verified. Cloud handling and band calibration are undocumented. Results describe this local Doukkala cohort.
