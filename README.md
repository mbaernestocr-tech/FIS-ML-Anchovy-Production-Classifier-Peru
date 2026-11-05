# FIS-ML: Multi-Classifier Decision-Support System for Anchovy Production Levels — Supplementary code

Supplementary code for the manuscript **"FIS-ML: Multi-Classifier Decision-Support System for Anchovy Production Levels Using Climate and Fishery Data from Peru"**, submitted to *Expert Systems with Applications* (Elsevier), manuscript ESWA-D-25-29468, revised version.

Corresponding author: Ernesto David Cancho-Rodriguez, Universidad Nacional Mayor de San Marcos (UNMSM), Lima, Peru (ernesto.cancho@unmsm.edu.pe).

## Repository contents

| File | Description |
|---|---|
| `FIS_ML_Unified_Notebook_v7.ipynb` | **Main notebook (v7.0).** Performs the complete analysis of the revised manuscript. It is saved without outputs; the results of the reported run are in `DATA/results_v6_20260903_19h_54m_23s.json`. |
| `FIS_ML_Unified_Notebook_v7.py` | Plain-text export of the main notebook, for reading and code review. |
| `FIS_ML_Figures_Notebook_v2.ipynb` | **Figure notebook (v2).** Redraws the figures of the manuscript with English labels from the files of the reported run, without retraining any model. It is saved with its outputs and the 13 figures. |
| `FIS_ML_Figures_Notebook_v2.py` | Plain-text export of the figure notebook. |
| `DATA/results_v6_20260903_19h_54m_23s.json` | Results file of the reported run (see below). |
| `DATA/DS_DGSF_DESCARGAS_anonimizado.csv` | Fish landing records (PRODUCE open data). |
| `DATA/EXALMAR_EXALMC1_LM_OHLC_diario.csv`, `DATA/AUSTRAL_AUSTRAC1_LM_OHLC_diario.csv` | Daily prices of the two listed fishing companies (used by the liquidity diagnosis). |
| `DATA/README.md` | Description and sources of the data files. |
| `README.md` | This file. |

Open in Google Colab:
- Main notebook: [FIS_ML_Unified_Notebook_v7.ipynb](https://colab.research.google.com/github/mbaernestocr-tech/FIS-ML-Anchovy-Production-Classifier-Peru/blob/main/FIS_ML_Unified_Notebook_v7.ipynb)
- Figure notebook: [FIS_ML_Figures_Notebook_v2.ipynb](https://colab.research.google.com/github/mbaernestocr-tech/FIS-ML-Anchovy-Production-Classifier-Peru/blob/main/FIS_ML_Figures_Notebook_v2.ipynb)

Archived versions of this repository are available at Zenodo: https://doi.org/10.5281/zenodo.23978816

Note on language: the comments and explanatory text of the notebooks are written in English. Some messages printed by the code and some column labels are in Spanish (the working language of the authors).

## Reported run

| Item | Value |
|---|---|
| Main notebook version | v7.0 (adds the 1D-CNN, LSTM and GRU benchmark) |
| Run identifier | `20260903_19h_54m_23s` |
| Random seed | 2025 |
| Platform | Google Colab, Linux-6.6.122+-x86_64-with-glibc2.35 |
| Python | 3.13.15 |
| Library versions | NumPy 2.1.3, pandas 2.2.3, matplotlib 3.10.0, scikit-learn 1.6.1, SciPy 1.16.3, XGBoost 3.4.1, TensorFlow/Keras 2.20.0 (joblib, xarray, netCDF4 and requests unpinned) |

Every numerical value quoted in the manuscript is produced by this single run. The main notebook writes all its outputs with the run identifier in the file name (so that runs never overwrite each other), including the results file `results_v6_<run identifier>.json`. For the reported run, this file is `results_v6_20260903_19h_54m_23s.json`. It holds, for every estimator, the metrics of each evaluated split together with the statistical tests, the sensitivity analysis, the computational cost, the liquidity diagnosis and the backtest. The file is provided in `DATA/` of this repository and with the journal submission. Its internal field `notebook_version` reads "v6.0" because that constant was not updated when the deep-learning benchmark was added in v7.0; the run identifier is the reference.

## What the main notebook does

1. **Data.** Daily anchovy landings (Peruvian Ministry of Production) and monthly HadCRUT5 temperature anomalies of four 5° x 5° cells over the Peruvian coast, interpolated to daily resolution. Study window: 2023-01-01 to 2024-03-14 (439 calendar days; 436 modelling rows: 72 High and 364 Low).
2. **Features and target.** Twelve predictors (six base and six derived; data dictionary in Table 1 of the manuscript) and a binary target (High/Low landings) defined by a threshold estimated on training rows only.
3. **Validation.** Forward-chaining temporal validation: 5 blocks of 72 days with a 3-day gap; the first block is used only for training, so 4 splits are evaluated (288 pooled test observations). Standardization and the class threshold are estimated inside each split from training rows only.
4. **Estimators.** Eight classifiers, three naive baselines (majority class, seasonal naive, stratified random) and three deep-learning architectures (1D-CNN, LSTM, GRU).
5. **Evaluation.** Balanced accuracy, MCC and ROC-AUC (plus accuracy, precision, recall and F1-score); McNemar tests with Holm correction; bootstrap confidence intervals; Wilcoxon and paired t-tests across splits; permutation importance and directed-perturbation sensitivity analysis; computational cost.
6. **Investment module (conceptual design).** Pre-registered liquidity gate (60% of sessions with traded volume and 40% with a price change) applied to EXALMC1 and AUSTRAC1, and an exploratory backtest reported as a diagnostic.

## Data sources

- **Fish landings:** Ministerio de la Produccion del Peru, open data portal (`DS_DGSF_DESCARGAS_anonimizado.csv`, https://www.datosabiertos.gob.pe/sites/default/files/DS_DGSF_DESCARGAS_anonimizado.csv). License: Open Data Commons Attribution License; publisher: Ministerio de la Produccion (PRODUCE). A copy is provided in `DATA/`; the notebook uses a local copy if present and downloads the file otherwise.
- **Temperature anomaly:** HadCRUT 5.0.2.0 ensemble mean, Climatic Research Unit, University of East Anglia (https://crudata.uea.ac.uk/cru/data/temperature/HadCRUT.5.0.2.0.anomalies.ensemble_mean.nc). The notebook downloads it.
- **Stock prices:** Bolsa de Valores de Lima, daily OHLC for EXALMC1 and AUSTRAC1 (`EXALMAR_EXALMC1_LM_OHLC_diario.csv` and `AUSTRAL_AUSTRAC1_LM_OHLC_diario.csv`). They are provided in `DATA/`. They are used only by the liquidity diagnosis and the exploratory backtest; if they are absent, the notebook skips that section and runs the rest.

## How to reproduce

1. Open the main notebook in Google Colab (links above).
2. The notebook mounts Google Drive and uses `/content/drive/My Drive/Colab Notebooks/2024-2.SVM.Pesca/` as working directory, with the subfolders `DATA`, `models` and `IMAGES` (created automatically). To use another location, change `BASE_DIR` in section 1.2.
3. Optional: copy the files of `DATA/` of this repository to the `DATA` folder of the working directory (the PRODUCE file is downloaded if absent; the two stock-price files are only needed by the liquidity diagnosis).
4. Run all cells from top to bottom. The random seed is fixed (2025). Small differences in the last decimals can appear across library versions; the versions of the reported run are listed above.
5. Compare the new results file with `DATA/results_v6_20260903_19h_54m_23s.json`.
6. To redraw the figures, run the figure notebook after the main notebook. It reads the CSV and JSON files of the run identifier set in its section 1 (`RUN_ID`, frozen at `20260903_19h_54m_23s`; change it to use the files of a new run) and writes the figures with the suffix `_EN` to `IMAGES`. The ROC curves are not regenerated because the per-observation probabilities are not exported; the ROC-AUC values are in Table 3 of the manuscript.

## Funding and contact

Universidad Nacional Mayor de San Marcos, Lima, Peru. Funded by CONCYTEC/PROCIENCIA.
Contact: Ernesto David Cancho-Rodriguez (ernesto.cancho@unmsm.edu.pe).
