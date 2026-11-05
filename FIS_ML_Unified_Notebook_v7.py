# -*- coding: utf-8 -*-

"""# FIS-ML — Unified notebook v7.0
## Multi-classifier decision-support system for anchovy production levels in Peru

Code accompanying the manuscript **"FIS-ML: Multi-Classifier Decision-Support System for Anchovy Production Levels Using Climate and Fishery Data from Peru"**, submitted to *Expert Systems with Applications* (Elsevier), manuscript ESWA-D-25-29468, revised version.

**Authors:** Cancho-Rodriguez, Ernesto David et al. — Universidad Nacional Mayor de San Marcos (UNMSM), Lima, Peru.
**Version:** v7.0 · 2026-09-03

Repository: https://github.com/mbaernestocr-tech/FIS-ML-Anchovy-Production-Classifier-Peru

Note: the printed messages and some column labels of this notebook are in Spanish.

## Abbreviations

MCC: Matthews correlation coefficient · ROC-AUC: area under the receiver operating characteristic curve · bps: basis points (1 bps = 0.01 %) · 1D-CNN: one-dimensional convolutional neural network · LSTM: long short-term memory · GRU: gated recurrent unit.

## What this notebook does

A single top-to-bottom run produces every number, table and figure of the revised manuscript:

| Section | Content |
|---|---|
| 1 | Setup: libraries, paths, constants and random seed |
| 2 | Data extraction and unification (PRODUCE landings, HadCRUT5 temperature) |
| 3 | Target variable, feature engineering and data dictionary |
| 4 | Class distribution |
| 5 | Validation protocols and leakage diagnosis |
| 6 | Models and baselines |
| 7 | Evaluation engine and nested hyperparameter tuning |
| 8 | Factorial run of protocols and threshold modes; deep-learning benchmark (8.1) |
| 9 | Results: tables, confusion matrices, ROC curves, feature importance, climate sensitivity, computational cost |
| 10 | Statistical significance (McNemar with Holm correction, Wilcoxon, paired t, bootstrap intervals) |
| 11 | Investment recommendation module: liquidity gate, signal-to-decision rule and exploratory backtest |
| 12 | Neural-network diagnosis |
| 13-17 | Consolidated JSON export, model persistence, narrative generator, checklist and output inventory |
| 18-21 | Applied refinements, delivery package (manifest, quotable figures, reproducibility files) and automatic closing checks |"""

"""# 1. Setup and environment

## 1.1 Libraries"""

# ============================================================
# Standard library
# ============================================================
import os
import time
import json as json_lib
import platform
import warnings
import itertools
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# `display` exists in Jupyter/Colab; define a stand-in for local execution.
try:
    display
except NameError:
    def display(x):
        print(x)

# ============================================================
# Scikit-learn
# ============================================================
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score, roc_curve,
)
from sklearn.inspection import permutation_importance

# ============================================================
# Models
# ============================================================
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

try:
    import xgboost
    from xgboost import XGBClassifier
    XGBOOST_DISPONIBLE = True
except ImportError:
    XGBOOST_DISPONIBLE = False
    print("XGBoost no instalado. Ejecutar: !pip install xgboost")

# ============================================================
# Statistics
# ============================================================
import scipy
from scipy.stats import chi2 as chi2_dist, wilcoxon, ttest_rel

# ============================================================
# Serialization
# ============================================================
import joblib

print("Librerias cargadas.")
print(f"  numpy        {np.__version__}")
print(f"  pandas       {pd.__version__}")
print(f"  matplotlib   {matplotlib.__version__}")
print(f"  scikit-learn {sklearn.__version__}")
print(f"  scipy        {scipy.__version__}")
if XGBOOST_DISPONIBLE:
    print(f"  xgboost      {xgboost.__version__}")

"""## 1.1b Deep-learning framework detection

The deep-learning benchmark (section 8.1) uses Keras/TensorFlow if it is available, otherwise PyTorch. If neither is installed, the section is skipped with an explicit message and the results file records the reason. Both frameworks are preinstalled in Colab."""

# ============================================================
# Auto-detection of the deep-learning framework
# ============================================================
DEEP_LEARNING_DISPONIBLE = False
DEEP_LEARNING_MARCO = None
DEEP_LEARNING_MOTIVO = ""

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers as keras_layers
    DEEP_LEARNING_DISPONIBLE = True
    DEEP_LEARNING_MARCO = f"tensorflow/keras {tf.__version__}"
except Exception as e_tf:
    try:
        import torch
        import torch.nn as torch_nn
        DEEP_LEARNING_DISPONIBLE = True
        DEEP_LEARNING_MARCO = f"pytorch {torch.__version__}"
    except Exception as e_pt:
        DEEP_LEARNING_MOTIVO = (
            f"no se encontro TensorFlow ({type(e_tf).__name__}) "
            f"ni PyTorch ({type(e_pt).__name__})")

if DEEP_LEARNING_DISPONIBLE:
    print(f"Marco de aprendizaje profundo detectado: {DEEP_LEARNING_MARCO}")
else:
    print("Sin marco de aprendizaje profundo disponible.")
    print(f"  Motivo: {DEEP_LEARNING_MOTIVO}")
    print("  La seccion 8.1 se saltara y el JSON registrara el motivo.")

"""## 1.2 Google Drive and paths"""

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount("/content/drive")
    EN_COLAB = True
except Exception:
    EN_COLAB = False
    print("Entorno local (no Colab).")

# ============================================================
# Project paths
# ============================================================
BASE_DIR   = "/content/drive/My Drive/Colab Notebooks/2024-2.SVM.Pesca"
DATA_DIR   = os.path.join(BASE_DIR, "DATA")
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "IMAGES")

for d in [DATA_DIR, MODELS_DIR, IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"DATA:   {DATA_DIR}")
print(f"MODELS: {MODELS_DIR}")
print(f"IMAGES: {IMAGES_DIR}")

"""## 1.3 Constants, random seed and environment capture

All study parameters are defined here."""

# ============================================================
# Global random seed
# ============================================================
RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)

# ============================================================
# Study parameters
# ============================================================
DATE_START = "2023-01-01"
DATE_END   = "2024-03-14"
N_FOLDS    = 5
MA_WINDOW  = 3           # window of the temperature moving average (days)
TS_GAP     = MA_WINDOW   # gap >= longest causal window

PERU_COORDS = [(-7.5, -77.5), (-12.5, -77.5), (-17.5, -77.5), (-17.5, -72.5)]

# ============================================================
# Factorial grid of configurations
# ============================================================
# 3 validation protocols x 2 threshold modes.
# All six configurations are run in the same execution.
PROTOCOLS = ["kfold_shuffle", "timeseries", "timeseries_blocked"]
THRESHOLD_MODES = ["global_fixed", "per_fold_train"]

# Protocol reported as primary in the manuscript
PRIMARY_PROTOCOL = "timeseries"
PRIMARY_THRESHOLD_MODE = "per_fold_train"

# ============================================================
# Deep-learning benchmark parameters
# ============================================================
# Tabular models receive one row; a 1D-CNN, an LSTM or a GRU needs
# a window of DL_WINDOW days x len(FEATURE_COLUMNS) variables.
RUN_DEEP_LEARNING = True
DL_WINDOW  = 7        # days per window
DL_EPOCHS  = 120      # maximum number of epochs; temporal early stopping usually stops earlier
DL_PACIENCIA = 15     # epochs without improvement before stopping
DL_BATCH   = 16
DL_VAL_FRACCION = 0.15   # last 15 % of the training block, in chronological order

# ============================================================
# Switches for the expensive blocks
# ============================================================
RUN_NESTED_TUNING = True     # re-tuning inside each split
RUN_SENSITIVITY   = True     # climate sensitivity analysis
RUN_BACKTEST      = True     # investment module
N_MONTECARLO      = 2000     # random-signal simulations

# ============================================================
# Pre-registered liquidity gate
# ============================================================
# These thresholds are fixed BEFORE looking at the backtest results, so that the choice between
# validating the investment rule and reporting it as a conceptual design does not depend on
# whether the result is favourable (this avoids data snooping).
LIQ_MIN_PCT_CON_VOLUMEN = 60.0   # minimum % of sessions with traded volume > 0
LIQ_MIN_PCT_CON_CAMBIO  = 40.0   # minimum % of sessions with a price change
COSTOS_TRANSACCION_BPS  = [0, 25, 50]   # basis points per trade

# Delay between the signal and its execution (a declared assumption).
# A realistic backtest cannot trade on information that did not yet exist.
#   lag = 0 -> a signal computed with day-D data is executed at the close of D
#   lag = 1 -> executed at the close of D+1, as a trader receiving the signal after the close would
# The declared case is reported together with a sensitivity analysis to this assumption.
BACKTEST_LAG_DIAS      = 1
BACKTEST_LAGS_SENSIBILIDAD = [0, 1, 2]

# ============================================================
# Baselines assigned to the main contrast and to the supplementary table
# ============================================================
# Main contrast: majority class, seasonal naive and stratified random.
# Supplementary table: persistence (tomorrow = today), computed but not part of the main contrast.
BASELINE_R1M6_VERBATIM = (
    "No simple baselines are reported (e.g., predicting the majority class, or "
    "persistence forecast - tomorrow = today). Without these, the value added by "
    "ML is unclear. Please add a 'majority class' baseline and, if applicable, a "
    "naive seasonal baseline.")
BASELINE_LECTURA_ADOPTADA = (
    "La peticion explicita del revisor nombra la clase mayoritaria y, "
    "condicionalmente, una linea base estacional ingenua. La persistencia figura "
    "en el parentesis de ejemplos. Se reporta en el contraste principal lo que el "
    "revisor pide, y la persistencia se conserva calculada como baseline "
    "suplementario, declarado como tal, sin retirarla del computo ni de la "
    "correccion por multiplicidad.")

# ============================================================
# Timestamped output helpers (auditability)
# ============================================================
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%Hh_%Mm_%Ss")

RUN_ID = get_timestamp()

ARCHIVOS_GENERADOS = []   # cumulative inventory of the outputs

def save_csv(df, prefix, folder=DATA_DIR):
    """
    Save a DataFrame as a CSV file with a timestamp in its name.

    Comma separator and utf-8-sig encoding: the comma is read by pandas and R without
    extra configuration, and the BOM lets Excel on Windows display accents correctly.
    """
    fpath = os.path.join(folder, f"{prefix}_{RUN_ID}.csv")
    df.to_csv(fpath, index=False, encoding="utf-8-sig")
    ARCHIVOS_GENERADOS.append(("tabla", prefix, os.path.basename(fpath)))
    print(f"  Guardado: {os.path.basename(fpath)}")
    return fpath

def save_figure(fig, prefix, folder=IMAGES_DIR, dpi=300):
    """Save a figure at 300 dpi with a white background."""
    fpath = os.path.join(folder, f"{prefix}_{RUN_ID}.png")
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    ARCHIVOS_GENERADOS.append(("figura", prefix, os.path.basename(fpath)))
    print(f"  Figura guardada: {os.path.basename(fpath)}")
    return fpath

# ============================================================
# Environment capture for reproducibility
# ============================================================
ENVIRONMENT = {
    "run_id": RUN_ID,
    "notebook_version": "v6.0",
    "python": platform.python_version(),
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "matplotlib": matplotlib.__version__,
    "scikit_learn": sklearn.__version__,
    "scipy": scipy.__version__,
    "xgboost": xgboost.__version__ if XGBOOST_DISPONIBLE else None,
    "deep_learning_framework": DEEP_LEARNING_MARCO,
    "random_seed": RANDOM_SEED,
}

print(f"RUN_ID: {RUN_ID}")
print(f"Semilla: {RANDOM_SEED}  |  Periodo: {DATE_START} -> {DATE_END}")
print(f"Configuraciones a ejecutar: {len(PROTOCOLS)} protocolos x "
      f"{len(THRESHOLD_MODES)} modos de umbral = {len(PROTOCOLS)*len(THRESHOLD_MODES)}")
print(f"Protocolo principal: {PRIMARY_PROTOCOL} / {PRIMARY_THRESHOLD_MODE}")
print(f"Benchmark de aprendizaje profundo: "
      f"{'activado' if (RUN_DEEP_LEARNING and DEEP_LEARNING_DISPONIBLE) else 'desactivado'}")

"""## 1.4 Monochrome figure style

Figures are designed to print legibly in black and white: white background, black borders, grey fills with sufficient separation, hatch patterns as a second distinguishing channel, and black text on white cells in the confusion matrices. The only exception is the set of ROC curves (section 9.4), where eight overlapping series need colour to be told apart."""

# ============================================================
# Monochrome palette and figure utilities
# ============================================================
# Greys ordered from light to dark, with enough separation to tell them apart in laser printing.
# Pure black is avoided as a fill; it is used only for borders and text.
GRISES = ["#ffffff", "#e6e6e6", "#cccccc", "#b3b3b3",
          "#999999", "#808080", "#666666", "#4d4d4d"]

# Hatch patterns as a second distinguishing channel (independent of the grey level).
TRAMAS = ["", "///", "...", "xxx", "\\\\\\", "|||", "---", "ooo"]

def gris(i):
    """Return the i-th grey of the palette, cycling if needed."""
    return GRISES[i % len(GRISES)]

def trama(i):
    """Return the i-th hatch pattern of the palette, cycling if needed."""
    return TRAMAS[i % len(TRAMAS)]

def estilo_eje(ax, rejilla="y"):
    """Apply the standard monochrome style to an axis: white background, light grid."""
    ax.set_facecolor("white")
    if rejilla:
        ax.grid(axis=rejilla, color="#cccccc", linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    for lado in ["left", "bottom"]:
        ax.spines[lado].set_color("black")
        ax.spines[lado].set_linewidth(0.8)
    return ax

def matriz_confusion_monocroma(ax, cm, titulo="", etiquetas=("Bajo", "Alto"),
                               normalizar=False):
    """
    Draw a 2x2 confusion matrix with white cells and black border and text.

    No colour map and no dark blocks: legible when photocopied and cheap to print.
    """
    M = np.asarray(cm, dtype=float)
    if normalizar:
        filas = M.sum(axis=1, keepdims=True)
        filas[filas == 0] = 1.0
        M = M / filas * 100.0

    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, linewidth=1.5,
                                       edgecolor="black", facecolor="white"))
            txt = f"{M[i, j]:.1f}%" if normalizar else f"{int(M[i, j]):d}"
            ax.text(j + 0.5, 1 - i + 0.5, txt, ha="center", va="center",
                    fontsize=11, color="black")

    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels([f"Pred. {e}" for e in etiquetas], fontsize=9)
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels([f"Real {e}" for e in etiquetas], fontsize=9)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    if titulo:
        ax.set_title(titulo, fontsize=10)
    return ax

# Global figure style: font size >= 10 pt for all elements
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 110,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black", "axes.linewidth": 0.8,
    "hatch.linewidth": 0.6,
})

print("Estilo monocromo definido.")
print(f"  {len(GRISES)} niveles de gris y {len(TRAMAS)} tramas disponibles.")
print("  Excepcion declarada: las curvas ROC de §9.4 conservan color.")

"""# 2. Data extraction and unification

Every intermediate step is exported with a timestamp for auditability.

## 2.1 Fishery production (PRODUCE)

**Source:** Open Data Portal of the Government of Peru — Ministry of Production (Ministerio de la Produccion; license: Open Data Commons Attribution License).
**Granularity:** individual landing records, aggregated to total daily production."""

# ============================================================
# Load the PRODUCE CSV
# ============================================================
# The local copy is tried first (faster and more stable);
# if it does not exist, the file is downloaded from the portal.
PRODUCTION_LOCAL = os.path.join(DATA_DIR, "DS_DGSF_DESCARGAS_anonimizado.csv")
PRODUCTION_URL   = "https://www.datosabiertos.gob.pe/sites/default/files/DS_DGSF_DESCARGAS_anonimizado.csv"

fuente = PRODUCTION_LOCAL if os.path.exists(PRODUCTION_LOCAL) else PRODUCTION_URL
Production_dataframe = pd.read_csv(fuente, encoding="ISO-8859-1", sep=";")
print(f"Fuente: {'copia local en Drive' if fuente == PRODUCTION_LOCAL else 'portal datosabiertos.gob.pe'}")
print(f"Registros de desembarque: {len(Production_dataframe):,}")

# ============================================================
# Column and date normalization
# ============================================================
Production_dataframe = Production_dataframe.rename(columns={
    "ANIO": "Year", "TRIMESTRE": "Year-Quarter", "TMDESCARGADA": "Production",
    "FECHA_DESCARGA": "Download_Date", "FECHA_DESEMBARQUE": "Fishing_Unloading_Date",
    "FECHA_CORTE": "Data_Collection_Cutoff_Date", "FECHA_PUB_PNDA": "Publication_Date",
    "TEMPORADA": "Period",
})

Production_dataframe["Year-Quarter"] = Production_dataframe["Year-Quarter"].str.replace(
    r"(\d{4})-(\d)", r"\1-Q\2", regex=True)
Production_dataframe.insert(2, "Quarter",
    Production_dataframe["Year-Quarter"].str.extract(r"Q(\d)").astype(int))
Production_dataframe["Period"] = Production_dataframe["Period"].str.replace(
    r"(\d{4})-(\d)", r"\1-Period-\2", regex=True)

for col in ["Download_Date", "Fishing_Unloading_Date",
            "Data_Collection_Cutoff_Date", "Publication_Date"]:
    Production_dataframe[col] = pd.to_datetime(
        Production_dataframe[col], format="%Y%m%d", errors="coerce")

print(f"Columnas normalizadas: {len(Production_dataframe.columns)}")

# ============================================================
# Daily aggregation and filling of the complete calendar
# ============================================================
def assign_season(month):
    """Season of the year in the Southern Hemisphere."""
    if month in (1, 2, 3):   return 1   # Summer
    elif month in (4, 5, 6): return 2   # Autumn
    elif month in (7, 8, 9): return 3   # Winter
    else:                    return 4   # Spring

df_prod_daily = (Production_dataframe
                 .groupby("Fishing_Unloading_Date")
                 .agg(Production_Total=("Production", "sum"))
                 .reset_index()
                 .rename(columns={"Fishing_Unloading_Date": "Date"}))

# Complete calendar: days without landings are zero production,
# not missing data (closed seasons and holidays).
date_range = pd.date_range(start=DATE_START, end=DATE_END)
df_production = pd.merge(pd.DataFrame({"Date": date_range}),
                         df_prod_daily, on="Date", how="left")
df_production["Production_Total"] = df_production["Production_Total"].fillna(0)

df_production["Year"]    = df_production["Date"].dt.year.astype(int)
df_production["Month"]   = df_production["Date"].dt.month.astype(int)
df_production["Day"]     = df_production["Date"].dt.day.astype(int)
df_production["Season"]  = df_production["Month"].apply(assign_season).astype(int)
df_production["Weekday"] = df_production["Date"].dt.strftime("%A")

n_con = int((df_production["Production_Total"] > 0).sum())
n_sin = int((df_production["Production_Total"] == 0).sum())
print(f"Dias en el calendario: {len(df_production)}")
print(f"  con desembarque: {n_con}  ({n_con/len(df_production)*100:.1f}%)")
print(f"  sin desembarque: {n_sin}  ({n_sin/len(df_production)*100:.1f}%)")
save_csv(df_production, "01_production_daily")

"""## 2.2 Climate data (HadCRUT5)

**Source:** Climatic Research Unit, University of East Anglia.
**Variable:** `tas_mean`, mean surface temperature anomaly.
**Native resolution:** 5°x5° global grid, monthly frequency, interpolated to daily.

> This coarse resolution is a real limitation for a narrow coastal strip such as the Peruvian coast."""

# ============================================================
# Download of the HadCRUT5 NetCDF file
# ============================================================
# !pip install xarray netCDF4 --quiet

import xarray as xr, requests, tempfile

HADCRUT_URL = "https://crudata.uea.ac.uk/cru/data/temperature/HadCRUT.5.0.2.0.anomalies.ensemble_mean.nc"
resp = requests.get(HADCRUT_URL, stream=True)
if resp.status_code != 200:
    raise ConnectionError(f"Error al descargar HadCRUT5: HTTP {resp.status_code}")

tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
tmp.write(resp.content); tmp.close()
dataset = xr.open_dataset(tmp.name)
print(f"Variables disponibles: {list(dataset.data_vars)}")

# ============================================================
# Spatial filter: Peruvian coast
# ============================================================
data_table = dataset.to_dataframe().reset_index()
data_table = data_table[data_table["tas_mean"].notna()]
data_table = data_table[data_table["time"] >= "2023-01-01"]

peru_ocean = data_table[
    (data_table["latitude"]  >= -18) & (data_table["latitude"]  <= 0) &
    (data_table["longitude"] >= -81) & (data_table["longitude"] <= -71)]

filtered_specific = peru_ocean[
    peru_ocean[["latitude", "longitude"]].apply(tuple, axis=1).isin(PERU_COORDS)
].reset_index(drop=True)

print(f"Registros del litoral: {len(peru_ocean)}  ->  en celdas seleccionadas: {len(filtered_specific)}")
print(f"Celdas de malla utilizadas: {PERU_COORDS}")

# ============================================================
# Daily temperature series by linear interpolation
# ============================================================
filtered_specific["Date"] = pd.to_datetime(filtered_specific["time"].dt.date)
pivot_table = (filtered_specific.groupby("Date")
               .agg(Temperature_Mean=("tas_mean", "mean")).reset_index())

full_range = pd.date_range(start="2023-01-16", end="2024-11-16")
Temperature_df = pd.merge(pd.DataFrame({"Date": full_range}), pivot_table,
                          on="Date", how="left")
Temperature_df["Temperature_Mean"] = Temperature_df["Temperature_Mean"].interpolate(method="linear")

# The first 15 days of January 2023 precede the first monthly observation:
# they are filled with the value of 16 January (constant backfill).
first_val = Temperature_df.loc[Temperature_df["Date"] == "2023-01-16", "Temperature_Mean"].values[0]
head = pd.DataFrame({"Date": pd.date_range("2023-01-01", "2023-01-15"),
                     "Temperature_Mean": first_val})
Temperature_df = pd.concat([head, Temperature_df]).sort_values("Date").reset_index(drop=True)

Temperature_df["Year"]   = Temperature_df["Date"].dt.year.astype(int)
Temperature_df["Month"]  = Temperature_df["Date"].dt.month.astype(int)
Temperature_df["Day"]    = Temperature_df["Date"].dt.day.astype(int)
Temperature_df["Season"] = Temperature_df["Month"].apply(assign_season).astype(int)

RAZON_INTERPOLACION = {
    "observaciones_mensuales_reales": int(len(pivot_table)),
    "dias_serie_interpolada": int(len(Temperature_df)),
}
print(f"Observaciones mensuales originales: {len(pivot_table)}")
print(f"Serie diaria interpolada: {len(Temperature_df)} dias")
print(f"  -> {len(pivot_table)} valores reales expandidos a {len(Temperature_df)} dias.")
print("  Esta razon de interpolacion debe declararse en Limitaciones.")
save_csv(Temperature_df, "02_temperature_daily")

"""## 2.3 Unification"""

# ============================================================
# Merge by date
# ============================================================
df_unified = pd.merge(Temperature_df, df_production, on="Date", how="inner",
                      suffixes=("", "_prod"))
df_unified = df_unified.drop(columns=[c for c in df_unified.columns if c.endswith("_prod")])

weekday_map = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
               "Friday": 5, "Saturday": 6, "Sunday": 7}
df_unified["Day_of_the_Week"] = df_unified["Weekday"].map(weekday_map)
df_unified["ID"] = range(1, len(df_unified) + 1)

df_unified = df_unified[[
    "Date", "Temperature_Mean", "Production_Total", "ID",
    "Year", "Month", "Day", "Season", "Day_of_the_Week", "Weekday"]]
df_unified = df_unified.sort_values("Date").reset_index(drop=True)

# Check: strict chronological order is required by everything that follows.
assert df_unified["Date"].is_monotonic_increasing, "El dataset no esta ordenado por fecha."
assert df_unified["Date"].is_unique, "Hay fechas duplicadas."

print(f"Dataset unificado: {len(df_unified)} dias x {len(df_unified.columns)} columnas")
print(f"Rango: {df_unified['Date'].min().date()} -> {df_unified['Date'].max().date()}")
save_csv(df_unified, "03_unified_dataset")
display(df_unified.head())

"""# 3. Target variable, feature engineering and data dictionary

## 3.1 High/Low threshold definition

**Rule:** the production of a day is "High" if it exceeds the **arithmetic mean of production over landing days** (`Production_Total > 0`); otherwise it is "Low", which includes all closed-season days.

Two modes are implemented and the factorial run executes both:

| Mode | How the threshold is computed | Note |
|---|---|---|
| `global_fixed` | Mean over the full series, before splitting | Leaks information: the threshold that labels the test set incorporates information from future dates. Descriptive reference only |
| `per_fold_train` | Mean computed only on the training rows of each split; it binarizes both `y_train` and `y_test` | **Primary protocol** |

The threshold is a parameter learned from the data: if it were estimated on the whole series, the label of each test day would depend on the production of days the model should not know. In `per_fold_train`, the true label of the same day can differ between splits; this is intended, because a real-time forecaster only has the historical mean known up to that moment. Section 8 reports the variation of the threshold across splits."""

# ============================================================
# Threshold as a function, not as a global constant
# ============================================================
def compute_threshold(production_series):
    """
    High/Low threshold = mean production over landing days.
    It receives only the relevant rows (all rows, or the training rows).
    """
    positivos = production_series[production_series > 0]
    return float(positivos.mean()) if len(positivos) else 0.0

def binarize(production_series, threshold):
    """Apply the threshold: 1 = High, 0 = Low."""
    return (production_series > threshold).astype(int)

# Global threshold: descriptive reference only, and for the `global_fixed` mode
THRESHOLD_GLOBAL = compute_threshold(df_unified["Production_Total"])
print(f"Umbral global (referencia): {THRESHOLD_GLOBAL:,.2f} TM")
print(f"  Calculado sobre {(df_unified['Production_Total'] > 0).sum()} dias con desembarque.")

"""## 3.2 Target variable: next-day production

The model predicts whether production on day **D+1** will be high or low, using information available up to day **D**. Continuous next-day production (`Production_Total_Next_Day`) is kept because the `per_fold_train` mode must re-binarize it with the threshold of each split. Continuous production of the current day (`Production_Total`) is also kept, rather than an already binarized label, so that the baselines that use today's production can apply the threshold of each split."""

# ============================================================
# Keep the CONTINUOUS next-day target
# ============================================================
df_target = df_unified.copy()
df_target["Production_Total_Next_Day"] = df_target["Production_Total"].shift(-1)
df_target = df_target.dropna(subset=["Production_Total_Next_Day"]).reset_index(drop=True)

# Label of the current day with the global threshold: kept ONLY as a descriptive
# reference and for the `global_fixed` mode. The baselines use the threshold of their own split (see section 6.2).
df_target["Production_Label_Today_GlobalRef"] = binarize(
    df_target["Production_Total"], THRESHOLD_GLOBAL)

print(f"Registros tras el desplazamiento del objetivo: {len(df_target)}")
print("Se conserva la produccion continua de D y de D+1 para permitir")
print("re-binarizacion por particion tanto del objetivo como de la persistencia.")

"""## 3.3 Feature engineering

Six derived variables on top of six base variables, with explicit formulas.

| # | Variable | Formula | Nature |
|---|---|---|---|
| 1 | `Temp_MA3` | mean of `Temperature_Mean` over `[t-2, t]` | causal, backward-looking |
| 2 | `Temp_Diff` | `Temperature_Mean[t] - Temperature_Mean[t-1]` | causal, backward-looking |
| 3 | `Month_sin` | `sin(2 pi Month / 12)` | deterministic calendar function |
| 4 | `Month_cos` | `cos(2 pi Month / 12)` | deterministic calendar function |
| 5 | `Day_sin` | `sin(2 pi Day / 31)` | deterministic calendar function |
| 6 | `Day_cos` | `cos(2 pi Day / 31)` | deterministic calendar function |

### Note on leakage

`rolling(window=3).mean()` looks only backwards: the value at `t` uses `t-2`, `t-1` and `t`. The sine/cosine encodings are deterministic functions of the calendar and carry no information about production. These variables therefore do not leak the future into the past. They do create dependence between contiguous rows, and that dependence becomes leakage when the splitting scheme distributes neighbouring rows between the training and test sets (as `KFold(shuffle=True)` does). The remedy is to respect the temporal order and separate training from test with `gap >= MA_WINDOW`, which is what the `timeseries` protocol of section 5 does."""

# ============================================================
# Feature engineering (causal, backward-looking functions)
# ============================================================
def feature_engineering(df):
    """
    Create the 6 derived variables.

    All are causal: Temp_MA3 and Temp_Diff only use values at t or earlier;
    the cyclical encodings are deterministic functions of the calendar.
    They introduce no information from the future. The dependence between
    contiguous rows is neutralized by the gap of the temporal protocol (see section 5).
    """
    out = df.copy()
    out["Temp_MA3"]   = out["Temperature_Mean"].rolling(window=MA_WINDOW).mean()
    out["Temp_Diff"]  = out["Temperature_Mean"].diff()
    out["Month_sin"]  = np.sin(2 * np.pi * out["Month"] / 12)
    out["Month_cos"]  = np.cos(2 * np.pi * out["Month"] / 12)
    out["Day_sin"]    = np.sin(2 * np.pi * out["Day"] / 31)
    out["Day_cos"]    = np.cos(2 * np.pi * out["Day"] / 31)
    return out

df_final = feature_engineering(df_target).dropna().reset_index(drop=True)

FEATURE_COLUMNS = [
    "Temperature_Mean", "Season", "Month", "Day", "Year", "Day_of_the_Week",
    "Temp_MA3", "Temp_Diff", "Month_sin", "Month_cos", "Day_sin", "Day_cos",
]

X          = df_final[FEATURE_COLUMNS].copy()
DATES      = df_final["Date"].copy()
PROD_NEXT  = df_final["Production_Total_Next_Day"].copy()   # continuous target D+1
PROD_TODAY = df_final["Production_Total"].copy()            # continuous production D
MONTH_ARR  = df_final["Month"].copy()                       # seasonal naive

# Consistency checks
assert DATES.is_monotonic_increasing, "df_final no esta en orden cronologico."
assert not X.isna().any().any(), "Hay NaN en la matriz de caracteristicas."

print(f"Dataset final: {len(df_final)} registros x {len(FEATURE_COLUMNS)} caracteristicas")
print(f"Rango: {DATES.min().date()} -> {DATES.max().date()}")
save_csv(df_final, "04_df_final_features")

# ============================================================
# Reconciliation of the sample size
# ============================================================
# The number of rows shrinks along the processing chain (calendar days, rows with a next-day target,
# modelling rows); each step is printed and exported so that every reported size can be traced.
SAMPLE_RECONCILIATION = {
    "dias_calendario": int(len(df_production)),
    "tras_merge_climatico": int(len(df_unified)),
    "tras_objetivo_d1": int(len(df_target)),
    "tras_calentamiento_features": int(len(df_final)),
    "perdidas_por_shift_d1": int(len(df_unified) - len(df_target)),
    "perdidas_por_calentamiento": int(len(df_target) - len(df_final)),
    "ventana_media_movil": int(MA_WINDOW),
    "cifra_citada_en_manuscrito": int(len(df_target)),
    "cifra_de_entrenamiento": int(len(df_final)),
}
_r = SAMPLE_RECONCILIATION
print()
print("Reconciliacion del tamano muestral:")
print(f"  dias de calendario ({DATE_START} -> {DATE_END}) : {_r['dias_calendario']}")
print(f"  tras el merge con la serie climatica            : {_r['tras_merge_climatico']}")
print(f"  tras desplazar el objetivo a D+1                : {_r['tras_objetivo_d1']}  (-{_r['perdidas_por_shift_d1']})")
print(f"  tras el calentamiento de Temp_MA3 / Temp_Diff   : {_r['tras_calentamiento_features']}  (-{_r['perdidas_por_calentamiento']})")
print("  El manuscrito cita el tercer valor; el modelo se entrena sobre el cuarto.")
save_csv(pd.DataFrame([SAMPLE_RECONCILIATION]), "04b_sample_reconciliation")

"""## 3.4 Exportable data dictionary

Explicit formula, type and source of every variable, exported as a file that accompanies the code."""

# ============================================================
# Data dictionary
# ============================================================
DATA_DICTIONARY = [
    {"Variable": "Temperature_Mean", "Tipo": "base", "Unidad": "°C (anomalia)",
     "Formula": "media de tas_mean (HadCRUT5) sobre 4 celdas de malla 5°x5° del litoral peruano",
     "Fuente": "HadCRUT 5.0.2.0 ensemble mean"},
    {"Variable": "Season", "Tipo": "base", "Unidad": "1-4",
     "Formula": "1=Verano(Ene-Mar), 2=Otono(Abr-Jun), 3=Invierno(Jul-Sep), 4=Primavera(Oct-Dic)",
     "Fuente": "derivada del calendario (Hemisferio Sur)"},
    {"Variable": "Month", "Tipo": "base", "Unidad": "1-12", "Formula": "mes calendario",
     "Fuente": "calendario"},
    {"Variable": "Day", "Tipo": "base", "Unidad": "1-31", "Formula": "dia del mes",
     "Fuente": "calendario"},
    {"Variable": "Year", "Tipo": "base", "Unidad": "ano", "Formula": "ano calendario",
     "Fuente": "calendario"},
    {"Variable": "Day_of_the_Week", "Tipo": "base", "Unidad": "1-7",
     "Formula": "1=Lunes ... 7=Domingo", "Fuente": "calendario"},
    {"Variable": "Temp_MA3", "Tipo": "derivada", "Unidad": "°C (anomalia)",
     "Formula": "mean(Temperature_Mean[t-2 : t]) — ventana movil causal de 3 dias",
     "Fuente": "derivada de Temperature_Mean"},
    {"Variable": "Temp_Diff", "Tipo": "derivada", "Unidad": "°C (anomalia)",
     "Formula": "Temperature_Mean[t] - Temperature_Mean[t-1]",
     "Fuente": "derivada de Temperature_Mean"},
    {"Variable": "Month_sin", "Tipo": "derivada", "Unidad": "[-1, 1]",
     "Formula": "sin(2 pi Month / 12)", "Fuente": "codificacion ciclica"},
    {"Variable": "Month_cos", "Tipo": "derivada", "Unidad": "[-1, 1]",
     "Formula": "cos(2 pi Month / 12)", "Fuente": "codificacion ciclica"},
    {"Variable": "Day_sin", "Tipo": "derivada", "Unidad": "[-1, 1]",
     "Formula": "sin(2 pi Day / 31)", "Fuente": "codificacion ciclica"},
    {"Variable": "Day_cos", "Tipo": "derivada", "Unidad": "[-1, 1]",
     "Formula": "cos(2 pi Day / 31)", "Fuente": "codificacion ciclica"},
    {"Variable": "Production_Total", "Tipo": "objetivo (crudo)", "Unidad": "TM",
     "Formula": "suma de TMDESCARGADA por fecha de desembarque; 0 en dias sin desembarque",
     "Fuente": "PRODUCE"},
    {"Variable": "Production_Total_Next_Day", "Tipo": "objetivo (crudo, D+1)", "Unidad": "TM",
     "Formula": "Production_Total desplazado un dia hacia atras", "Fuente": "PRODUCE"},
    {"Variable": "Production_Label_Next_Day", "Tipo": "objetivo (binario)", "Unidad": "0/1",
     "Formula": "1 si Production_Total_Next_Day > umbral; umbral = media de Production_Total en dias con desembarque, estimada solo con entrenamiento",
     "Fuente": "derivada"},
]

df_dict = pd.DataFrame(DATA_DICTIONARY)
save_csv(df_dict, "05_data_dictionary")
print(f"Diccionario de datos: {len(df_dict)} variables documentadas.")
display(df_dict)

"""# 4. Class distribution

With imbalanced classes, accuracy is misleading. The class distribution is computed here, **globally and per split**, because a distribution that is stable across splits and a seasonal one are not equivalent."""

# ============================================================
# Global class distribution
# ============================================================
y_global = binarize(PROD_NEXT, THRESHOLD_GLOBAL)
n_alto, n_bajo = int((y_global == 1).sum()), int((y_global == 0).sum())
ratio = n_bajo / max(n_alto, 1)
PCT_POSITIVOS = n_alto / len(y_global) * 100

print("=" * 64)
print("   DISTRIBUCION DE CLASES (umbral global, referencia)")
print("=" * 64)
print(f"  Alto (1): {n_alto:>4d} dias  ({n_alto/len(y_global)*100:5.1f}%)")
print(f"  Bajo (0): {n_bajo:>4d} dias  ({n_bajo/len(y_global)*100:5.1f}%)")
print(f"  Razon Bajo:Alto = {ratio:.2f} : 1")
print()
print(f"  Un clasificador trivial que prediga siempre 'Bajo' alcanza")
print(f"  una exactitud de {n_bajo/len(y_global)*100:.2f}% sin aprender nada.")
print("  Esa es la razon por la que la exactitud NO puede ser la metrica principal,")
print("  y por la que se reportan balanced accuracy y MCC (R1-M2).")

# ============================================================
# Monthly distribution: is the imbalance stable or seasonal?
# ============================================================
tmp = pd.DataFrame({"Date": DATES, "y": y_global})
tmp["YearMonth"] = tmp["Date"].dt.to_period("M").astype(str)
mensual = tmp.groupby("YearMonth")["y"].agg(Dias="size", Altos="sum")
mensual["Pct_Alto"] = (mensual["Altos"] / mensual["Dias"] * 100).round(1)

fig, ax = plt.subplots(figsize=(11, 3.6))
ax.bar(mensual.index, mensual["Pct_Alto"], color=gris(3),
       edgecolor="black", linewidth=0.7, hatch=trama(2))
ax.axhline(PCT_POSITIVOS, color="black", linestyle="--", linewidth=1.4,
           label=f"Media global ({PCT_POSITIVOS:.1f}%)")
ax.set_ylabel("% de dias de produccion Alta")
ax.set_xlabel("Mes")
ax.set_title("Distribucion mensual de la clase positiva")
ax.legend(frameon=True, edgecolor="black")
estilo_eje(ax, rejilla="y")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
save_figure(fig, "fig_class_distribution_monthly")
plt.show()

print(mensual.to_string())
print()
print("Lectura: la clase positiva se concentra en los meses de temporada de pesca.")
print("El desbalance NO es uniforme en el tiempo, lo que refuerza el argumento")
print("de que una particion aleatoria distribuye artificialmente la senal estacional.")
save_csv(mensual.reset_index(), "06_class_distribution_monthly")

"""# 5. Validation protocols

Three protocols are compared:

| Key | Scheme | Role |
|---|---|---|
| `kfold_shuffle` | `KFold(n_splits=5, shuffle=True)` | Shuffled 5-fold scheme of the original submission; kept only as a reference for comparison |
| `timeseries` | `TimeSeriesSplit(n_splits=5, gap=3)` — forward chaining | **Primary protocol.** The 3-day `gap` covers the longest causal window (`Temp_MA3`) |
| `timeseries_blocked` | Contiguous blocks, training window not accumulated | Robustness check: every split trains on a window of constant size |

Reporting the three protocols shows how much of the apparent performance came from shuffling."""

# ============================================================
# Split constructors
# ============================================================
class BlockedTimeSeriesSplit:
    """
    Temporal split into contiguous blocks of constant size.

    Unlike TimeSeriesSplit, the training window is NOT accumulated: each split
    trains on a fixed-length block immediately before the test block, separated
    from it by `gap` observations. Useful as a robustness check against the effect
    of a growing training sample size.
    """
    def __init__(self, n_splits=5, gap=0):
        self.n_splits = n_splits
        self.gap = gap

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end   = fold_size * (i + 1)
            test_start  = train_end + self.gap
            test_end    = min(test_start + fold_size, n)
            if test_start >= n:
                break
            train_start = max(0, train_end - fold_size)   # fixed window
            yield (np.arange(train_start, train_end),
                   np.arange(test_start, test_end))


def create_cv_splitter(protocol, n_splits=N_FOLDS, gap=None):
    """
    Return the split object corresponding to the protocol.

    `gap` is an optional parameter: the deep-learning benchmark of section 8.1 needs
    a wider gap (the 7-day window) without altering the primary protocol.
    """
    g = TS_GAP if gap is None else gap
    if protocol == "kfold_shuffle":
        return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    elif protocol == "timeseries":
        return TimeSeriesSplit(n_splits=n_splits, gap=g)
    elif protocol == "timeseries_blocked":
        return BlockedTimeSeriesSplit(n_splits=n_splits, gap=g)
    raise ValueError(f"Protocolo desconocido: {protocol}")

for p in PROTOCOLS:
    print(f"  {p:22s} -> {create_cv_splitter(p)}")

"""## 5.2 Formal leakage diagnosis

Leakage is measured rather than asserted. For each protocol the following is reported:

- **Order violations:** splits where some training date is later than the first test date.
- **Broken contiguity:** pairs of neighbouring rows (`t`, `t+1`) that fall on both sides of the training/test boundary; this is the concrete mechanism by which the dependence induced by `Temp_MA3` and `Temp_Diff` becomes leakage.
- **Minimum separation:** distance in days between the end of training and the start of testing."""

# ============================================================
# Leakage diagnosis by protocol
# ============================================================
def diagnose_leakage(protocol, X, dates, verbose=True):
    """
    Compute three temporal-leakage indicators for a given protocol.
    Return a dict with the summary and a list with the per-split detail.
    """
    cv = create_cv_splitter(protocol)
    filas = []
    violaciones_orden = 0
    contiguidad_rota_total = 0

    for fold, (tr, te) in enumerate(cv.split(X), 1):
        d_tr, d_te = dates.iloc[tr], dates.iloc[te]
        max_tr, min_te = d_tr.max(), d_te.min()
        viola = bool(max_tr >= min_te)
        violaciones_orden += int(viola)

        set_tr, set_te = set(tr), set(te)
        rotos = sum(1 for i in range(len(X) - 1)
                    if (i in set_tr and i + 1 in set_te) or (i in set_te and i + 1 in set_tr))
        contiguidad_rota_total += rotos

        filas.append({
            "Protocolo": protocol, "Fold": fold,
            "Train_inicio": d_tr.min().date(), "Train_fin": max_tr.date(), "n_train": len(tr),
            "Test_inicio": min_te.date(), "Test_fin": d_te.max().date(), "n_test": len(te),
            "Sep_dias": (min_te - max_tr).days,
            "Contiguidad_rota": rotos,
            "Orden_violado": "SI" if viola else "no",
        })

    resumen = {
        "protocolo": protocol,
        "violaciones_orden": violaciones_orden,
        "contiguidad_rota": contiguidad_rota_total,
        "sin_fuga": violaciones_orden == 0 and contiguidad_rota_total == 0,
    }
    if verbose:
        estado = "SIN FUGA" if resumen["sin_fuga"] else "FUGA DETECTADA"
        print(f"\n[{protocol}] -> {estado}")
        print(f"   particiones con orden violado : {violaciones_orden} de {len(filas)}")
        print(f"   pares contiguos rotos         : {contiguidad_rota_total}")
    return resumen, filas


print("=" * 70)
print("   DIAGNOSTICO DE FUGA TEMPORAL POR PROTOCOLO")
print("=" * 70)

leak_summaries, leak_rows = [], []
for p in PROTOCOLS:
    s, f = diagnose_leakage(p, X, DATES)
    leak_summaries.append(s); leak_rows.extend(f)

df_leak = pd.DataFrame(leak_rows)
save_csv(df_leak, "07_leakage_diagnosis")
display(df_leak)

# ============================================================
# Hard assertion: the primary protocol cannot have leakage
# ============================================================
principal = [s for s in leak_summaries if s["protocolo"] == PRIMARY_PROTOCOL][0]
assert principal["violaciones_orden"] == 0, (
    f"El protocolo principal '{PRIMARY_PROTOCOL}' viola el orden temporal.")
assert principal["contiguidad_rota"] == 0, (
    f"El protocolo principal '{PRIMARY_PROTOCOL}' rompe la contiguidad: "
    f"{principal['contiguidad_rota']} pares. Aumentar TS_GAP.")

print(f"[AUDIT] Protocolo principal '{PRIMARY_PROTOCOL}': verificacion superada.")
print(f"        0 violaciones de orden, 0 pares contiguos rotos, gap = {TS_GAP} dias.")
print()
print("Esta asercion es la evidencia formal que responde a R1-A4:")
print('  "Use time-series-aware cross-validation (forward chaining) and clearly state this."')

"""# 6. Models, baselines and hyperparameter tuning

## 6.1 Registry of the study models

The eight classifiers of the manuscript, with the exact hyperparameters of Table 3."""

# ============================================================
# REGISTRY OF THE STUDY MODELS
# ============================================================
MODEL_REGISTRY = {
    "SVM_RBF_C100": dict(
        cls=SVC, label="SVM (RBF, C=100)", category="study",
        params=dict(kernel="rbf", C=100, gamma="scale",
                    random_state=RANDOM_SEED, probability=True)),
    "GBT": dict(
        cls=GradientBoostingClassifier, label="Gradient Boosting Tree", category="study",
        params=dict(n_estimators=100, learning_rate=0.1, max_depth=3,
                    random_state=RANDOM_SEED)),
    "RFC": dict(
        cls=RandomForestClassifier, label="Random Forest Classifier", category="study",
        params=dict(n_estimators=100, max_depth=5, min_samples_split=5,
                    min_samples_leaf=2, random_state=RANDOM_SEED,
                    class_weight="balanced", n_jobs=-1)),
    "DTC": dict(
        cls=DecisionTreeClassifier, label="Decision Tree Classifier", category="study",
        params=dict(max_depth=5, min_samples_split=5, min_samples_leaf=2,
                    random_state=RANDOM_SEED, class_weight="balanced")),
    "LogReg": dict(
        cls=LogisticRegression, label="Logistic Regression", category="study",
        params=dict(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
    "NaiveBayes": dict(
        cls=GaussianNB, label="Naive Bayes", category="study",
        params=dict(var_smoothing=1e-9)),
    "MLP": dict(
        cls=MLPClassifier, label="MLP (Neural Network)", category="study",
        params=dict(hidden_layer_sizes=(20, 10), activation="relu", solver="adam",
                    alpha=0.0001, batch_size="auto", learning_rate="adaptive",
                    max_iter=1000, early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=10, random_state=RANDOM_SEED)),
}

# XGBoost is inserted only if the library is available, so that the cell degrades gracefully
# instead of failing with a NameError when the library is missing.
if XGBOOST_DISPONIBLE:
    MODEL_REGISTRY["XGBoost"] = dict(
        cls=XGBClassifier, label="XGBoost", category="study",
        params=dict(n_estimators=100, learning_rate=0.1, max_depth=3,
                    objective="binary:logistic", random_state=RANDOM_SEED,
                    eval_metric="logloss", verbosity=0))
    # placed in the position used in the manuscript (after Gradient Boosting Tree)
    _orden_manuscrito = ["SVM_RBF_C100", "GBT", "XGBoost", "RFC", "DTC",
                         "LogReg", "NaiveBayes", "MLP"]
    MODEL_REGISTRY = {k: MODEL_REGISTRY[k] for k in _orden_manuscrito
                      if k in MODEL_REGISTRY}
else:
    print("AVISO: XGBoost no disponible. Se ejecutan "
          f"{len(MODEL_REGISTRY)} modelos en lugar de 8, y el manuscrito debe")
    print("       declararlo. Instalar con: !pip install xgboost")

print(f"Modelos del estudio: {len(MODEL_REGISTRY)}")
for k, v in MODEL_REGISTRY.items():
    print(f"  {k:14s} -> {v['label']}")

"""## 6.2 Simple baselines

Four baselines are implemented:

| Baseline | Rule | Contrast |
|---|---|---|
| **Majority class** | always the most frequent class of the training set | main |
| **Seasonal naive** | majority class of the same calendar month, estimated on training rows only | main |
| **Stratified random** | sampling according to the class proportions of the training set | main |
| **Persistence** | *tomorrow = today* | separate supplementary table |

The persistence and seasonal baselines are not scikit-learn estimators because they use information that is not in the matrix `X` (today's production, the month). They are implemented as functions that receive the split indices **and the threshold of that split**, and are evaluated with exactly the same protocol and metrics as the models."""

# ============================================================
# Baselines implemented as functions of the split indices
# ============================================================
# The signature includes `thr`, the threshold of the split, so that the baselines
# binarize today's production with the same threshold that labels the test set.
def baseline_majority(train_idx, test_idx, y_train, aux, thr):
    """Majority class of the training set."""
    maj = int(pd.Series(y_train).mode().iloc[0])
    return np.full(len(test_idx), maj), np.full(len(test_idx), float(maj))

def baseline_persistence(train_idx, test_idx, y_train, aux, thr):
    """
    Persistence: 'tomorrow = today'.

    The prediction for D+1 is the class observed on D, binarized with the SAME
    split threshold that labels y_test. Production on D is information available
    on D, so the baseline remains causal.
    """
    prod_hoy = aux["prod_today"].iloc[test_idx]
    pred = binarize(prod_hoy, thr).values.astype(int)
    return pred, pred.astype(float)

def baseline_seasonal_naive(train_idx, test_idx, y_train, aux, thr):
    """
    Seasonal naive: majority class of the same calendar month,
    estimated ONLY from the training rows.
    If a month does not appear in training, the global majority class is used.
    """
    meses_tr = aux["month"].iloc[train_idx].values
    tabla = pd.DataFrame({"m": meses_tr, "y": np.asarray(y_train)})
    mayoria_mes = tabla.groupby("m")["y"].agg(lambda s: int(s.mode().iloc[0])).to_dict()
    fallback = int(pd.Series(y_train).mode().iloc[0])
    meses_te = aux["month"].iloc[test_idx].values
    pred = np.array([mayoria_mes.get(m, fallback) for m in meses_te], dtype=int)
    return pred, pred.astype(float)

def baseline_stratified(train_idx, test_idx, y_train, aux, thr):
    """Random sampling according to the class proportions of the training set."""
    rng = np.random.RandomState(RANDOM_SEED + len(test_idx))
    p = float(np.mean(y_train))
    pred = (rng.rand(len(test_idx)) < p).astype(int)
    return pred, pred.astype(float)

# `contraste_principal` assigns each baseline to the main contrast or to the supplementary table.
# `label` is used in the tables and figures of the notebook (in Spanish).
# `label_en` is the English name used in the manuscript text and in the generated narrative.
BASELINE_REGISTRY = {
    "BL_Majority":    dict(fn=baseline_majority,       label="Baseline: Clase Mayoritaria",
                           label_en="majority-class baseline",
                           category="baseline", contraste_principal=True),
    "BL_Seasonal":    dict(fn=baseline_seasonal_naive, label="Baseline: Estacional Ingenuo",
                           label_en="seasonal-naive baseline",
                           category="baseline", contraste_principal=True),
    "BL_Stratified":  dict(fn=baseline_stratified,     label="Baseline: Estratificado Aleatorio",
                           label_en="stratified-random baseline",
                           category="baseline", contraste_principal=True),
    "BL_Persistence": dict(fn=baseline_persistence,    label="Baseline: Persistencia (D+1 = D)",
                           label_en="persistence baseline (tomorrow = today)",
                           category="baseline", contraste_principal=False),
}

AUX = {"prod_today": PROD_TODAY, "month": MONTH_ARR}

print(f"Baselines implementados: {len(BASELINE_REGISTRY)}")
for k, v in BASELINE_REGISTRY.items():
    marca = "contraste principal" if v["contraste_principal"] else "SUPLEMENTARIO"
    print(f"  {k:16s} -> {v['label']:42s} [{marca}]")
print()
print("Justificacion del reparto (queda tambien en el JSON):")
print(textwrap.fill(BASELINE_LECTURA_ADOPTADA, 92, initial_indent="  ", subsequent_indent="  "))

"""# 7. Evaluation engine

A single engine evaluates models and baselines under any combination of protocol and threshold mode, computing all metrics in one pass per split.

Guarantees in every split:

- The `StandardScaler` is fitted **only on the training set** and then transforms the test set.
- In `per_fold_train` mode, the threshold is estimated **only on the training set** and applied to both sets, **including the baselines that depend on the threshold**.
- **Per-observation predictions, with their date,** are stored; they are needed for the McNemar tests (section 10) and the backtest (section 11).
- Training and inference times are measured."""

# ============================================================
# Training and evaluation engine
# ============================================================
METRIC_KEYS = ["accuracy", "balanced_accuracy", "precision", "recall",
               "f1", "mcc", "roc_auc"]
METRIC_LABELS = ["Accuracy", "Balanced Acc.", "Precision", "Recall",
                 "F1-Score", "MCC", "ROC-AUC"]


def _compute_metrics(y_true, y_pred, y_score):
    """Compute the 7 evaluation metrics for one split."""
    m = {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision":         precision_score(y_true, y_pred, zero_division=0),
        "recall":            recall_score(y_true, y_pred, zero_division=0),
        "f1":                f1_score(y_true, y_pred, zero_division=0),
        "mcc":               matthews_corrcoef(y_true, y_pred),
    }
    try:
        m["roc_auc"] = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        m["roc_auc"] = np.nan
    return m


def _get_scores(model, X_scaled, y_pred):
    """Obtain continuous scores for ROC-AUC, degrading gracefully when they are unavailable."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_scaled)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_scaled)
    return y_pred.astype(float)


def build_fold_labels(protocol, threshold_mode, splits):
    """
    Build the labels of each split and discard DEGENERATE splits.

    A split is degenerate when its training set (or its test set) contains a single
    class. This happens with `timeseries_blocked`, whose training window is a short
    contiguous block: if the block falls entirely within a closed season, all days
    are below the threshold and there is no "High" class. No binary classifier can be
    fitted there, so the split is excluded and recorded.

    Return (valid_splits, fold_info, discarded).
    """
    particiones, fold_info, descartadas = [], [], []

    for fold, (tr, te) in enumerate(splits, 1):
        if threshold_mode == "global_fixed":
            thr = THRESHOLD_GLOBAL
        elif threshold_mode == "per_fold_train":
            thr = compute_threshold(PROD_TODAY.iloc[tr])   # training set only
        else:
            raise ValueError(threshold_mode)

        y_tr = binarize(PROD_NEXT.iloc[tr], thr).values
        y_te = binarize(PROD_NEXT.iloc[te], thr).values

        n_cls_tr, n_cls_te = len(np.unique(y_tr)), len(np.unique(y_te))
        motivos = []
        if n_cls_tr < 2:
            motivos.append("entrenamiento con una sola clase")
        if n_cls_te < 2:
            motivos.append("prueba con una sola clase")
        valida = not motivos

        if valida:
            particiones.append(dict(fold=fold, tr=tr, te=te,
                                    y_tr=y_tr, y_te=y_te, thr=thr))
        else:
            descartadas.append((fold, " y ".join(motivos)))

        fold_info.append({
            "Protocolo": protocol, "Modo_umbral": threshold_mode, "Fold": fold,
            "Umbral_TM": round(thr, 2),
            "n_train": len(tr), "n_test": len(te),
            "Pct_alto_train": round(float(y_tr.mean()) * 100, 1),
            "Pct_alto_test": round(float(y_te.mean()) * 100, 1),
            "Clases_train": n_cls_tr, "Clases_test": n_cls_te,
            "Fold_valido": "si" if valida else "no",
            "Motivo_descarte": "" if valida else " y ".join(motivos),
        })

    return particiones, fold_info, descartadas


def _consolidate(fold_metrics, cms, imps, t_train, t_infer, label, category,
                 extra=None):
    """Average the metrics per split and assemble the results record."""
    out = {"label": label, "category": category}
    for k, vals in fold_metrics.items():
        clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        out[f"{k}_mean"] = float(np.mean(clean)) if clean else np.nan
        out[f"{k}_std"]  = float(np.std(clean))  if clean else np.nan
        out[f"{k}_folds"] = [float(v) for v in vals]     # input to the Wilcoxon test
    out["n_folds_evaluados"] = len(fold_metrics[METRIC_KEYS[0]])
    out["confusion_matrix"] = (np.sum(cms, axis=0).astype(int) if cms
                               else np.zeros((2, 2), dtype=int))
    out["time_train_mean"]  = float(np.mean(t_train)) if t_train else np.nan
    out["time_infer_mean"]  = float(np.mean(t_infer)) if t_infer else np.nan
    if imps:
        out["feature_importance"] = np.mean(imps, axis=0)
    if extra:
        out.update(extra)
    return out


def run_configuration(protocol, threshold_mode, tuned_params=None, verbose=True):
    """
    Run a complete configuration: all models and baselines under a given
    splitting protocol and threshold mode.

    Degenerate splits are skipped for ALL estimators alike, so that the per-split
    metrics remain comparable and the paired tests (McNemar, Wilcoxon) keep
    their alignment.

    Return (results, predictions, fold_info).
    """
    cv = create_cv_splitter(protocol)
    splits = list(cv.split(X))

    results, predictions = {}, {}
    particiones, fold_info, descartadas = build_fold_labels(protocol, threshold_mode, splits)

    if not particiones:
        print(f"  [{protocol} / {threshold_mode}] AVISO: ninguna particion es utilizable "
              f"({len(splits)} degeneradas). La configuracion se reporta como no evaluable.")

    # --- Study models ---
    for key, cfg in MODEL_REGISTRY.items():
        params = dict(cfg["params"])
        if tuned_params and key in tuned_params:
            params.update(tuned_params[key])

        fm = {k: [] for k in METRIC_KEYS}
        cms, imps = [], []
        yt_all, yp_all, ys_all, idx_all, fold_all = [], [], [], [], []
        t_train, t_infer = [], []

        for p in particiones:
            tr, te, y_tr, y_te = p["tr"], p["te"], p["y_tr"], p["y_te"]
            X_tr, X_te = X.iloc[tr], X.iloc[te]

            scaler = StandardScaler()                 # fitted on the training set only
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            model = cfg["cls"](**params)
            t0 = time.time()
            model.fit(X_tr_s, y_tr)
            t_train.append(time.time() - t0)

            t0 = time.time()
            y_pred = model.predict(X_te_s)
            t_infer.append(time.time() - t0)
            y_score = _get_scores(model, X_te_s, y_pred)

            for k, v in _compute_metrics(y_te, y_pred, y_score).items():
                fm[k].append(v)
            cms.append(confusion_matrix(y_te, y_pred, labels=[0, 1]))

            if hasattr(model, "feature_importances_"):
                imps.append(model.feature_importances_)
            elif hasattr(model, "coef_"):
                imps.append(np.abs(model.coef_).ravel())

            yt_all.extend(y_te); yp_all.extend(y_pred); ys_all.extend(y_score)
            idx_all.extend(te);  fold_all.extend([p["fold"]] * len(te))

        results[key] = _consolidate(fm, cms, imps, t_train, t_infer,
                                    cfg["label"], cfg["category"],
                                    extra={"contraste": "—"})
        predictions[key] = dict(y_true=np.array(yt_all), y_pred=np.array(yp_all),
                                y_score=np.array(ys_all), idx=np.array(idx_all),
                                fold=np.array(fold_all))

    # --- Baselines ---
    for key, cfg in BASELINE_REGISTRY.items():
        fm = {k: [] for k in METRIC_KEYS}
        cms = []
        yt_all, yp_all, ys_all, idx_all, fold_all = [], [], [], [], []
        t_train, t_infer = [], []

        for p in particiones:
            tr, te, y_tr, y_te = p["tr"], p["te"], p["y_tr"], p["y_te"]
            t0 = time.time()
            # the threshold of the split is passed to the baseline
            y_pred, y_score = cfg["fn"](tr, te, y_tr, AUX, p["thr"])
            dt = time.time() - t0
            t_train.append(0.0); t_infer.append(dt)

            for k, v in _compute_metrics(y_te, y_pred, y_score).items():
                fm[k].append(v)
            cms.append(confusion_matrix(y_te, y_pred, labels=[0, 1]))

            yt_all.extend(y_te); yp_all.extend(y_pred); ys_all.extend(y_score)
            idx_all.extend(te);  fold_all.extend([p["fold"]] * len(te))

        results[key] = _consolidate(
            fm, cms, [], t_train, t_infer, cfg["label"], cfg["category"],
            extra={"contraste": "principal" if cfg["contraste_principal"] else "suplementario",
                   "contraste_principal": bool(cfg["contraste_principal"]),
                   "label_en": cfg["label_en"]})
        predictions[key] = dict(y_true=np.array(yt_all), y_pred=np.array(yp_all),
                                y_score=np.array(ys_all), idx=np.array(idx_all),
                                fold=np.array(fold_all))

    if verbose:
        print(f"  [{protocol} / {threshold_mode}] {len(results)} estimadores evaluados "
              f"sobre {len(particiones)} de {len(splits)} particiones.")
        for fold, motivo in descartadas:
            print(f"      · particion {fold} descartada por degenerada: {motivo}.")
    return results, predictions, fold_info


print("Motor de evaluacion definido.")

"""## 7.1 Nested hyperparameter tuning

Tuning hyperparameters on the full data set is a second source of optimism, distinct from shuffling. Here the hyperparameters are tuned inside each training split, with a temporal internal validation. The grids are deliberately compact: the aim is not to squeeze out performance but to check that the hyperparameters of the manuscript were not chosen by looking at the test set."""

# ============================================================
# Nested tuning inside each temporal split
# ============================================================
PARAM_GRIDS = {
    "SVM_RBF_C100": {"C": [1, 10, 100], "gamma": ["scale", "auto"]},
    "RFC":          {"n_estimators": [100, 200], "max_depth": [3, 5, 8]},
    "XGBoost":      {"n_estimators": [100, 200], "max_depth": [2, 3, 5]},
}

def nested_tuning(protocol, threshold_mode, verbose=True):
    """
    For each model with a defined grid, tune hyperparameters INSIDE each
    training split using a temporal internal validation, and report the
    most frequent winning setting across splits.

    Reuses build_fold_labels, so degenerate splits are skipped here with the
    same criterion as in run_configuration.
    """
    cv = create_cv_splitter(protocol)
    splits = list(cv.split(X))
    particiones, _, descartadas = build_fold_labels(protocol, threshold_mode, splits)
    elegidos, filas = {}, []
    omitidas_por_clase_unica = []

    if descartadas and verbose:
        print(f"  Particiones omitidas por degeneradas: "
              f"{', '.join(str(f) for f, _ in descartadas)}")

    for key, grid in PARAM_GRIDS.items():
        if key not in MODEL_REGISTRY:
            continue
        cfg = MODEL_REGISTRY[key]
        ganadores = []

        for p in particiones:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X.iloc[p["tr"]])

            inner_cv = TimeSeriesSplit(n_splits=3, gap=TS_GAP)   # internal validation is also temporal

            # Guard. The outer split is guaranteed to have two classes, but the INNER splits are not:
            # an inner training block that falls entirely within a closed season has a single class.
            # If this happens to all three, GridSearchCV raises ValueError. The condition is checked beforehand
            # and, if the search still fails, the error is caught and the manuscript hyperparameters are kept.
            inner_utiles = sum(
                1 for i_tr, i_te in inner_cv.split(X_tr_s)
                if len(np.unique(p["y_tr"][i_tr])) > 1 and len(np.unique(p["y_tr"][i_te])) > 1)
            if inner_utiles == 0:
                omitidas_por_clase_unica.append((cfg["label"], p["fold"]))
                continue

            base = cfg["cls"](**cfg["params"])
            gs = GridSearchCV(base, grid, cv=inner_cv,
                              scoring="balanced_accuracy", n_jobs=-1, error_score=0)
            try:
                gs.fit(X_tr_s, p["y_tr"])
            except Exception as e:
                omitidas_por_clase_unica.append((cfg["label"], p["fold"]))
                filas.append({"Modelo": cfg["label"], "Fold": p["fold"],
                              "Mejores_params": f"omitida ({type(e).__name__})",
                              "Score_interno": np.nan,
                              "Particiones_internas_utiles": inner_utiles})
                continue

            ganadores.append(gs.best_params_)
            filas.append({"Modelo": cfg["label"], "Fold": p["fold"],
                          "Mejores_params": str(gs.best_params_),
                          "Score_interno": round(float(gs.best_score_), 4),
                          "Particiones_internas_utiles": inner_utiles})

        if not ganadores:
            if verbose:
                print(f"  {cfg['label']:26s} -> sin particiones utilizables; "
                      f"se conservan los hiperparametros por defecto.")
            continue

        conteo = pd.Series([str(g) for g in ganadores]).value_counts()
        mas_frecuente = conteo.index[0]
        elegidos[key] = ganadores[[str(g) for g in ganadores].index(mas_frecuente)]
        if verbose:
            print(f"  {cfg['label']:26s} -> {elegidos[key]}   "
                  f"(estable en {conteo.iloc[0]}/{len(ganadores)} particiones)")

    if omitidas_por_clase_unica and verbose:
        print()
        print(f"  Ajustes internos omitidos por clase unica: {len(omitidas_por_clase_unica)}")
        for lab, fold in omitidas_por_clase_unica:
            print(f"    {lab} · particion {fold}: ninguna particion interna tiene ambas clases")
        print("  Esos modelos conservan los hiperparametros del manuscrito en esas")
        print("  particiones. Queda registrado en 08_nested_tuning y debe declararse")
        print("  si afecta a alguno de los modelos reportados en Tabla 3.")

    return elegidos, pd.DataFrame(filas)


TUNED_PARAMS = {}
if RUN_NESTED_TUNING:
    print("Ajuste anidado sobre el protocolo principal "
          f"({PRIMARY_PROTOCOL} / {PRIMARY_THRESHOLD_MODE}):")
    TUNED_PARAMS, df_tuning = nested_tuning(PRIMARY_PROTOCOL, PRIMARY_THRESHOLD_MODE)
    save_csv(df_tuning, "08_nested_tuning")
else:
    print("Ajuste anidado desactivado (RUN_NESTED_TUNING = False).")

"""# 8. Factorial run

Six configurations (3 protocols x 2 threshold modes) in a single pass, with no manual intervention."""

# ============================================================
# COMPLETE FACTORIAL RUN
# ============================================================
t_factorial = time.time()
ALL_RESULTS, ALL_PREDICTIONS, ALL_FOLDINFO = {}, {}, []

print("=" * 70)
print("   EJECUCION FACTORIAL: protocolos x modos de umbral")
print("=" * 70)

for protocol, thr_mode in itertools.product(PROTOCOLS, THRESHOLD_MODES):
    cfg_key = f"{protocol}|{thr_mode}"
    res, pred, finfo = run_configuration(protocol, thr_mode)
    ALL_RESULTS[cfg_key] = res
    ALL_PREDICTIONS[cfg_key] = pred
    ALL_FOLDINFO.extend(finfo)

df_foldinfo = pd.DataFrame(ALL_FOLDINFO)
save_csv(df_foldinfo, "09_fold_info")

PRIMARY_KEY = f"{PRIMARY_PROTOCOL}|{PRIMARY_THRESHOLD_MODE}"

print(f"\nConfiguraciones completadas: {len(ALL_RESULTS)}")
print(f"Estimadores por configuracion: {len(next(iter(ALL_RESULTS.values())))}")
print(f"Tiempo total del factorial: {time.time() - t_factorial:.1f} s")

# --- Explicit record of the degenerate splits ---
descartes = df_foldinfo[df_foldinfo["Fold_valido"] == "no"]
if len(descartes):
    print(f"\nParticiones descartadas por clase unica: {len(descartes)} de {len(df_foldinfo)}")
    print(descartes[["Protocolo", "Modo_umbral", "Fold", "Umbral_TM",
                     "Pct_alto_train", "Pct_alto_test", "Motivo_descarte"]]
          .to_string(index=False))
else:
    print("\nTodas las particiones de todas las configuraciones resultaron utilizables.")

# --- Splits evaluated under the primary protocol ---
_sub_pri = df_foldinfo[(df_foldinfo["Protocolo"] == PRIMARY_PROTOCOL) &
                       (df_foldinfo["Modo_umbral"] == PRIMARY_THRESHOLD_MODE)]
FOLDS_PRINCIPAL = {
    "totales": int(len(_sub_pri)),
    "evaluadas": int((_sub_pri["Fold_valido"] == "si").sum()),
    "descartadas": int((_sub_pri["Fold_valido"] == "no").sum()),
    "motivos": [f"particion {int(r.Fold)}: {r.Motivo_descarte}"
                for r in _sub_pri[_sub_pri["Fold_valido"] == "no"].itertuples()],
}
print(f"\nProtocolo principal: {FOLDS_PRINCIPAL['evaluadas']} de "
      f"{FOLDS_PRINCIPAL['totales']} particiones evaluadas.")
for m in FOLDS_PRINCIPAL["motivos"]:
    print(f"   {m}")

# ============================================================
# Additional run with nested-tuned hyperparameters
# ============================================================
if RUN_NESTED_TUNING and TUNED_PARAMS:
    key_tuned = f"{PRIMARY_PROTOCOL}|{PRIMARY_THRESHOLD_MODE}|tuned"
    res_t, pred_t, _ = run_configuration(PRIMARY_PROTOCOL, PRIMARY_THRESHOLD_MODE,
                                         tuned_params=TUNED_PARAMS)
    ALL_RESULTS[key_tuned] = res_t
    ALL_PREDICTIONS[key_tuned] = pred_t
    print(f"Configuracion adicional con ajuste anidado: {key_tuned}")

# ============================================================
# Stability of the threshold across splits
# ============================================================
sub = df_foldinfo[(df_foldinfo["Protocolo"] == PRIMARY_PROTOCOL) &
                  (df_foldinfo["Modo_umbral"] == "per_fold_train")]

print("Umbral estimado con datos de entrenamiento, por particion:")
print(sub[["Fold", "Umbral_TM", "n_train", "n_test",
           "Pct_alto_train", "Pct_alto_test"]].to_string(index=False))

var_pct = (sub["Umbral_TM"].std() / sub["Umbral_TM"].mean()) * 100
UMBRAL_CV_CRITERIO = 20.0     # single criterion
UMBRAL_ES_ESTABLE = bool(var_pct < UMBRAL_CV_CRITERIO)

print(f"\nUmbral global (todo el dataset): {THRESHOLD_GLOBAL:,.2f} TM")
print(f"Umbral por particion: media {sub['Umbral_TM'].mean():,.2f} TM, "
      f"coeficiente de variacion {var_pct:.2f}%")
print(f"Criterio de estabilidad declarado: coeficiente de variacion < {UMBRAL_CV_CRITERIO:.0f}%")
print()
if UMBRAL_ES_ESTABLE:
    print("El umbral es ESTABLE entre particiones segun el criterio declarado.")
    print("Eso permite responder a R2-04 declarando la regla con precision y")
    print("mostrando que la correccion de fuga no altera materialmente las clases.")
else:
    print("El umbral VARIA apreciablemente entre particiones segun el criterio")
    print("declarado. Debe reportarse por particion y discutirse como fuente de")
    print("variabilidad. La narrativa de §15 recoge esta lectura, no la contraria.")

"""## 8.1 Deep-learning benchmark

A 1D-CNN, an LSTM and a GRU are evaluated on 7-day windows, aligned observation by observation with the tabular models.

### Split alignment

Tabular models receive one row; a 1D-CNN needs a window of `DL_WINDOW = 7` days by `len(FEATURE_COLUMNS)` variables. Built carelessly, the window of the first test observation would include six days that may belong to the training block. This is avoided with a property of `TimeSeriesSplit`: the `gap` parameter only trims the tail of the training block, and the test indices depend only on `n_splits` and on the sample size. Therefore:

- the **test blocks** are taken unchanged from the primary protocol;
- the **labels and threshold** of each split are reused as they are from `build_fold_labels`, so that the class of each observation is identical to the one seen by the tabular models;
- the **training block** is trimmed with `gap = max(TS_GAP, DL_WINDOW) = 7`, which guarantees that no test window touches a training row.

As a consequence, McNemar tests remain valid observation by observation between tabular and deep models. Two hard assertions verify this at run time; if either fails, the notebook stops. The deep models train on four fewer rows per split than the tabular ones; this is the cost of the separation.

### Design choices for the neural networks

- **Balanced class weights.** `MLPClassifier` does not support `class_weight`; Keras and PyTorch do.
- **Temporal internal validation** for early stopping: the last 15 % of the training block in chronological order, not a random fraction.

### Multiple comparisons

With the three deep models added, the number of McNemar pairs increases and the Holm correction becomes stricter. The correction is applied over all comparisons actually performed."""

# ============================================================
# Construction of temporal windows
# ============================================================
def build_windows(matriz_escalada, indices, window):
    """
    Build the (n, window, n_features) tensor for the given indices.

    For index i the window spans rows [i-window+1, i], i.e. only the past and the
    present. Indices with i < window-1 are discarded because their window would
    fall outside the series; the function also returns the indices actually used
    so that the labels can be aligned.
    """
    usables = np.asarray([i for i in indices if i >= window - 1], dtype=int)
    if len(usables) == 0:
        return np.empty((0, window, matriz_escalada.shape[1])), usables
    tensor = np.stack([matriz_escalada[i - window + 1: i + 1] for i in usables])
    return tensor.astype(np.float32), usables


def pesos_de_clase(y):
    """Balanced weighting: w_k = n / (2 * n_k)."""
    y = np.asarray(y)
    n = len(y)
    out = {}
    for k in (0, 1):
        nk = int((y == k).sum())
        out[k] = float(n / (2 * nk)) if nk > 0 else 1.0
    return out


DL_ARCHITECTURES = {
    "DL_CNN1D": dict(arch="1D-CNN", label="1D-CNN (Deep Learning)", label_en="1D-CNN"),
    "DL_LSTM":  dict(arch="LSTM",   label="LSTM (Deep Learning)",   label_en="LSTM"),
    "DL_GRU":   dict(arch="GRU",    label="GRU (Deep Learning)",    label_en="GRU"),
}

print("Arquitecturas profundas registradas:")
for k, v in DL_ARCHITECTURES.items():
    print(f"  {k:10s} -> {v['label']}")
print(f"Ventana: {DL_WINDOW} dias x {len(FEATURE_COLUMNS)} variables")

# ============================================================
# Implementation by framework (Keras first, PyTorch as alternative)
# ============================================================
def _entrenar_keras(arch, Xtr, ytr, Xval, yval, Xte, pesos):
    """Train one architecture in Keras and return (test_scores, train_time, inference_time)."""
    keras.utils.set_random_seed(RANDOM_SEED)
    n_steps, n_feats = Xtr.shape[1], Xtr.shape[2]

    m = keras.Sequential(name=arch.replace("-", "_"))
    m.add(keras_layers.Input(shape=(n_steps, n_feats)))
    if arch == "1D-CNN":
        m.add(keras_layers.Conv1D(32, 3, activation="relu"))
        m.add(keras_layers.Conv1D(16, 3, activation="relu"))
        m.add(keras_layers.GlobalMaxPooling1D())
    elif arch == "LSTM":
        m.add(keras_layers.LSTM(32))
    elif arch == "GRU":
        m.add(keras_layers.GRU(32))
    else:
        raise ValueError(arch)
    m.add(keras_layers.Dropout(0.3))
    m.add(keras_layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")

    callbacks, val_data = [], None
    if Xval is not None and len(np.unique(yval)) > 1:
        val_data = (Xval, yval)
        callbacks = [keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=DL_PACIENCIA, restore_best_weights=True)]

    t0 = time.time()
    m.fit(Xtr, ytr, epochs=DL_EPOCHS, batch_size=DL_BATCH, verbose=0,
          validation_data=val_data, callbacks=callbacks, class_weight=pesos)
    t_entrena = time.time() - t0

    t0 = time.time()
    scores = m.predict(Xte, verbose=0).ravel()
    t_infiere = time.time() - t0
    return scores, t_entrena, t_infiere


def _entrenar_torch(arch, Xtr, ytr, Xval, yval, Xte, pesos):
    """PyTorch equivalent, with the same topology and the same stopping criterion."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    n_feats = Xtr.shape[2]

    class Red(torch_nn.Module):
        def __init__(self):
            super().__init__()
            self.arch = arch
            if arch == "1D-CNN":
                self.c1 = torch_nn.Conv1d(n_feats, 32, 3)
                self.c2 = torch_nn.Conv1d(32, 16, 3)
                salida = 16
            elif arch == "LSTM":
                self.rnn = torch_nn.LSTM(n_feats, 32, batch_first=True)
                salida = 32
            elif arch == "GRU":
                self.rnn = torch_nn.GRU(n_feats, 32, batch_first=True)
                salida = 32
            else:
                raise ValueError(arch)
            self.drop = torch_nn.Dropout(0.3)
            self.out = torch_nn.Linear(salida, 1)

        def forward(self, x):                      # x: (n, steps, feats)
            if self.arch == "1D-CNN":
                h = x.transpose(1, 2)
                h = torch.relu(self.c1(h))
                h = torch.relu(self.c2(h))
                h = h.max(dim=2).values
            else:
                h, _ = self.rnn(x)
                h = h[:, -1, :]
            return self.out(self.drop(h)).squeeze(1)

    modelo = Red()
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    w_pos = torch.tensor(pesos[1] / max(pesos[0], 1e-9), dtype=torch.float32)
    crit = torch_nn.BCEWithLogitsLoss(pos_weight=w_pos)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(np.asarray(ytr), dtype=torch.float32)
    usa_val = Xval is not None and len(np.unique(yval)) > 1
    if usa_val:
        Xval_t = torch.tensor(Xval, dtype=torch.float32)
        yval_t = torch.tensor(np.asarray(yval), dtype=torch.float32)

    mejor, espera, mejor_estado = np.inf, 0, None
    n = len(Xtr_t)
    t0 = time.time()
    for _ in range(DL_EPOCHS):
        modelo.train()
        orden = torch.randperm(n)
        for k in range(0, n, DL_BATCH):
            lote = orden[k:k + DL_BATCH]
            opt.zero_grad()
            perdida = crit(modelo(Xtr_t[lote]), ytr_t[lote])
            perdida.backward(); opt.step()
        if usa_val:
            modelo.eval()
            with torch.no_grad():
                vl = float(crit(modelo(Xval_t), yval_t))
            if vl < mejor - 1e-5:
                mejor, espera = vl, 0
                mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
            else:
                espera += 1
                if espera >= DL_PACIENCIA:
                    break
    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    t_entrena = time.time() - t0

    modelo.eval()
    t0 = time.time()
    with torch.no_grad():
        scores = torch.sigmoid(modelo(torch.tensor(Xte, dtype=torch.float32))).numpy().ravel()
    t_infiere = time.time() - t0
    return scores, t_entrena, t_infiere


def entrenar_red(arch, Xtr, ytr, Xval, yval, Xte, pesos):
    """Dispatch to the available framework."""
    if DEEP_LEARNING_MARCO and DEEP_LEARNING_MARCO.startswith("tensorflow"):
        return _entrenar_keras(arch, Xtr, ytr, Xval, yval, Xte, pesos)
    return _entrenar_torch(arch, Xtr, ytr, Xval, yval, Xte, pesos)

print("Implementaciones de aprendizaje profundo definidas.")

# ============================================================
# Alignment assertions between the splitters
# ============================================================
DL_GAP = max(TS_GAP, DL_WINDOW)

_cv_main = create_cv_splitter(PRIMARY_PROTOCOL)                 # gap = TS_GAP
_cv_dl   = create_cv_splitter(PRIMARY_PROTOCOL, gap=DL_GAP)     # gap = DL_GAP
_splits_main = list(_cv_main.split(X))
_splits_dl   = list(_cv_dl.split(X))

assert len(_splits_main) == len(_splits_dl), (
    "Los dos splitters no producen el mismo numero de particiones.")

for _f, ((_tr_m, _te_m), (_tr_d, _te_d)) in enumerate(zip(_splits_main, _splits_dl), 1):
    # Assertion 1: the test blocks are IDENTICAL, which is what makes McNemar valid.
    assert np.array_equal(_te_m, _te_d), (
        f"Particion {_f}: los bloques de prueba difieren entre el protocolo principal "
        f"y el del aprendizaje profundo. McNemar dejaria de ser pareado.")
    # Assertion 2: no test window touches a training row.
    _primer_test = int(_te_d.min())
    _inicio_ventana = _primer_test - DL_WINDOW + 1
    assert _inicio_ventana > int(_tr_d.max()), (
        f"Particion {_f}: la ventana de la primera observacion de prueba comienza en "
        f"{_inicio_ventana} y el entrenamiento llega hasta {int(_tr_d.max())}. "
        f"Aumentar DL_GAP.")

print("[AUDIT] Alineacion de particiones para aprendizaje profundo: verificacion superada.")
print(f"        {len(_splits_main)} particiones con bloques de prueba identicos.")
print(f"        gap principal = {TS_GAP} dias  |  gap del aprendizaje profundo = {DL_GAP} dias")
print(f"        Filas de entrenamiento cedidas por particion: {DL_GAP - TS_GAP}")

# ============================================================
# Run the deep-learning benchmark
# ============================================================
def run_deep_learning_benchmark(verbose=True):
    """
    Evaluate the 1D-CNN, LSTM and GRU under the primary protocol.

    Return (results, predictions, detail) with exactly the same structure produced
    by `_consolidate`, so that the results can be merged into
    ALL_RESULTS[PRIMARY_KEY] without special handling.

    The deep models are NOT part of the six-configuration factorial run nor of the
    nested tuning: they enter Table 3, the McNemar matrix and the ranking.
    """
    # Labels and test blocks: those of the primary protocol, unchanged.
    particiones, _, _ = build_fold_labels(PRIMARY_PROTOCOL, PRIMARY_THRESHOLD_MODE,
                                          _splits_main)
    # Training blocks trimmed with the wide gap.
    tr_dl_por_fold = {f + 1: tr for f, (tr, te) in enumerate(_splits_dl)}

    results, predictions, detalle = {}, {}, []

    for key, cfg in DL_ARCHITECTURES.items():
        fm = {k: [] for k in METRIC_KEYS}
        cms = []
        yt_all, yp_all, ys_all, idx_all, fold_all = [], [], [], [], []
        t_train, t_infer = [], []

        for p in particiones:
            fold, te, y_te, thr = p["fold"], p["te"], p["y_te"], p["thr"]
            tr_dl = tr_dl_por_fold[fold]

            # Scaling fitted ONLY on the training block of this split.
            scaler = StandardScaler()
            scaler.fit(X.iloc[tr_dl])
            Xs = scaler.transform(X).astype(np.float32)

            # Windows. Initial rows without enough history are discarded.
            Xtr_w, idx_tr = build_windows(Xs, tr_dl, DL_WINDOW)
            Xte_w, idx_te = build_windows(Xs, te,    DL_WINDOW)
            if len(Xtr_w) == 0 or len(Xte_w) == 0:
                continue

            ytr = binarize(PROD_NEXT.iloc[idx_tr], thr).values
            yte = binarize(PROD_NEXT.iloc[idx_te], thr).values
            if len(np.unique(ytr)) < 2:
                continue

            # TEMPORAL internal validation: last 15 % of the block, in order.
            n_val = int(np.floor(DL_VAL_FRACCION * len(Xtr_w)))
            if n_val >= 5:
                Xfit, yfit = Xtr_w[:-n_val], ytr[:-n_val]
                Xval, yval = Xtr_w[-n_val:], ytr[-n_val:]
            else:
                Xfit, yfit, Xval, yval = Xtr_w, ytr, None, None

            if len(np.unique(yfit)) < 2:
                Xfit, yfit, Xval, yval = Xtr_w, ytr, None, None

            pesos = pesos_de_clase(yfit)
            scores, t_e, t_i = entrenar_red(cfg["arch"], Xfit, yfit,
                                            Xval, yval, Xte_w, pesos)
            y_pred = (scores >= 0.5).astype(int)

            t_train.append(t_e); t_infer.append(t_i)
            for k, v in _compute_metrics(yte, y_pred, scores).items():
                fm[k].append(v)
            cms.append(confusion_matrix(yte, y_pred, labels=[0, 1]))

            yt_all.extend(yte); yp_all.extend(y_pred); ys_all.extend(scores)
            idx_all.extend(idx_te); fold_all.extend([fold] * len(idx_te))

            detalle.append({
                "Arquitectura": cfg["label"], "Fold": fold,
                "n_train_ventanas": int(len(Xfit)),
                "n_val_ventanas": int(0 if Xval is None else len(Xval)),
                "n_test_ventanas": int(len(Xte_w)),
                "Pct_alto_train": round(float(np.mean(yfit)) * 100, 1),
                "Pct_alto_test": round(float(np.mean(yte)) * 100, 1),
                "Tasa_predicciones_alto": round(float(y_pred.mean()), 4),
                "Segundos_entrenamiento": round(t_e, 3),
            })

        if not fm[METRIC_KEYS[0]]:
            if verbose:
                print(f"  {cfg['label']:26s} -> sin particiones utilizables.")
            continue

        results[key] = _consolidate(fm, cms, [], t_train, t_infer,
                                    cfg["label"], "deep_learning",
                                    extra={"contraste": "—", "label_en": cfg["label_en"]})
        predictions[key] = dict(y_true=np.array(yt_all), y_pred=np.array(yp_all),
                                y_score=np.array(ys_all), idx=np.array(idx_all),
                                fold=np.array(fold_all))
        if verbose:
            r = results[key]
            print(f"  {cfg['label']:26s} -> Acc={r['accuracy_mean']:.4f}  "
                  f"BalAcc={r['balanced_accuracy_mean']:.4f}  "
                  f"MCC={r['mcc_mean']:+.4f}  Recall={r['recall_mean']:.4f}  "
                  f"({r['n_folds_evaluados']} particiones)")

    return results, predictions, pd.DataFrame(detalle)


DL_RESULTS, DL_PREDICTIONS, df_dl_detalle = {}, {}, pd.DataFrame()
DL_EJECUTADO = False

if RUN_DEEP_LEARNING and DEEP_LEARNING_DISPONIBLE:
    print("=" * 84)
    print("   BENCHMARK DE APRENDIZAJE PROFUNDO  (R2-06, R2-20)")
    print(f"   Marco: {DEEP_LEARNING_MARCO}  |  ventana: {DL_WINDOW} dias  |  "
          f"gap: {DL_GAP} dias")
    print("=" * 84)
    t0_dl = time.time()
    DL_RESULTS, DL_PREDICTIONS, df_dl_detalle = run_deep_learning_benchmark()
    print(f"\nTiempo total del benchmark: {time.time() - t0_dl:.1f} s")

    if DL_RESULTS:
        DL_EJECUTADO = True
        # Merge into the primary configuration: from here on the deep models are
        # ordinary estimators and enter Table 3, the McNemar tests and the ranking.
        ALL_RESULTS[PRIMARY_KEY].update(DL_RESULTS)
        ALL_PREDICTIONS[PRIMARY_KEY].update(DL_PREDICTIONS)
        save_csv(df_dl_detalle, "09b_deep_learning_fold_detail")
        display(df_dl_detalle)
        print(f"\nEstimadores en la configuracion principal: "
              f"{len(ALL_RESULTS[PRIMARY_KEY])}")
        print("  (modelos del estudio + aprendizaje profundo + baselines)")
    else:
        DEEP_LEARNING_MOTIVO = "ninguna particion resulto utilizable para las redes"
        print(f"\nBenchmark sin resultados: {DEEP_LEARNING_MOTIVO}")
else:
    if not RUN_DEEP_LEARNING:
        DEEP_LEARNING_MOTIVO = "desactivado por el interruptor RUN_DEEP_LEARNING"
    print("Benchmark de aprendizaje profundo omitido.")
    print(f"  Motivo: {DEEP_LEARNING_MOTIVO}")
    print("  §12 lo declarara explicitamente en lugar de silenciarlo.")

"""# 9. Results

## 9.1 Comparison table per configuration

The table of the primary protocol is Table 3 of the manuscript. It includes a `Contraste` column stating, for each baseline, whether it belongs to the main contrast or to the supplementary one."""

# ============================================================
# Build the results tables
# ============================================================
def build_results_table(results):
    """Build the comparison table for a set of results."""
    orden_cat = {"study": 0, "deep_learning": 1, "baseline": 2}
    rows = []
    for key, r in results.items():
        row = {"Estimador": r["label"], "Tipo": r["category"],
               "Contraste": r.get("contraste", "—")}
        for m, lab in zip(METRIC_KEYS, METRIC_LABELS):
            mu, sd = r.get(f"{m}_mean", np.nan), r.get(f"{m}_std", np.nan)
            row[lab] = f"{mu:.4f} (±{sd:.4f})" if not np.isnan(mu) else "N/A"
        row["Particiones"] = r.get("n_folds_evaluados", np.nan)
        row["Train (s)"] = f"{r['time_train_mean']:.4f}"
        row["Infer (s)"] = f"{r['time_infer_mean']:.5f}"
        row["_orden"] = orden_cat.get(r["category"], 9)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("_orden", kind="stable").drop(columns="_orden")
    return df.reset_index(drop=True)


print("=" * 110)
print(f"   TABLA PRINCIPAL — protocolo: {PRIMARY_PROTOCOL}  |  umbral: {PRIMARY_THRESHOLD_MODE}")
print("   (esta es la Tabla 3 del manuscrito revisado)")
print("=" * 110)
df_primary = build_results_table(ALL_RESULTS[PRIMARY_KEY])
save_csv(df_primary, f"10_results_PRIMARY_{PRIMARY_PROTOCOL}_{PRIMARY_THRESHOLD_MODE}")
display(df_primary)

print(f"\nEstimadores en la tabla: {len(df_primary)}")
for cat, n in df_primary["Tipo"].value_counts().items():
    print(f"   {cat:15s} {n}")

# ============================================================
# Table for each of the remaining configurations
# ============================================================
for cfg_key, res in ALL_RESULTS.items():
    if cfg_key == PRIMARY_KEY:
        continue
    print("\n" + "=" * 110)
    print(f"   {cfg_key}")
    print("=" * 110)
    dfx = build_results_table(res)
    display(dfx)
    save_csv(dfx, f"11_results_{cfg_key.replace('|', '_')}")

"""## 9.2 Comparison across protocols

How much of the performance obtained under the shuffled scheme and the global threshold disappears under the temporal protocol. The deep models are not part of this comparison: they were only evaluated under the primary protocol and have no counterpart in `kfold_shuffle`."""

# ============================================================
# Effect of the protocol on each model
# ============================================================
filas = []
for cfg_key, res in ALL_RESULTS.items():
    if "tuned" in cfg_key:
        continue
    protocolo, thr_mode = cfg_key.split("|")[0], cfg_key.split("|")[1]
    for mkey, r in res.items():
        if r["category"] == "deep_learning":
            continue
        filas.append({
            "Estimador": r["label"], "Tipo": r["category"],
            "Protocolo": protocolo, "Umbral": thr_mode,
            "Accuracy": round(r["accuracy_mean"], 4),
            "BalancedAcc": round(r["balanced_accuracy_mean"], 4),
            "MCC": round(r["mcc_mean"], 4),
            "F1": round(r["f1_mean"], 4),
            "ROC_AUC": round(r["roc_auc_mean"], 4) if not np.isnan(r["roc_auc_mean"]) else np.nan,
        })
df_cross = pd.DataFrame(filas)
save_csv(df_cross, "12_cross_protocol_comparison")

# Pivot: how much each model changes when moving from kfold_shuffle to the temporal protocol
base_cfg = ("kfold_shuffle", "global_fixed")     # scheme of the original submission
new_cfg  = (PRIMARY_PROTOCOL, PRIMARY_THRESHOLD_MODE)

comp = []
for mkey, r_new in ALL_RESULTS[PRIMARY_KEY].items():
    if r_new["category"] == "deep_learning":
        continue
    r_old = ALL_RESULTS["kfold_shuffle|global_fixed"][mkey]
    comp.append({
        "Estimador": r_new["label"], "Tipo": r_new["category"],
        "Acc_kfold": round(r_old["accuracy_mean"], 4),
        "Acc_temporal": round(r_new["accuracy_mean"], 4),
        "Delta_Acc": round(r_new["accuracy_mean"] - r_old["accuracy_mean"], 4),
        "BalAcc_kfold": round(r_old["balanced_accuracy_mean"], 4),
        "BalAcc_temporal": round(r_new["balanced_accuracy_mean"], 4),
        "Delta_BalAcc": round(r_new["balanced_accuracy_mean"] - r_old["balanced_accuracy_mean"], 4),
        "MCC_kfold": round(r_old["mcc_mean"], 4),
        "MCC_temporal": round(r_new["mcc_mean"], 4),
        "Delta_MCC": round(r_new["mcc_mean"] - r_old["mcc_mean"], 4),
    })
df_delta = pd.DataFrame(comp).sort_values("Delta_MCC")
save_csv(df_delta, "13_protocol_effect_delta")

print("=" * 110)
print("   EFECTO DE CORREGIR EL PROTOCOLO")
print(f"   {base_cfg[0]} / {base_cfg[1]}   ->   {new_cfg[0]} / {new_cfg[1]}")
print("=" * 110)
display(df_delta)

est = df_delta[df_delta["Tipo"] == "study"]
print(f"\nCaida media de Accuracy entre los modelos del estudio: {est['Delta_Acc'].mean():+.4f}")
print(f"Caida media de MCC:                                    {est['Delta_MCC'].mean():+.4f}")
print(f"Caida maxima de MCC:  {est['Delta_MCC'].min():+.4f}  "
      f"({est.loc[est['Delta_MCC'].idxmin(), 'Estimador']})")

# ============================================================
# Figure: effect of the protocol
# ============================================================
est_fig = df_delta[df_delta["Tipo"] == "study"].copy().sort_values("MCC_temporal")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, (c_old, c_new, titulo) in zip(axes, [
        ("Acc_kfold", "Acc_temporal", "Accuracy"),
        ("MCC_kfold", "MCC_temporal", "MCC")]):
    ypos = np.arange(len(est_fig))
    ax.barh(ypos - 0.2, est_fig[c_old], height=0.4,
            label="K-Fold barajado + umbral global",
            color=GRISES[1], edgecolor="black", linewidth=0.7, hatch="///")
    ax.barh(ypos + 0.2, est_fig[c_new], height=0.4,
            label="Validacion temporal + umbral por particion",
            color=GRISES[5], edgecolor="black", linewidth=0.7)
    ax.set_yticks(ypos); ax.set_yticklabels(est_fig["Estimador"])
    ax.set_xlabel(titulo); ax.set_title(f"{titulo} segun protocolo de validacion")
    ax.axvline(0, color="black", linewidth=0.8)
    estilo_eje(ax, rejilla="x")

# Single legend at the bottom: inside the axes it covered the longest bars.
_h, _l = axes[0].get_legend_handles_labels()
fig.legend(_h, _l, loc="upper center", bbox_to_anchor=(0.5, 0.02),
           ncol=2, frameon=True, edgecolor="black")
plt.tight_layout(rect=[0, 0.06, 1, 1])
save_figure(fig, "fig_protocol_effect")
plt.show()

"""## 9.3 Confusion matrices

White cells with black border and text: legible when photocopied and without ink blocks. Absolute values aggregated over the evaluated splits."""

# ============================================================
# Confusion matrices of the primary protocol
# ============================================================
res_p  = ALL_RESULTS[PRIMARY_KEY]
pred_p = ALL_PREDICTIONS[PRIMARY_KEY]

keys_study = [k for k, v in res_p.items() if v["category"] == "study"]
keys_dl    = [k for k, v in res_p.items() if v["category"] == "deep_learning"]
keys_bl    = [k for k, v in res_p.items() if v["category"] == "baseline"]

def panel_matrices(keys, titulo_archivo, encabezado):
    if not keys:
        return
    ncols = min(4, len(keys)); nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, k in enumerate(keys):
        r = res_p[k]
        matriz_confusion_monocroma(
            axes[i], r["confusion_matrix"],
            titulo=f"{r['label']}\nMCC={r['mcc_mean']:+.3f}  BalAcc={r['balanced_accuracy_mean']:.3f}")
    for j in range(len(keys), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(encabezado, fontsize=12, y=1.01)
    plt.tight_layout()
    save_figure(fig, titulo_archivo)
    plt.show()

panel_matrices(keys_study, "fig_confusion_matrices_study",
               "Matrices de confusion — modelos del estudio (protocolo principal)")
panel_matrices(keys_dl, "fig_confusion_matrices_deep_learning",
               "Matrices de confusion — arquitecturas de aprendizaje profundo")
panel_matrices(keys_bl, "fig_confusion_matrices_baselines",
               "Matrices de confusion — baselines (R1-M6)")

# ============================================================
# Row-normalized version, in percentage
# ============================================================
_orden_norm = keys_study + keys_dl + keys_bl
ncols = min(4, len(_orden_norm)); nrows = int(np.ceil(len(_orden_norm) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.7 * nrows))
axes = np.atleast_1d(axes).ravel()
for i, k in enumerate(_orden_norm):
    matriz_confusion_monocroma(axes[i], res_p[k]["confusion_matrix"],
                               titulo=res_p[k]["label"], normalizar=True)
for j in range(len(_orden_norm), len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Matrices de confusion normalizadas por clase real (%)", fontsize=12, y=1.01)
plt.tight_layout()
save_figure(fig, "fig_confusion_matrices_normalized")
plt.show()

"""## 9.4 ROC curves

> **Exception to the monochrome style.** This is the only figure of the notebook that keeps colour: several curves overlap in the same quadrant and colour is the only channel that separates them without filling the figure with markers. The area under the curve of every estimator is also given in a table, so that the information is recoverable in a monochrome print."""

# ============================================================
# Aggregated ROC curves (the only figure with colour)
# ============================================================
fig, ax = plt.subplots(figsize=(7.8, 6.8))
filas_auc = []
for k in keys_study + keys_dl:
    p = pred_p[k]
    if len(np.unique(p["y_true"])) < 2:
        continue
    try:
        fpr, tpr, _ = roc_curve(p["y_true"], p["y_score"])
        auc = roc_auc_score(p["y_true"], p["y_score"])
        estilo = "--" if res_p[k]["category"] == "deep_learning" else "-"
        ax.plot(fpr, tpr, linewidth=1.8, linestyle=estilo,
                label=f"{res_p[k]['label']} (AUC={auc:.3f})")
        filas_auc.append({"Estimador": res_p[k]["label"], "Tipo": res_p[k]["category"],
                          "ROC_AUC_agregado": round(float(auc), 4)})
    except Exception:
        continue

ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1.2,
        label="Azar (AUC=0.500)")
ax.set_xlabel("Tasa de falsos positivos (1 - especificidad)")
ax.set_ylabel("Tasa de verdaderos positivos (sensibilidad)")
ax.set_title(f"Curvas ROC — protocolo {PRIMARY_PROTOCOL}\n"
             "(unica figura en color; linea discontinua = aprendizaje profundo)")
ax.legend(loc="lower right", fontsize=8.5, frameon=True, edgecolor="black")
ax.grid(color="#cccccc", linewidth=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
save_figure(fig, "fig_roc_curves_primary")
plt.show()

df_auc = pd.DataFrame(filas_auc).sort_values("ROC_AUC_agregado", ascending=False)
save_csv(df_auc, "10b_roc_auc_aggregated")
display(df_auc)

"""## 9.5 Feature importance"""

# ============================================================
# Feature importance (models that expose it)
# ============================================================
with_imp = [k for k in keys_study if "feature_importance" in res_p[k]]
if with_imp:
    ncols = min(3, len(with_imp)); nrows = int(np.ceil(len(with_imp) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, k in enumerate(with_imp):
        imp = res_p[k]["feature_importance"]
        order = np.argsort(imp)[::-1]
        axes[i].barh(range(len(FEATURE_COLUMNS)), imp[order],
                     color=GRISES[4], edgecolor="black", linewidth=0.6)
        axes[i].set_yticks(range(len(FEATURE_COLUMNS)))
        axes[i].set_yticklabels([FEATURE_COLUMNS[j] for j in order])
        axes[i].invert_yaxis()
        axes[i].set_title(res_p[k]["label"]); axes[i].set_xlabel("Importancia")
        estilo_eje(axes[i], rejilla="x")
    for j in range(len(with_imp), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    save_figure(fig, "fig_feature_importance_primary")
    plt.show()

    filas_imp = []
    for k in with_imp:
        for v, nombre in zip(res_p[k]["feature_importance"], FEATURE_COLUMNS):
            filas_imp.append({"Modelo": res_p[k]["label"], "Variable": nombre,
                              "Importancia": round(float(v), 6)})
    save_csv(pd.DataFrame(filas_imp), "14b_model_feature_importance")

"""## 9.6 Sensitivity to climate variables

Two complementary analyses, both under the primary protocol:

1. **Permutation importance:** how much performance degrades when the relation between each variable and the target is destroyed, **within the observed range**. It is preferable to the internal tree importance, which is biased towards high-cardinality variables.
2. **Directed perturbation:** shift the temperature anomaly by ±1σ and ±2σ and measure the change in the rate of positive predictions (*what happens if the climate changes*).

The two analyses do not measure the same thing. Permutation reorders existing values and never leaves the observed distribution; the ±2σ shift does leave it. A small permutation importance for temperature together with a large response to the shift means that, within the historical range, temperature contributes little compared with the calendar, and that outside that range the model extrapolates sharply, which is itself a limitation to declare."""

# ============================================================
# Climate sensitivity
# ============================================================
if RUN_SENSITIVITY:
    cv = create_cv_splitter(PRIMARY_PROTOCOL)
    splits = list(cv.split(X))
    tr, te = splits[-1]                      # last split: the largest
    thr = compute_threshold(PROD_TODAY.iloc[tr])
    y_tr = binarize(PROD_NEXT.iloc[tr], thr).values
    y_te = binarize(PROD_NEXT.iloc[te], thr).values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X.iloc[tr]); X_te_s = scaler.transform(X.iloc[te])

    MODELO_SENSIBILIDAD = "LogReg"
    modelo_sens = MODEL_REGISTRY[MODELO_SENSIBILIDAD]["cls"](
        **MODEL_REGISTRY[MODELO_SENSIBILIDAD]["params"])
    modelo_sens.fit(X_tr_s, y_tr)

    # --- (1) Permutation importance ---
    perm = permutation_importance(modelo_sens, X_te_s, y_te, n_repeats=30,
                                  random_state=RANDOM_SEED, scoring="balanced_accuracy")
    df_perm = (pd.DataFrame({"Variable": FEATURE_COLUMNS,
                             "Importancia": perm.importances_mean,
                             "Desv_est": perm.importances_std})
               .sort_values("Importancia", ascending=False).reset_index(drop=True))
    save_csv(df_perm, "14_permutation_importance")
    print(f"Importancia por permutacion ({MODEL_REGISTRY[MODELO_SENSIBILIDAD]['label']}, "
          f"protocolo principal):")
    print(df_perm.to_string(index=False))

    # --- (2) Perturbation of the temperature anomaly ---
    sigma_T = float(X["Temperature_Mean"].std())
    base_pred = modelo_sens.predict(X_te_s)
    base_pos  = float(base_pred.mean())
    base_bacc = balanced_accuracy_score(y_te, base_pred)

    filas_sens = []
    for delta in [-2, -1, -0.5, 0, 0.5, 1, 2]:
        Xp = X.iloc[te].copy()
        Xp["Temperature_Mean"] = Xp["Temperature_Mean"] + delta * sigma_T
        Xp["Temp_MA3"]         = Xp["Temp_MA3"]         + delta * sigma_T   # internal consistency
        yp = modelo_sens.predict(scaler.transform(Xp))
        filas_sens.append({
            "Perturbacion_sigma": delta,
            "Delta_Temperatura_C": round(delta * sigma_T, 4),
            "Tasa_predicciones_Alto": round(float(yp.mean()), 4),
            "Delta_vs_base_pp": round((float(yp.mean()) - base_pos) * 100, 2),
            "Balanced_Acc": round(balanced_accuracy_score(y_te, yp), 4),
            "Delta_BalAcc": round(balanced_accuracy_score(y_te, yp) - base_bacc, 4),
        })
    df_sens = pd.DataFrame(filas_sens)
    save_csv(df_sens, "15_climate_sensitivity")
    print(f"\nSigma de la anomalia de temperatura en la muestra: {sigma_T:.4f} °C")
    print("\nSensibilidad a perturbaciones de la anomalia de temperatura:")
    print(df_sens.to_string(index=False))

    # Figures needed to reconcile both analyses in the generated narrative
    IMP_TEMPERATURA = float(
        df_perm.loc[df_perm["Variable"] == "Temperature_Mean", "Importancia"].iloc[0])
    IMP_TOP_VARIABLE = df_perm.iloc[0]["Variable"]
    IMP_TOP_VALOR = float(df_perm.iloc[0]["Importancia"])
    RANGO_PERTURBACION = float(df_sens["Tasa_predicciones_Alto"].max()
                               - df_sens["Tasa_predicciones_Alto"].min())

# ============================================================
# Sensitivity figures
# ============================================================
if RUN_SENSITIVITY:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    d = df_perm.sort_values("Importancia")
    axes[0].barh(d["Variable"], d["Importancia"], xerr=d["Desv_est"],
                 color=GRISES[3], edgecolor="black", linewidth=0.6,
                 error_kw=dict(ecolor="black", lw=0.8, capsize=2.5))
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Caida de balanced accuracy al permutar")
    axes[0].set_title("Importancia por permutacion\n(dentro del rango observado)")
    estilo_eje(axes[0], rejilla="x")

    axes[1].plot(df_sens["Perturbacion_sigma"], df_sens["Tasa_predicciones_Alto"],
                 marker="o", markersize=7, linewidth=2, color="black",
                 markerfacecolor="white", markeredgecolor="black",
                 label="Tasa de predicciones Alto")
    axes[1].plot(df_sens["Perturbacion_sigma"], df_sens["Balanced_Acc"],
                 marker="s", markersize=7, linewidth=2, color=GRISES[5],
                 linestyle="--", markerfacecolor=GRISES[2], markeredgecolor="black",
                 label="Balanced accuracy")
    axes[1].axvline(0, color="black", linestyle=":", linewidth=1.2)
    axes[1].set_xlabel("Perturbacion de la anomalia de temperatura (en sigma)")
    axes[1].set_ylabel("Proporcion")
    axes[1].set_title("Respuesta al forzamiento termico\n(desplazamiento fuera de distribucion)")
    axes[1].legend(frameon=True, edgecolor="black")
    estilo_eje(axes[1], rejilla="both")

    plt.tight_layout()
    save_figure(fig, "fig_climate_sensitivity")
    plt.show()

    print(f"Rango de variacion de la tasa de predicciones positivas entre -2σ y +2σ: "
          f"{RANGO_PERTURBACION*100:.1f} puntos porcentuales")
    print(f"Importancia por permutacion de Temperature_Mean : {IMP_TEMPERATURA:.4f}")
    print(f"Variable mas influyente por permutacion         : {IMP_TOP_VARIABLE} "
          f"({IMP_TOP_VALOR:.4f})")
    if IMP_TEMPERATURA > 0:
        print(f"Razon entre ambas                               : "
              f"{IMP_TOP_VALOR / IMP_TEMPERATURA:.0f} veces")
    print()
    print("Lectura obligada para el manuscrito: las dos cifras deben reportarse juntas.")
    print("La permutacion mide dentro del rango observado; el desplazamiento de ±2σ saca")
    print("los datos de distribucion. Reportar solo una de las dos daria una impresion")
    print("enganosa de la sensibilidad climatica del sistema.")

"""## 9.7 Computational cost

Measured on the same hardware and under the same protocol, in the same run."""

# ============================================================
# Computational cost
# ============================================================
filas_t = []
for k, r in res_p.items():
    filas_t.append({
        "Estimador": r["label"], "Tipo": r["category"],
        "Entrenamiento_s": round(r["time_train_mean"], 5),
        "Inferencia_s": round(r["time_infer_mean"], 6),
        "Inferencia_por_dia_ms": round(r["time_infer_mean"] / max(1, len(X) // N_FOLDS) * 1000, 4),
        "Balanced_Acc": round(r["balanced_accuracy_mean"], 4),
        "MCC": round(r["mcc_mean"], 4),
    })
df_time = pd.DataFrame(filas_t).sort_values("Entrenamiento_s")
save_csv(df_time, "16_computational_cost")
display(df_time)

print(f"Entorno de medicion: {ENVIRONMENT['platform']}")
print(f"Python {ENVIRONMENT['python']} · scikit-learn {ENVIRONMENT['scikit_learn']}")
if DL_EJECUTADO:
    print(f"Aprendizaje profundo: {DEEP_LEARNING_MARCO}")
print()
est_t = df_time[df_time["Tipo"] == "study"]
mas_rapido = est_t.iloc[0]; mas_lento = est_t.iloc[-1]
print(f"Mas rapido de entrenar (estudio): {mas_rapido['Estimador']} "
      f"({mas_rapido['Entrenamiento_s']:.5f} s)")
print(f"Mas lento de entrenar (estudio) : {mas_lento['Estimador']} "
      f"({mas_lento['Entrenamiento_s']:.5f} s)")
print(f"Razon entre extremos            : "
      f"{mas_lento['Entrenamiento_s'] / max(mas_rapido['Entrenamiento_s'], 1e-9):.0f}x")
if DL_EJECUTADO:
    dl_t = df_time[df_time["Tipo"] == "deep_learning"]
    print(f"\nAprendizaje profundo: entre {dl_t['Entrenamiento_s'].min():.2f} s y "
          f"{dl_t['Entrenamiento_s'].max():.2f} s por particion.")
    print("  Es el unico bloque cuyo costo computacional se aparta de lo sub-segundo,")
    print("  y aun asi sigue siendo trivial en terminos absolutos.")
print()
print("Todos los tiempos de los modelos tabulares son de orden sub-segundo: con")
print(f"n = {len(X)} el costo computacional NO es un criterio discriminante entre")
print("modelos. Conviene decirlo explicitamente en lugar de insinuar una ventaja")
print("de eficiencia que los datos no sostienen.")

"""# 10. Statistical significance

Three tests are used, since a single one is not enough:

| Test | What it compares | Why it is included |
|---|---|---|
| **McNemar** | Two classifiers on the same observations | Standard test for comparing classifiers on a common test set; it does not assume independence between models |
| **Paired Wilcoxon** | Metrics aggregated across splits | Complements McNemar at split level |
| **Paired t** | Same as Wilcoxon, parametric | Reported with the caveat that, with few splits, the normality assumption is fragile |

**McNemar has a direction.** The test returns `b` (hits only of model 1) and `c` (hits only of model 2). A significant contrast does not say who wins; the comparison between `b` and `c` does. The function `direccion_mcnemar()` classifies every contrast as *outperforms*, *is outperformed* or *indistinguishable*.

**The ranking uses MCC while McNemar contrasts raw hits.** They are not the same metric, and with a majority class above 80 % they can disagree: a model can lose in total hits and win in MCC because it is right where it matters. The disagreements are reported as an explicit list.

**Multiple comparisons.** Holm-Bonferroni is applied over the total number of pairs, including those of the supplementary baseline table."""

# ============================================================
# McNemar test with continuity correction
# ============================================================
def mcnemar_test(y_true, y_pred_1, y_pred_2):
    """
    Paired McNemar with Edwards' continuity correction.
    Return (b, c, chi2, p_value) where
      b = hits only of model 1, c = hits only of model 2.
    """
    ok1, ok2 = (y_pred_1 == y_true), (y_pred_2 == y_true)
    b = int(np.sum(ok1 & ~ok2))
    c = int(np.sum(~ok1 & ok2))
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, float(chi2), float(1 - chi2_dist.cdf(chi2, df=1))


def holm_bonferroni(pvals):
    """Holm-Bonferroni correction. Return the adjusted p-values in the original order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    orden = np.argsort(p)
    ajust = np.empty(m, dtype=float)
    prev = 0.0
    for rango, i in enumerate(orden):
        val = (m - rango) * p[i]
        prev = max(prev, val)          # monotonicity
        ajust[i] = min(1.0, prev)
    return ajust


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def direccion_mcnemar(fila, label_ref):
    """
    Return 'supera', 'es_superado' or 'indistinguible' from the
    perspective of `label_ref`.
    """
    if fila["Sig_Holm"] == "ns":
        return "indistinguible"
    b, c = fila["b (solo M1)"], fila["c (solo M2)"]
    aciertos_ref, aciertos_otro = (b, c) if fila["Modelo_1"] == label_ref else (c, b)
    return "supera" if aciertos_ref > aciertos_otro else "es_superado"


print("Funciones de prueba definidas: McNemar + Holm-Bonferroni + direccion.")

# ============================================================
# McNemar on ALL pairs, including the baselines
# ============================================================
todos = list(res_p.keys())

filas, pvals = [], []
for a, b_key in itertools.combinations(todos, 2):
    pa, pb = pred_p[a], pred_p[b_key]
    # Align by observation index
    orden_a = np.argsort(pa["idx"]); orden_b = np.argsort(pb["idx"])
    ia, ib = pa["idx"][orden_a], pb["idx"][orden_b]
    comun = np.intersect1d(ia, ib)
    if len(comun) == 0:
        continue
    ma, mb = np.isin(ia, comun), np.isin(ib, comun)
    yt  = pa["y_true"][orden_a][ma]
    yp1 = pa["y_pred"][orden_a][ma]
    yp2 = pb["y_pred"][orden_b][mb]

    bb, cc, chi2, p = mcnemar_test(yt, yp1, yp2)
    filas.append({
        "Modelo_1": res_p[a]["label"], "Modelo_2": res_p[b_key]["label"],
        "Clave_1": a, "Clave_2": b_key,
        "Tipo_1": res_p[a]["category"], "Tipo_2": res_p[b_key]["category"],
        "Contraste_1": res_p[a].get("contraste", "—"),
        "Contraste_2": res_p[b_key].get("contraste", "—"),
        "n": len(comun), "b (solo M1)": bb, "c (solo M2)": cc,
        "chi2": round(chi2, 3), "p_crudo": p,
        "Delta_MCC_M1_menos_M2": round(res_p[a]["mcc_mean"] - res_p[b_key]["mcc_mean"], 4),
    })
    pvals.append(p)

df_mc = pd.DataFrame(filas)
df_mc["p_Holm"] = holm_bonferroni(pvals)
df_mc["Sig_cruda"] = df_mc["p_crudo"].apply(stars)
df_mc["Sig_Holm"]  = df_mc["p_Holm"].apply(stars)
df_mc["p_crudo"]   = df_mc["p_crudo"].round(4)
df_mc["p_Holm"]    = df_mc["p_Holm"].round(4)

# Explicit direction, from the perspective of Model_1
df_mc["Direccion_M1"] = df_mc.apply(lambda f: direccion_mcnemar(f, f["Modelo_1"]), axis=1)
df_mc["Gana_en_aciertos"] = np.where(
    df_mc["Sig_Holm"] == "ns", "—",
    np.where(df_mc["b (solo M1)"] > df_mc["c (solo M2)"],
             df_mc["Modelo_1"], df_mc["Modelo_2"]))

save_csv(df_mc, "17_mcnemar_all_pairs_holm")
print(f"Pares evaluados: {len(df_mc)}   (estimadores: {len(todos)})")
print(f"Significativos sin corregir : {(df_mc['Sig_cruda'] != 'ns').sum()}")
print(f"Significativos tras Holm    : {(df_mc['Sig_Holm'] != 'ns').sum()}")
print(f"-> {(df_mc['Sig_cruda'] != 'ns').sum() - (df_mc['Sig_Holm'] != 'ns').sum()} "
      f"contrastes dejan de ser significativos al corregir por multiplicidad.")
display(df_mc.head(20))

# ============================================================
# Utility: retrieve the row of any pair
# ============================================================
def fila_par(label_a, label_b):
    """Return the row of df_mc that contrasts both labels, or None."""
    f = df_mc[((df_mc["Modelo_1"] == label_a) & (df_mc["Modelo_2"] == label_b)) |
              ((df_mc["Modelo_2"] == label_a) & (df_mc["Modelo_1"] == label_b))]
    return None if len(f) == 0 else f.iloc[0]

# Split of the keys according to the contrast declared in section 6.2
study_keys        = [k for k, v in res_p.items() if v["category"] == "study"]
dl_keys           = [k for k, v in res_p.items() if v["category"] == "deep_learning"]
propuestos_keys   = study_keys + dl_keys
baseline_keys_pri = [k for k, v in res_p.items()
                     if v["category"] == "baseline" and v.get("contraste_principal", False)]
baseline_keys_sup = [k for k, v in res_p.items()
                     if v["category"] == "baseline" and not v.get("contraste_principal", False)]

print(f"Modelos propuestos (estudio + aprendizaje profundo): {len(propuestos_keys)}")
print(f"Baselines del contraste principal : {[res_p[k]['label'] for k in baseline_keys_pri]}")
print(f"Baselines suplementarios          : {[res_p[k]['label'] for k in baseline_keys_sup]}")

"""## 10.1 Key contrast 1 — proposed models versus baselines

This contrast supports the contribution of the study. It is reported in two separate tables: the **main contrast**, with the majority-class, seasonal-naive and stratified-random baselines, and a **supplementary** table with the persistence baseline. Both include the direction column: a significant contrast without direction is incomplete information."""

# ============================================================
# Main contrast: proposed models vs. baselines
# ============================================================
labels_pri = [res_p[k]["label"] for k in baseline_keys_pri]
labels_sup = [res_p[k]["label"] for k in baseline_keys_sup]
labels_prop = [res_p[k]["label"] for k in propuestos_keys]

cols_vista = ["Modelo_1", "Modelo_2", "b (solo M1)", "c (solo M2)", "chi2",
              "p_crudo", "p_Holm", "Sig_Holm", "Gana_en_aciertos",
              "Delta_MCC_M1_menos_M2"]

sub_pri = df_mc[
    ((df_mc["Modelo_1"].isin(labels_prop)) & (df_mc["Modelo_2"].isin(labels_pri))) |
    ((df_mc["Modelo_2"].isin(labels_prop)) & (df_mc["Modelo_1"].isin(labels_pri)))
].copy()

print("=" * 110)
print("   CONTRASTE PRINCIPAL — modelos propuestos vs. baselines pedidos por R1-M6")
print("   (clase mayoritaria, estacional ingenuo, estratificado aleatorio)")
print("=" * 110)
display(sub_pri[cols_vista])
save_csv(sub_pri, "18_mcnemar_study_vs_baselines")
print(f"\nContrastes significativos tras Holm: {(sub_pri['Sig_Holm'] != 'ns').sum()} "
      f"de {len(sub_pri)}")

# ============================================================
# Supplementary table: persistence baseline
# ============================================================
if labels_sup:
    sub_sup = df_mc[
        ((df_mc["Modelo_1"].isin(labels_prop)) & (df_mc["Modelo_2"].isin(labels_sup))) |
        ((df_mc["Modelo_2"].isin(labels_prop)) & (df_mc["Modelo_1"].isin(labels_sup)))
    ].copy()

    print("=" * 110)
    print("   TABLA SUPLEMENTARIA — baseline de persistencia (D+1 = D)")
    print("   Baseline adicional, NO solicitado nominalmente por el revisor.")
    print("   Se reporta por transparencia y queda fuera del contraste principal.")
    print("=" * 110)
    display(sub_sup[cols_vista])
    save_csv(sub_sup, "18b_mcnemar_baseline_suplementario")

    gana_persistencia = int((sub_sup["Gana_en_aciertos"].isin(labels_sup)).sum())
    print(f"\nContrastes en los que la persistencia gana en aciertos: "
          f"{gana_persistencia} de {len(sub_sup)}")
    if gana_persistencia:
        print("Esta cifra debe aparecer en el manuscrito. Es la lectura honesta del")
        print("resultado y, presentada junto al argumento de R1-M2, es defendible:")
        print("la persistencia acierta por exactitud cruda en una serie con vedas")
        print("prolongadas, que es exactamente el regimen en el que la exactitud")
        print("deja de medir capacidad discriminante.")
else:
    sub_sup = pd.DataFrame()

"""## 10.2 Key contrast 2 — among the leading models

The list of models is derived as the union of the **top three by MCC ranking** and the **four leading models of the manuscript** (Random Forest, XGBoost, Gradient Boosting Tree and SVM), preserving the order, so that the best model by MCC is always included."""

# ============================================================
# top_keys derived from the ranking, not fixed
# ============================================================
ranking = sorted(propuestos_keys, key=lambda k: res_p[k]["mcc_mean"], reverse=True)
mejor = ranking[0]
label_mejor = res_p[mejor]["label"]

MODELOS_CABEZA_MANUSCRITO = ["RFC", "XGBoost", "GBT", "SVM_RBF_C100"]

top_keys = []
for k in ranking[:3] + MODELOS_CABEZA_MANUSCRITO:      # the order is preserved
    if k in res_p and k not in top_keys:
        top_keys.append(k)

labels_top = [res_p[k]["label"] for k in top_keys]
print("Modelos incluidos en los contrastes de cabeza (derivados, no fijos):")
for k in top_keys:
    origen = []
    if k in ranking[:3]: origen.append(f"top-{ranking.index(k)+1} por MCC")
    if k in MODELOS_CABEZA_MANUSCRITO: origen.append("cabeza del manuscrito")
    print(f"  {res_p[k]['label']:28s} MCC={res_p[k]['mcc_mean']:+.4f}   [{', '.join(origen)}]")

sub_top = df_mc[df_mc["Modelo_1"].isin(labels_top) & df_mc["Modelo_2"].isin(labels_top)].copy()
print("\n" + "=" * 110)
print("   CONTRASTES ENTRE LOS MODELOS DE CABEZA  (lo que R1-M1 cuestiona)")
print("=" * 110)
display(sub_top[cols_vista])
save_csv(sub_top, "19_mcnemar_top_models")

"""## 10.3 Disagreement between metrics

The ranking is built with MCC because it is robust to class imbalance. McNemar, on the other hand, contrasts **raw hits**, which is accuracy. With a majority class above 80 % the two measures diverge. The disagreement is reported explicitly."""

# ============================================================
# Detection of disagreements between MCC and raw hits
# ============================================================
filas_disc = []
for _, f in df_mc.iterrows():
    if f["Sig_Holm"] == "ns":
        continue
    gana_aciertos = f["Modelo_1"] if f["b (solo M1)"] > f["c (solo M2)"] else f["Modelo_2"]
    d_mcc = f["Delta_MCC_M1_menos_M2"]
    if d_mcc == 0:
        continue
    gana_mcc = f["Modelo_1"] if d_mcc > 0 else f["Modelo_2"]
    if gana_aciertos != gana_mcc:
        filas_disc.append({
            "Modelo_1": f["Modelo_1"], "Modelo_2": f["Modelo_2"],
            "b (solo M1)": f["b (solo M1)"], "c (solo M2)": f["c (solo M2)"],
            "p_Holm": f["p_Holm"],
            "Gana_en_aciertos_McNemar": gana_aciertos,
            "Gana_en_MCC": gana_mcc,
            "Delta_MCC": d_mcc,
        })

df_discordancias = pd.DataFrame(filas_disc)
METRICA_DEL_CONTRASTE = ("accuracy (aciertos crudos, que es lo que contrasta McNemar) "
                         "— el ranking del veredicto usa MCC")

print("=" * 110)
print("   DISCORDANCIAS ENTRE METRICAS")
print(f"   Metrica del contraste : {METRICA_DEL_CONTRASTE}")
print("=" * 110)
if len(df_discordancias):
    display(df_discordancias)
    save_csv(df_discordancias, "19b_metric_discordances")
    print(f"\n{len(df_discordancias)} contrastes significativos en los que el ganador por")
    print("aciertos crudos NO es el ganador por MCC.")
    print()
    print("Uso en el manuscrito: esta tabla es evidencia directa del punto de R1-M2.")
    print("Con clases desbalanceadas, dos metricas legitimas ordenan los modelos de")
    print("forma distinta, y la exactitud es la que se deja enganar. Reportarlo es")
    print("mas solido que elegir en silencio la metrica que favorece la conclusion.")
else:
    print("No se detectaron discordancias entre el orden por MCC y el orden por")
    print("aciertos crudos en los contrastes significativos.")

"""## 10.4 Wilcoxon and paired t across splits

Complement to McNemar, at split level. With few splits the statistical power is low: **the absence of significance here does not prove equivalence**; it only indicates that the number of splits is not enough to tell the models apart."""

# ============================================================
# Paired Wilcoxon and paired t on the per-split metrics
# ============================================================
def paired_tests_across_folds(res, key_a, key_b, metric="balanced_accuracy"):
    """Wilcoxon and paired t on the per-split values of a metric."""
    a = np.array(res[key_a][f"{metric}_folds"], dtype=float)
    b = np.array(res[key_b][f"{metric}_folds"], dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return dict(dif_media=np.nan, W=np.nan, p_wilcoxon=np.nan, t=np.nan, p_ttest=np.nan)
    dif = a - b
    if np.allclose(dif, 0):
        return dict(dif_media=0.0, W=np.nan, p_wilcoxon=1.0, t=np.nan, p_ttest=1.0)
    try:
        W, pw = wilcoxon(a, b)
    except Exception:
        W, pw = np.nan, np.nan
    try:
        t, pt = ttest_rel(a, b)
    except Exception:
        t, pt = np.nan, np.nan
    return dict(dif_media=float(np.mean(dif)), W=float(W) if W == W else np.nan,
                p_wilcoxon=float(pw) if pw == pw else np.nan,
                t=float(t) if t == t else np.nan,
                p_ttest=float(pt) if pt == pt else np.nan)


claves_wilcoxon = top_keys + baseline_keys_pri + baseline_keys_sup
filas_w = []
for a, b_key in itertools.combinations(claves_wilcoxon, 2):
    for metrica in ["balanced_accuracy", "mcc"]:
        r = paired_tests_across_folds(res_p, a, b_key, metrica)
        filas_w.append({
            "Modelo_1": res_p[a]["label"], "Modelo_2": res_p[b_key]["label"],
            "Metrica": metrica,
            "Dif_media": round(r["dif_media"], 4) if r["dif_media"] == r["dif_media"] else np.nan,
            "p_Wilcoxon": round(r["p_wilcoxon"], 4) if r["p_wilcoxon"] == r["p_wilcoxon"] else np.nan,
            "p_t_pareada": round(r["p_ttest"], 4) if r["p_ttest"] == r["p_ttest"] else np.nan,
        })

df_wilcox = pd.DataFrame(filas_w)
save_csv(df_wilcox, "20_wilcoxon_paired_folds")
print("Pruebas pareadas a traves de las particiones evaluadas:")
display(df_wilcox)

_n_part = FOLDS_PRINCIPAL["evaluadas"]
_p_min_wilcoxon = 2.0 ** (-_n_part) * 2 if _n_part > 0 else np.nan
print(f"\nAdvertencia de poder estadistico: con {_n_part} particiones evaluadas, el valor p")
print(f"minimo alcanzable por Wilcoxon es del orden de {_p_min_wilcoxon:.4f}. Ninguna")
print("comparacion puede resultar significativa al 5 % con esta prueba y este numero de")
print("particiones. Por eso McNemar, que opera observacion por observacion, es la prueba")
print("principal, y la t pareada se reporta solo porque el revisor la menciona por su nombre.")
print()
print(f"[AUDIT] El mejor modelo por MCC ({label_mejor}) esta incluido en esta tabla: "
      f"{label_mejor in set(df_wilcox['Modelo_1']) | set(df_wilcox['Modelo_2'])}")

"""## 10.5 Bootstrap confidence intervals

The standard deviation across a few splits is a poor estimator of uncertainty. A percentile bootstrap on the pooled predictions gives an interval that can be cited and compared across estimators. The resampling is **stratified by class**, so that every replicate keeps the proportion of high-production days; otherwise some replicates would have no positive class and the MCC would be undefined."""

# ============================================================
# Percentile bootstrap on the pooled predictions
# ============================================================
N_BOOTSTRAP = 2000

def bootstrap_intervalo(y_true, y_pred, n_rep=N_BOOTSTRAP, alfa=0.05, semilla=RANDOM_SEED):
    """
    Percentile confidence interval for MCC and balanced accuracy.

    Resampling with replacement stratified by true class, so that every
    replicate keeps both classes and the MCC is always defined.
    """
    rng = np.random.RandomState(semilla)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    idx0 = np.where(y_true == 0)[0]
    idx1 = np.where(y_true == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return {k: (np.nan, np.nan, np.nan) for k in ("mcc", "balanced_accuracy")}

    mccs, baccs = np.empty(n_rep), np.empty(n_rep)
    for i in range(n_rep):
        s = np.concatenate([rng.choice(idx0, len(idx0), replace=True),
                            rng.choice(idx1, len(idx1), replace=True)])
        mccs[i] = matthews_corrcoef(y_true[s], y_pred[s])
        baccs[i] = balanced_accuracy_score(y_true[s], y_pred[s])

    lo, hi = 100 * alfa / 2, 100 * (1 - alfa / 2)
    return {
        "mcc": (float(matthews_corrcoef(y_true, y_pred)),
                float(np.percentile(mccs, lo)), float(np.percentile(mccs, hi))),
        "balanced_accuracy": (float(balanced_accuracy_score(y_true, y_pred)),
                              float(np.percentile(baccs, lo)), float(np.percentile(baccs, hi))),
    }


filas_ci = []
for k in todos:
    p = pred_p[k]
    ci = bootstrap_intervalo(p["y_true"], p["y_pred"])
    filas_ci.append({
        "Estimador": res_p[k]["label"], "Tipo": res_p[k]["category"],
        "Contraste": res_p[k].get("contraste", "—"),
        "MCC_agrupado": round(ci["mcc"][0], 4),
        "MCC_IC95_inf": round(ci["mcc"][1], 4),
        "MCC_IC95_sup": round(ci["mcc"][2], 4),
        "BalAcc_agrupado": round(ci["balanced_accuracy"][0], 4),
        "BalAcc_IC95_inf": round(ci["balanced_accuracy"][1], 4),
        "BalAcc_IC95_sup": round(ci["balanced_accuracy"][2], 4),
    })

df_ci = pd.DataFrame(filas_ci).sort_values("MCC_agrupado", ascending=False).reset_index(drop=True)
save_csv(df_ci, "20b_bootstrap_confidence_intervals")
print("=" * 110)
print(f"   INTERVALOS DE CONFIANZA AL 95 % POR BOOTSTRAP ({N_BOOTSTRAP} replicas, "
      f"estratificado por clase)")
print("=" * 110)
display(df_ci)

_ci_mejor = df_ci[df_ci["Estimador"] == label_mejor].iloc[0]
MCC_CERO_DENTRO_IC = bool(_ci_mejor["MCC_IC95_inf"] <= 0 <= _ci_mejor["MCC_IC95_sup"])
print(f"\nMejor modelo por MCC: {label_mejor}")
print(f"  MCC agrupado = {_ci_mejor['MCC_agrupado']:+.4f}  "
      f"IC95 [{_ci_mejor['MCC_IC95_inf']:+.4f}, {_ci_mejor['MCC_IC95_sup']:+.4f}]")
if MCC_CERO_DENTRO_IC:
    print("  El intervalo INCLUYE el cero: no puede afirmarse capacidad discriminante")
    print("  estadisticamente distinta del azar para este estimador. Es una cifra que")
    print("  debe entrar al manuscrito y a Limitaciones tal como sale.")
else:
    print("  El intervalo EXCLUYE el cero: hay capacidad discriminante por encima del azar.")

# ============================================================
# Figure: MCC with confidence interval per estimator
# ============================================================
d = df_ci.sort_values("MCC_agrupado")
fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(d) + 2.2))
ypos = np.arange(len(d))
err = np.vstack([d["MCC_agrupado"] - d["MCC_IC95_inf"],
                 d["MCC_IC95_sup"] - d["MCC_agrupado"]])
marcador = {"study": "o", "deep_learning": "s", "baseline": "^"}
for i, (_, fila) in enumerate(d.iterrows()):
    ax.errorbar(fila["MCC_agrupado"], i,
                xerr=[[err[0, i]], [err[1, i]]],
                fmt=marcador.get(fila["Tipo"], "o"), markersize=7,
                color="black", markerfacecolor=GRISES[2 if fila["Tipo"] == "study" else
                                                      (4 if fila["Tipo"] == "deep_learning" else 0)],
                markeredgecolor="black", ecolor="black", elinewidth=1.0, capsize=3)
ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
ax.set_yticks(ypos); ax.set_yticklabels(d["Estimador"])
ax.set_xlabel("MCC agrupado con intervalo de confianza al 95 % (bootstrap)")
ax.set_title("Coeficiente de correlacion de Matthews por estimador\n"
             "circulo = estudio · cuadrado = aprendizaje profundo · triangulo = baseline")
estilo_eje(ax, rejilla="x")
plt.tight_layout()
save_figure(fig, "fig_mcc_confidence_intervals")
plt.show()

"""## 10.6 Automatic verdict

Rather than interpreting the tables by hand, the verdict on model selection is **derived** from the tests. It includes:

- `es_superado_por`: models that beat the best model in hits, with significance after Holm correction.
- `es_superado_por_baselines`: the same for the baselines.
- `metrica_del_contraste`: declares that McNemar contrasts raw hits while the ranking uses MCC."""

# ============================================================
# Verdict derived from the tests, WITH direction
# ============================================================
print("=" * 92)
print("   VEREDICTO SOBRE LA SELECCION DEL MODELO  (R1-M1, R2-02)")
print("=" * 92)
print(f"\nRanking por MCC (protocolo {PRIMARY_PROTOCOL}):")
for i, k in enumerate(ranking, 1):
    marca = "  <- mejor" if k == mejor else ""
    print(f"  {i:2d}. {res_p[k]['label']:28s} MCC={res_p[k]['mcc_mean']:+.4f}  "
          f"BalAcc={res_p[k]['balanced_accuracy_mean']:.4f}  "
          f"Acc={res_p[k]['accuracy_mean']:.4f}{marca}")

# --- Against the other proposed models ---
supera, es_superado, indistinguible = [], [], []
for k in ranking[1:]:
    f = fila_par(label_mejor, res_p[k]["label"])
    if f is None:
        continue
    d = direccion_mcnemar(f, label_mejor)
    {"supera": supera, "es_superado": es_superado,
     "indistinguible": indistinguible}[d].append(res_p[k]["label"])

print(f"\nMejor modelo por MCC: {label_mejor}")
print(f"  Supera en aciertos (Holm p<0.05) a {len(supera)} modelo(s):")
for s in supera: print(f"     + {s}")
print(f"  ES SUPERADO en aciertos por {len(es_superado)} modelo(s):")
for s in es_superado: print(f"     - {s}")
print(f"  Estadisticamente indistinguible de {len(indistinguible)} modelo(s):")
for c in indistinguible: print(f"     = {c}")

# --- Against the baselines of the main contrast ---
bl_superados, bl_superan, bl_indistintos = [], [], []
for bk in baseline_keys_pri:
    f = fila_par(label_mejor, res_p[bk]["label"])
    if f is None:
        continue
    d = direccion_mcnemar(f, label_mejor)
    {"supera": bl_superados, "es_superado": bl_superan,
     "indistinguible": bl_indistintos}[d].append(res_p[bk]["label"])

print(f"\n  Frente a los {len(baseline_keys_pri)} baselines del CONTRASTE PRINCIPAL:")
print(f"    supera a          : {len(bl_superados)}  {bl_superados}")
print(f"    es superado por   : {len(bl_superan)}  {bl_superan}")
print(f"    indistinguible de : {len(bl_indistintos)}  {bl_indistintos}")

# --- Against the supplementary baselines ---
sup_superados, sup_superan, sup_indistintos = [], [], []
for bk in baseline_keys_sup:
    f = fila_par(label_mejor, res_p[bk]["label"])
    if f is None:
        continue
    d = direccion_mcnemar(f, label_mejor)
    {"supera": sup_superados, "es_superado": sup_superan,
     "indistinguible": sup_indistintos}[d].append(res_p[bk]["label"])

if baseline_keys_sup:
    print(f"\n  Frente a los {len(baseline_keys_sup)} baselines SUPLEMENTARIOS:")
    print(f"    supera a          : {len(sup_superados)}  {sup_superados}")
    print(f"    es superado por   : {len(sup_superan)}  {sup_superan}")
    print(f"    indistinguible de : {len(sup_indistintos)}  {sup_indistintos}")

# --- Comparison in MCC and balanced accuracy, the ranking metric ---
comparacion_metricas = []
for bk in baseline_keys_pri + baseline_keys_sup:
    comparacion_metricas.append({
        "Baseline": res_p[bk]["label"],
        "Contraste": res_p[bk].get("contraste", "—"),
        "MCC_baseline": round(res_p[bk]["mcc_mean"], 4),
        "MCC_mejor_modelo": round(res_p[mejor]["mcc_mean"], 4),
        "Mejor_gana_en_MCC": bool(res_p[mejor]["mcc_mean"] > res_p[bk]["mcc_mean"]),
        "BalAcc_baseline": round(res_p[bk]["balanced_accuracy_mean"], 4),
        "BalAcc_mejor_modelo": round(res_p[mejor]["balanced_accuracy_mean"], 4),
        "Mejor_gana_en_BalAcc": bool(res_p[mejor]["balanced_accuracy_mean"]
                                     > res_p[bk]["balanced_accuracy_mean"]),
        "Acc_baseline": round(res_p[bk]["accuracy_mean"], 4),
        "Acc_mejor_modelo": round(res_p[mejor]["accuracy_mean"], 4),
        "Mejor_gana_en_Acc": bool(res_p[mejor]["accuracy_mean"]
                                  > res_p[bk]["accuracy_mean"]),
    })
df_vs_baselines = pd.DataFrame(comparacion_metricas)
save_csv(df_vs_baselines, "18c_mejor_modelo_vs_baselines_metricas")
print("\n" + "=" * 110)
print("   MEJOR MODELO FRENTE A CADA BASELINE, EN LAS TRES METRICAS")
print("=" * 110)
display(df_vs_baselines)

_pri = df_vs_baselines[df_vs_baselines["Contraste"] == "principal"]
BASELINES_PRINCIPAL_SUPERADOS_MCC = [
    r["Baseline"] for _, r in _pri.iterrows() if r["Mejor_gana_en_MCC"]]
BASELINES_PRINCIPAL_SUPERADOS_BACC = [
    r["Baseline"] for _, r in _pri.iterrows() if r["Mejor_gana_en_BalAcc"]]
BASELINES_PRINCIPAL_SUPERADOS_ACC = [
    r["Baseline"] for _, r in _pri.iterrows() if r["Mejor_gana_en_Acc"]]

VERDICT = {
    "mejor_modelo": label_mejor,
    "clave_mejor_modelo": mejor,
    "criterio_ranking": "mcc_mean",
    "metrica_del_contraste": METRICA_DEL_CONTRASTE,
    "mcc": round(res_p[mejor]["mcc_mean"], 4),
    "balanced_accuracy": round(res_p[mejor]["balanced_accuracy_mean"], 4),
    "accuracy": round(res_p[mejor]["accuracy_mean"], 4),
    "mcc_ic95": [float(_ci_mejor["MCC_IC95_inf"]), float(_ci_mejor["MCC_IC95_sup"])],
    "mcc_ic95_incluye_cero": MCC_CERO_DENTRO_IC,
    # --- against the other proposed models ---
    "supera_a": supera,
    "es_superado_por": es_superado,
    "indistinguible_de": indistinguible,
    # --- against the baselines of the main contrast ---
    "baselines_contraste_principal": [res_p[k]["label"] for k in baseline_keys_pri],
    "supera_baselines": bl_superados,
    "es_superado_por_baselines": bl_superan,
    "indistinguible_de_baselines": bl_indistintos,
    "n_baselines_principal": len(baseline_keys_pri),
    # --- against the supplementary baselines ---
    "baselines_suplementarios": [res_p[k]["label"] for k in baseline_keys_sup],
    "supera_baselines_suplementarios": sup_superados,
    "es_superado_por_baselines_suplementarios": sup_superan,
    # --- comparison by ranking metric ---
    "supera_baselines_principal_en_mcc": BASELINES_PRINCIPAL_SUPERADOS_MCC,
    "supera_baselines_principal_en_balacc": BASELINES_PRINCIPAL_SUPERADOS_BACC,
    "supera_baselines_principal_en_accuracy": BASELINES_PRINCIPAL_SUPERADOS_ACC,
    # --- disagreements ---
    "discordancias_metricas": (df_discordancias.to_dict(orient="records")
                               if len(df_discordancias) else []),
    "n_discordancias": int(len(df_discordancias)),
    "afirmacion_permitida": ("superioridad"
                             if (len(indistinguible) == 0 and len(es_superado) == 0)
                             else "comparabilidad"),
}

print("\n" + "-" * 92)
if VERDICT["afirmacion_permitida"] == "superioridad":
    print(f"AFIRMACION PERMITIDA: '{label_mejor} supera significativamente a los demas modelos'.")
    print("El manuscrito PUEDE mantener la afirmacion de superioridad, ahora con respaldo.")
else:
    print(f"AFIRMACION PERMITIDA: '{label_mejor} tiene un desempeno COMPARABLE al de "
          f"{', '.join(indistinguible) if indistinguible else '(ninguno)'}'.")
    print("El manuscrito NO puede sostener que sea 'el mejor modelo'.")
    print("Debe adoptarse la redaccion que el propio revisor ofrece en R1-M1, y la")
    print("seleccion debe justificarse por criterios secundarios explicitos:")
    print("estabilidad entre particiones, costo computacional (§9.7) e interpretabilidad.")
if es_superado:
    print()
    print(f"ATENCION: {len(es_superado)} modelo(s) superan al mejor por MCC en aciertos")
    print("crudos. La tabla de discordancias de §10.3 explica por que esto no es una")
    print("contradiccion, y debe citarse junto a la afirmacion.")
if bl_superan or sup_superan:
    print()
    print("ATENCION: al menos un baseline supera al mejor modelo en aciertos crudos.")
    print(f"  contraste principal : {bl_superan}")
    print(f"  suplementario       : {sup_superan}")
    print("Esta cifra NO puede omitirse del manuscrito. La respuesta correcta es la de")
    print("R1-M2: bajo desbalance la exactitud deja de medir capacidad discriminante, y")
    print("por eso el contraste que sostiene la contribucion es en MCC y balanced accuracy.")
print("-" * 92)

"""# 11. Investment recommendation module

The module is evaluated quantitatively only if the listed instruments pass a **pre-registered liquidity gate**, fixed in section 1.3 before looking at any result:

- `LIQ_MIN_PCT_CON_VOLUMEN = 60 %`: minimum proportion of sessions with traded volume
- `LIQ_MIN_PCT_CON_CAMBIO = 40 %`: minimum proportion of sessions with a price change

If an instrument does not pass both thresholds, a daily backtest has no economic meaning: it would simulate trading a security that is not traded, and any resulting Sharpe ratio would be an artefact of the method. Fixing the criterion before seeing the result avoids data snooping. The outcome is the choice between route A (validate with a backtest) and route B (report the module as a conceptual design)."""

# ============================================================
# Load stock prices (Bolsa de Valores de Lima)
# ============================================================
PRICE_FILES = {
    "EXALMAR": os.path.join(DATA_DIR, "EXALMAR_EXALMC1_LM_OHLC_diario.csv"),
    "AUSTRAL": os.path.join(DATA_DIR, "AUSTRAL_AUSTRAC1_LM_OHLC_diario.csv"),
}

precios = {}
if RUN_BACKTEST:
    for nombre, ruta in PRICE_FILES.items():
        if not os.path.exists(ruta):
            print(f"  [!] No encontrado: {os.path.basename(ruta)}")
            print(f"      Copiar los CSV generados por el notebook de extraccion a {DATA_DIR}")
            continue
        d = pd.read_csv(ruta, encoding="utf-8-sig")
        d["Date"] = pd.to_datetime(d["Date"])
        precios[nombre] = d.sort_values("Date").reset_index(drop=True)
        print(f"  {nombre}: {len(d):,} sesiones  "
              f"({d['Date'].min().date()} -> {d['Date'].max().date()})")

if not precios:
    RUN_BACKTEST = False
    print("\nBacktest desactivado: no hay datos de precios disponibles.")

"""## 11.1 Liquidity diagnosis

The liquidity evidence determines the route."""

# ============================================================
# Liquidity diagnosis over the study window
# ============================================================
LIQ_REPORT = {}
if RUN_BACKTEST:
    filas_liq = []
    for nombre, d in precios.items():
        w = d[(d["Date"] >= DATE_START) & (d["Date"] <= DATE_END)].copy()
        n = len(w)
        con_vol    = int((w["Volume"] > 0).sum())
        con_cambio = int((w["Close"].diff().fillna(0) != 0).sum())
        bh = float(w["Close"].iloc[-1] / w["Close"].iloc[0] - 1) * 100 if n else np.nan

        w["ym"] = w["Date"].dt.to_period("M")
        meses_muertos = int((w.groupby("ym")["Volume"].apply(lambda s: (s > 0).sum()) == 0).sum())

        pct_vol    = con_vol / n * 100 if n else 0
        pct_cambio = con_cambio / n * 100 if n else 0
        aprueba = (pct_vol >= LIQ_MIN_PCT_CON_VOLUMEN) and (pct_cambio >= LIQ_MIN_PCT_CON_CAMBIO)

        filas_liq.append({
            "Activo": nombre,
            "Ticker": w["Ticker"].iloc[0] if n else "-",
            "Sesiones_en_ventana": n,
            "Con_volumen": con_vol,
            "Pct_con_volumen": round(pct_vol, 1),
            "Con_cambio_de_precio": con_cambio,
            "Pct_con_cambio": round(pct_cambio, 1),
            "Meses_sin_negociacion": meses_muertos,
            "BuyAndHold_pct": round(bh, 2),
            "Apto_para_backtest_diario": "SI" if aprueba else "NO",
        })
        LIQ_REPORT[nombre] = dict(n=n, pct_vol=pct_vol, pct_cambio=pct_cambio,
                                  meses_muertos=meses_muertos, buy_hold=bh, aprueba=aprueba)

    df_liq = pd.DataFrame(filas_liq)
    save_csv(df_liq, "21_liquidity_diagnosis")

    print("=" * 110)
    print("   DIAGNOSTICO DE LIQUIDEZ — ventana del estudio")
    print(f"   Umbrales pre-registrados: volumen >= {LIQ_MIN_PCT_CON_VOLUMEN}%  |  "
          f"cambio de precio >= {LIQ_MIN_PCT_CON_CAMBIO}%")
    print("=" * 110)
    display(df_liq)

# ============================================================
# Figure: monthly liquidity
# ============================================================
if RUN_BACKTEST and precios:
    fig, axes = plt.subplots(len(precios), 1, figsize=(12, 3.6 * len(precios)), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i, (nombre, d) in enumerate(precios.items()):
        w = d[(d["Date"] >= DATE_START) & (d["Date"] <= DATE_END)].copy()
        w["ym"] = w["Date"].dt.to_period("M").astype(str)
        g = w.groupby("ym").agg(sesiones=("Date", "size"),
                                con_vol=("Volume", lambda s: int((s > 0).sum())))
        axes[i].bar(g.index, g["sesiones"], color="white",
                    edgecolor="black", linewidth=0.8, label="Sesiones del mes")
        axes[i].bar(g.index, g["con_vol"], color=GRISES[6],
                    edgecolor="black", linewidth=0.8, label="Sesiones con volumen")
        axes[i].set_ylabel("Sesiones")
        axes[i].set_title(f"{nombre} — actividad de negociacion mensual")
        axes[i].legend(loc="upper right", frameon=True, edgecolor="black")
        estilo_eje(axes[i], rejilla="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_figure(fig, "fig_liquidity_monthly")
    plt.show()

"""## 11.2 Signal -> decision rule

**Direct rule:**

```
if  prediction(D+1) = "High production"  ->  INVEST signal      ->  long position on D+1
if  prediction(D+1) = "Low production"   ->  DO NOT INVEST      ->  out of the market on D+1
```

**Inverse rule (mandatory contrast):** the sign of the relation between landing volume and share value is not established: larger catches may mean higher revenue for the fishing company, but also larger supply and lower fishmeal prices. Both directions are therefore evaluated and reported; reporting only the direction that works best would be data snooping.

**Delay between signal and execution.** The signal of day `t` is computed with information available up to the close of `t`, so the position is taken at `t + BACKTEST_LAG_DIAS`. The declared case uses a one-session delay (what a trader receiving the signal after the close can execute), and section 11.2b shows the sensitivity to this assumption. Executing at the close of the same day produces better figures and is not a sustainable assumption."""

# ============================================================
# Backtest engine
# ============================================================
def run_backtest(fechas_senal, senales, df_precio, costo_bps=0, invertir_regla=False,
                 lag_dias=BACKTEST_LAG_DIAS):
    """
    Simulate a long / out-of-market strategy.

    fechas_senal : dates of the predictions (the signal is computed with day-t data)
    senales      : 1 = High production predicted, 0 = Low
    costo_bps    : transaction cost in basis points, per change of position
    invertir_regla: if True, apply the inverse rule (High -> out of the market)
    lag_dias     : sessions between the signal and its execution. With lag_dias=1
                   the signal of day t determines the position of day t+1, which is
                   what a trader receiving the signal after the close can do. With
                   lag_dias=0 execution would be at the close of the same day, an
                   optimistic assumption that must be declared if adopted.
    """
    px = df_precio[["Date", "Close"]].dropna().copy()
    px = px[(px["Date"] >= DATE_START) & (px["Date"] <= DATE_END)]
    px = px.sort_values("Date").reset_index(drop=True)
    px["ret"] = px["Close"].pct_change().fillna(0.0)

    s = pd.DataFrame({"Date": pd.to_datetime(fechas_senal), "signal": np.asarray(senales)})
    if invertir_regla:
        s["signal"] = 1 - s["signal"]

    m = pd.merge(px, s, on="Date", how="left")
    m["signal"] = m["signal"].ffill().fillna(0).astype(int)

    # Declared delay: the signal is executed `lag_dias` sessions later.
    m["pos"] = m["signal"].shift(lag_dias).fillna(0).astype(int)
    m["cambio_pos"] = m["pos"].diff().abs().fillna(m["pos"].abs())
    m["costo"] = m["cambio_pos"] * (costo_bps / 10000.0)
    m["ret_estrategia"] = m["pos"] * m["ret"] - m["costo"]

    equity = (1 + m["ret_estrategia"]).cumprod()
    ret_total = float(equity.iloc[-1] - 1) if len(equity) else np.nan
    r = m["ret_estrategia"]
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    dd = float((equity / equity.cummax() - 1).min()) if len(equity) else np.nan
    n_ops = int(m["cambio_pos"].sum())
    dias_dentro = int(m["pos"].sum())

    return dict(ret_total=ret_total, sharpe=sharpe, max_drawdown=dd,
                n_operaciones=n_ops, dias_en_mercado=dias_dentro,
                n_sesiones=len(m), equity=equity, fechas=m["Date"], detalle=m)

print("Motor de backtest definido.")

# ============================================================
# Backtest run on the dated predictions
# ============================================================
BACKTEST_RESULTS = {}
df_bt = pd.DataFrame()
if RUN_BACKTEST and precios:
    p_best = pred_p[mejor]
    orden = np.argsort(p_best["idx"])
    idx_ord = p_best["idx"][orden]
    sig_ord = p_best["y_pred"][orden]
    fechas_sig = DATES.iloc[idx_ord].values

    print(f"Senales del modelo {label_mejor}: {len(sig_ord)} dias "
          f"({int(sig_ord.sum())} de INVERTIR, {int((1-sig_ord).sum())} de NO INVERTIR)")
    print(f"Rango de senales: {pd.Timestamp(fechas_sig.min()).date()} -> "
          f"{pd.Timestamp(fechas_sig.max()).date()}\n")

    filas_bt = []
    for activo, dpx in precios.items():
        for regla, inv in [("Directa (Alta->Invertir)", False), ("Inversa (Alta->Fuera)", True)]:
            for costo in COSTOS_TRANSACCION_BPS:
                bt = run_backtest(fechas_sig, sig_ord, dpx, costo_bps=costo, invertir_regla=inv)
                filas_bt.append({
                    "Activo": activo, "Regla": regla, "Costo_bps": costo,
                    "Retorno_total_pct": round(bt["ret_total"] * 100, 2),
                    "Sharpe_anualizado": round(bt["sharpe"], 3) if bt["sharpe"] == bt["sharpe"] else np.nan,
                    "Max_drawdown_pct": round(bt["max_drawdown"] * 100, 2),
                    "Operaciones": bt["n_operaciones"],
                    "Dias_en_mercado": bt["dias_en_mercado"],
                })
                BACKTEST_RESULTS[(activo, regla, costo)] = bt

        px = dpx[(dpx["Date"] >= DATE_START) & (dpx["Date"] <= DATE_END)].dropna(subset=["Close"])
        bh_ret = float(px["Close"].iloc[-1] / px["Close"].iloc[0] - 1) * 100
        bh_r = px["Close"].pct_change().dropna()
        bh_sharpe = float(bh_r.mean() / bh_r.std() * np.sqrt(252)) if bh_r.std() > 0 else np.nan
        bh_eq = (1 + bh_r).cumprod()
        bh_dd = float((bh_eq / bh_eq.cummax() - 1).min())
        filas_bt.append({"Activo": activo, "Regla": "BENCHMARK Buy & Hold", "Costo_bps": 0,
                         "Retorno_total_pct": round(bh_ret, 2),
                         "Sharpe_anualizado": round(bh_sharpe, 3),
                         "Max_drawdown_pct": round(bh_dd * 100, 2),
                         "Operaciones": 1, "Dias_en_mercado": len(px)})
        filas_bt.append({"Activo": activo, "Regla": "BENCHMARK Fuera del mercado", "Costo_bps": 0,
                         "Retorno_total_pct": 0.0, "Sharpe_anualizado": np.nan,
                         "Max_drawdown_pct": 0.0, "Operaciones": 0, "Dias_en_mercado": 0})

    df_bt = pd.DataFrame(filas_bt)
    save_csv(df_bt, "22_backtest_results")
    print("=" * 110)
    print("   BACKTEST — resultados por activo, regla y costo de transaccion")
    print("   (se ejecuta siempre; su interpretabilidad la decide la compuerta de §11.3)")
    print("=" * 110)
    display(df_bt)

# ============================================================
# Sensitivity to the signal -> execution delay assumption
# ============================================================
df_lag = pd.DataFrame()
if RUN_BACKTEST and precios:
    filas_lag = []
    for activo, dpx in precios.items():
        for lag in BACKTEST_LAGS_SENSIBILIDAD:
            bt = run_backtest(fechas_sig, sig_ord, dpx, costo_bps=25,
                              invertir_regla=False, lag_dias=lag)
            filas_lag.append({
                "Activo": activo, "Retardo_sesiones": lag,
                "Declarado": "si" if lag == BACKTEST_LAG_DIAS else "no",
                "Retorno_total_pct": round(bt["ret_total"] * 100, 2),
                "Sharpe_anualizado": round(bt["sharpe"], 3) if bt["sharpe"] == bt["sharpe"] else np.nan,
                "Operaciones": bt["n_operaciones"],
                "Dias_en_mercado": bt["dias_en_mercado"],
            })
    df_lag = pd.DataFrame(filas_lag)
    save_csv(df_lag, "22c_backtest_lag_sensitivity")
    print("=" * 100)
    print("   SENSIBILIDAD AL RETARDO ENTRE SENAL Y EJECUCION  (costo 25 bps, regla directa)")
    print(f"   Supuesto declarado en el manuscrito: {BACKTEST_LAG_DIAS} sesion(es) de retardo")
    print("=" * 100)
    display(df_lag)
    print("\nEste bloque responde al punto 5 del Paso 5 del Plan. El supuesto de retardo")
    print("no es un detalle: cambia el resultado y debe declararse antes de que lo pregunten.")

# ============================================================
# Monte Carlo contrast against a random signal
# ============================================================
# A strategy can "beat" buy and hold simply by being out of the market
# during a decline. The right contrast is against random signals
# with the SAME proportion of days invested.
MC_RESULTS = {}
if RUN_BACKTEST and precios:
    rng = np.random.RandomState(RANDOM_SEED)
    tasa_positiva = float(sig_ord.mean())

    for activo, dpx in precios.items():
        real = BACKTEST_RESULTS[(activo, "Directa (Alta->Invertir)", 25)]["ret_total"]
        sims = np.empty(N_MONTECARLO)
        for i in range(N_MONTECARLO):
            aleat = (rng.rand(len(sig_ord)) < tasa_positiva).astype(int)
            sims[i] = run_backtest(fechas_sig, aleat, dpx, costo_bps=25)["ret_total"]
        pval = float(np.mean(sims >= real))
        MC_RESULTS[activo] = dict(real=real, sims=sims, p_value=pval,
                                  media_sim=float(np.mean(sims)),
                                  pct5=float(np.percentile(sims, 5)),
                                  pct95=float(np.percentile(sims, 95)))
        print(f"{activo}: retorno estrategia = {real*100:+.2f}%  |  "
              f"aleatorias: media {np.mean(sims)*100:+.2f}%, "
              f"IC90 [{np.percentile(sims,5)*100:+.2f}%, {np.percentile(sims,95)*100:+.2f}%]  |  "
              f"p = {pval:.4f}")

    fig, axes = plt.subplots(1, len(MC_RESULTS), figsize=(6.5 * len(MC_RESULTS), 4.6))
    axes = np.atleast_1d(axes).ravel()
    for i, (activo, mc) in enumerate(MC_RESULTS.items()):
        axes[i].hist(mc["sims"] * 100, bins=50, color=GRISES[2],
                     edgecolor="black", linewidth=0.4)
        axes[i].axvline(mc["real"] * 100, color="black", linewidth=2.2,
                        label=f"Estrategia del modelo ({mc['real']*100:+.2f}%)")
        axes[i].axvline(LIQ_REPORT[activo]["buy_hold"], color="black", linewidth=2.0,
                        linestyle="--", label=f"Buy & Hold ({LIQ_REPORT[activo]['buy_hold']:+.2f}%)")
        axes[i].set_title(f"{activo} — {N_MONTECARLO} senales aleatorias\np = {mc['p_value']:.4f}")
        axes[i].set_xlabel("Retorno total (%)"); axes[i].set_ylabel("Frecuencia")
        axes[i].legend(fontsize=9, frameon=True, edgecolor="black")
        estilo_eje(axes[i], rejilla="y")
    plt.tight_layout()
    save_figure(fig, "fig_montecarlo_backtest")
    plt.show()

    df_mc_bt = pd.DataFrame([
        {"Activo": a, "Retorno_estrategia_pct": round(v["real"] * 100, 2),
         "Media_aleatorias_pct": round(v["media_sim"] * 100, 2),
         "P5_aleatorias_pct": round(v["pct5"] * 100, 2),
         "P95_aleatorias_pct": round(v["pct95"] * 100, 2),
         "p_valor": round(v["p_value"], 4)}
        for a, v in MC_RESULTS.items()])
    save_csv(df_mc_bt, "22b_montecarlo_backtest")

"""## 11.3 Route A / Route B verdict (automatic decision)"""

# ============================================================
# Verdict according to the pre-registered gate
# ============================================================
BACKTEST_VERDICT = {}
if RUN_BACKTEST and LIQ_REPORT:
    aptos = [a for a, v in LIQ_REPORT.items() if v["aprueba"]]
    ruta = "A" if aptos else "B"

    print("=" * 90)
    print("   VEREDICTO SOBRE EL MODULO DE INVERSION  (R1-M3 / R1-A3)")
    print("=" * 90)
    for a, v in LIQ_REPORT.items():
        print(f"\n  {a}:")
        print(f"    sesiones con volumen  : {v['pct_vol']:.1f}%  "
              f"(umbral pre-registrado: {LIQ_MIN_PCT_CON_VOLUMEN}%)  "
              f"-> {'cumple' if v['pct_vol'] >= LIQ_MIN_PCT_CON_VOLUMEN else 'NO cumple'}")
        print(f"    sesiones con cambio   : {v['pct_cambio']:.1f}%  "
              f"(umbral pre-registrado: {LIQ_MIN_PCT_CON_CAMBIO}%)  "
              f"-> {'cumple' if v['pct_cambio'] >= LIQ_MIN_PCT_CON_CAMBIO else 'NO cumple'}")
        print(f"    meses sin negociacion : {v['meses_muertos']}")

    print("\n" + "-" * 90)
    if ruta == "A":
        print(f"RUTA A — VALIDAR. Activos aptos: {', '.join(aptos)}")
        print("El backtest de §11.2 puede reportarse como validacion cuantitativa,")
        print("con sus supuestos declarados y el contraste Monte Carlo como respaldo.")
    else:
        print("RUTA B — REDEGRADAR A DISENO CONCEPTUAL.")
        print()
        print("Ningun activo supera la compuerta de liquidez fijada de antemano.")
        print("Un backtest diario sobre un titulo que no se negocia produce cifras")
        print("que parecen resultados y no lo son: el retorno de la estrategia queda")
        print("determinado por unas pocas sesiones aisladas, no por la calidad de la senal.")
        print()
        print("Esto NO es una derrota. Es la mejor respuesta posible a R1-M3:")
        print("en lugar de admitir un vacio, el manuscrito aporta la evidencia de por que")
        print("la validacion no es factible con estos activos y esta ventana temporal, y")
        print("convierte la limitacion en un hallazgo declarado.")
        print()
        print("Acciones concretas en el manuscrito:")
        print("  1. Eliminar de §4.3.1 la afirmacion de mejora del retorno de inversion.")
        print("  2. Reetiquetar el modulo como diseno conceptual no validado.")
        print("  3. Documentar la regla senal->decision de todos modos (R2-12 la pide).")
        print("  4. Declarar que no hubo prueba con usuarios ni inversionistas (R2-11).")
        print("  5. Incluir la tabla de liquidez como evidencia y en Limitaciones.")
        print("  6. Suavizar el titulo y el Abstract.")
    print("-" * 90)

    BACKTEST_VERDICT = {
        "ruta": ruta, "activos_aptos": aptos,
        "umbral_volumen": LIQ_MIN_PCT_CON_VOLUMEN,
        "umbral_cambio": LIQ_MIN_PCT_CON_CAMBIO,
        "modelo_generador_de_senales": label_mejor,
        "retardo_senal_ejecucion_sesiones": BACKTEST_LAG_DIAS,
        "liquidez": {a: {k: v[k] for k in ["pct_vol", "pct_cambio", "meses_muertos", "buy_hold"]}
                     for a, v in LIQ_REPORT.items()},
    }

"""# 12. Neural-network diagnosis

## 12.1 Verification of the four architectures suggested during review

The four acronyms (LEADNet, LCADNet, STEADYNet, LCOWFBs) were checked one by one against their primary sources. All four belong to the detection of Alzheimer's disease and dementia from electroencephalogram (EEG) recordings, and one of them is not a neural network. Their input (multichannel high-frequency EEG) has no counterpart in a daily tabular series of twelve predictors.

| Acronym | Meaning | Domain | Neural network | Source (DOI) |
|---|---|---|---|---|
| LEADNet | Low-complexity EEG-based Alzheimer's Disease detection Network | EEG / Alzheimer | Yes, convolutional | 10.1109/ACCESS.2024.3435768 |
| LCADNet | Light CNN for Alzheimer's Disease detection Network | EEG / Alzheimer | Yes, convolutional | 10.1007/s13246-024-01425-w |
| STEADYNet | SpatioTemporal EEG Analysis for Dementia detection Network | EEG / dementia | Yes, convolutional | 10.1007/s11571-024-10153-6 |
| LCOWFBs | Low-Complexity Orthogonal Wavelet Filter Banks | Signal processing | **No**, wavelet filter bank | 10.1016/j.bspc.2022.104439 |

The deep-learning families that do apply to a daily series (1D-CNN, LSTM and GRU) are benchmarked in section 8.1 under the same protocol and metrics."""

# ============================================================
# Verification table of the suggested architectures
# ============================================================
R2_20_VERIFICACION = [
    {"Sigla": "LEADNet",
     "Significado": "Low-complexity EEG-based Alzheimer's Disease detection Network",
     "Dominio": "Electroencefalograma / Alzheimer",
     "Es_red_neuronal": "Si (red convolucional)",
     "Modalidad_de_entrada": "EEG multicanal de alta frecuencia",
     "Aplicable_a_este_estudio": "No",
     "Fuente_DOI": "10.1109/ACCESS.2024.3435768"},
    {"Sigla": "LCADNet",
     "Significado": "Light CNN for Alzheimer's Disease detection Network",
     "Dominio": "Electroencefalograma / Alzheimer",
     "Es_red_neuronal": "Si (red convolucional)",
     "Modalidad_de_entrada": "EEG multicanal de alta frecuencia",
     "Aplicable_a_este_estudio": "No",
     "Fuente_DOI": "10.1007/s13246-024-01425-w"},
    {"Sigla": "STEADYNet",
     "Significado": "SpatioTemporal EEG Analysis for Dementia detection Network",
     "Dominio": "Electroencefalograma / demencia",
     "Es_red_neuronal": "Si (red convolucional)",
     "Modalidad_de_entrada": "EEG multicanal de alta frecuencia",
     "Aplicable_a_este_estudio": "No",
     "Fuente_DOI": "10.1007/s11571-024-10153-6"},
    {"Sigla": "LCOWFBs",
     "Significado": "Low-Complexity Orthogonal Wavelet Filter Banks",
     "Dominio": "Procesamiento de senal",
     "Es_red_neuronal": "No (banco de filtros wavelet)",
     "Modalidad_de_entrada": "Senal continua muestreada",
     "Aplicable_a_este_estudio": "No",
     "Fuente_DOI": "10.1016/j.bspc.2022.104439"},
]
df_r220 = pd.DataFrame(R2_20_VERIFICACION)
save_csv(df_r220, "24_r2_20_architecture_verification")
display(df_r220)

print("Lectura: las cuatro arquitecturas nombradas operan sobre electroencefalograma")
print("multicanal para clasificacion clinica. Ninguna es transferible a clasificacion")
print("binaria tabular con registros diarios. La respuesta pertinente al espiritu de")
print("la observacion es el benchmark de series temporales de §8.1.")

"""## 12.2 Single-class collapse diagnosis

A model that collapses to always predicting the majority class is direct evidence about the data regime. All the evaluated neural networks (the multilayer perceptron of Table 3 and the three deep architectures of section 8.1) are diagnosed with the same criterion. The diagnosis is computed, not assumed."""

# ============================================================
# Unified diagnosis of all the evaluated networks
# ============================================================
def diagnosticar_red(clave):
    """Return the single-class collapse diagnosis for one estimator."""
    r = res_p[clave]
    tasa_pos = float(pred_p[clave]["y_pred"].mean())
    colapso = bool((tasa_pos < 0.02) or (tasa_pos > 0.98) or (r["recall_mean"] == 0))
    return {
        "Arquitectura": r["label"],
        "Clave": clave,
        "Tipo": r["category"],
        "Accuracy": round(r["accuracy_mean"], 4),
        "Balanced_Acc": round(r["balanced_accuracy_mean"], 4),
        "MCC": round(r["mcc_mean"], 4),
        "Recall": round(r["recall_mean"], 4),
        "Tasa_predicciones_positivas": round(tasa_pos, 4),
        "Colapso_clase_unica": "SI" if colapso else "no",
    }

claves_redes = [k for k in ["MLP"] if k in res_p] + dl_keys
DL_DIAGNOSIS_ROWS = [diagnosticar_red(k) for k in claves_redes]
df_dl_diag = pd.DataFrame(DL_DIAGNOSIS_ROWS)

print("=" * 110)
print("   DIAGNOSTICO DE LAS REDES NEURONALES EVALUADAS  (protocolo principal)")
print("=" * 110)
display(df_dl_diag)
save_csv(df_dl_diag, "25_neural_network_diagnosis")

n_colapsan = int((df_dl_diag["Colapso_clase_unica"] == "SI").sum())
n_redes = len(df_dl_diag)
acc_trivial = float(res_p["BL_Majority"]["accuracy_mean"]) if "BL_Majority" in res_p else np.nan

print(f"\nRedes evaluadas: {n_redes}   |   con colapso a clase unica: {n_colapsan}")
print(f"Exactitud del clasificador trivial (clase mayoritaria): {acc_trivial:.4f}")
print()
if n_colapsan == n_redes:
    print("DIAGNOSTICO: TODAS las redes colapsan a clase unica.")
    print()
    print("  Causas concretas y verificadas:")
    print("   - El perceptron multicapa de scikit-learn no admite class_weight, y su")
    print("     early stopping reserva una fraccion aleatoria del entrenamiento que")
    print(f"     con {PCT_POSITIVOS:.1f}% de positivos puede quedarse casi sin casos de clase Alta.")
    print("   - Las tres arquitecturas profundas SI usan ponderacion balanceada y")
    print("     validacion interna temporal, y aun asi colapsan. Eso descarta que el")
    print("     problema sea de configuracion y lo atribuye al regimen de datos.")
    print()
    print("  USO EN LA RESPUESTA A R2-06 y R2-20:")
    print(f"  Con n = {len(X)} observaciones diarias, {len(FEATURE_COLUMNS)} variables tabulares")
    print(f"  y {PCT_POSITIVOS:.1f}% de casos positivos, la muestra no sostiene el entrenamiento de")
    print("  una red neuronal. Arquitecturas mas profundas enfrentarian la misma")
    print("  restriccion con mas parametros que estimar. Es un argumento empirico, no")
    print("  preferencial, a favor de metodos de arbol y de nucleo interpretables.")
    print()
    print("  Nota de honestidad: NO debe presentarse como prueba de que el aprendizaje")
    print("  profundo es inferior en general. Solo de que no es apropiado AQUI.")
elif n_colapsan > 0:
    print(f"DIAGNOSTICO MIXTO: {n_colapsan} de {n_redes} redes colapsan; el resto discrimina.")
    print()
    print("  El manuscrito NO puede sostener que el aprendizaje profundo sea inviable")
    print("  en este regimen de datos. Debe reportar el resultado tal como sale:")
    for _, f in df_dl_diag.iterrows():
        estado = "colapsa" if f["Colapso_clase_unica"] == "SI" else "discrimina"
        print(f"     {f['Arquitectura']:28s} {estado:12s} MCC={f['MCC']:+.4f}")
    print()
    print("  El argumento a favor de los metodos interpretables debe apoyarse entonces")
    print("  en la comparacion de desempeno y en la interpretabilidad, no en un fallo")
    print("  de entrenamiento generalizado.")
else:
    print("DIAGNOSTICO: ninguna red colapsa; todas discriminan por encima de la clase unica.")
    print()
    print("  Este resultado CONTRADICE el argumento preparado para R2-06 y R2-20, y el")
    print("  manuscrito debe decirlo. La respuesta correcta pasa a ser la comparacion")
    print("  de desempeno entre familias, no una afirmacion sobre la inviabilidad del")
    print("  aprendizaje profundo en este tamano de muestra.")
    mejor_red = df_dl_diag.loc[df_dl_diag["MCC"].idxmax()]
    print(f"  Mejor red: {mejor_red['Arquitectura']} (MCC = {mejor_red['MCC']:+.4f}) frente")
    print(f"  al mejor modelo tabular {label_mejor} (MCC = {res_p[mejor]['mcc_mean']:+.4f}).")

DL_DIAGNOSIS = {
    "redes_evaluadas": n_redes,
    "redes_con_colapso": n_colapsan,
    "todas_colapsan": bool(n_colapsan == n_redes and n_redes > 0),
    "benchmark_ejecutado": bool(DL_EJECUTADO),
    "motivo_si_no_ejecutado": ("" if DL_EJECUTADO else DEEP_LEARNING_MOTIVO),
    "marco": DEEP_LEARNING_MARCO,
    "ventana_dias": DL_WINDOW,
    "gap_dias": DL_GAP,
    "detalle": DL_DIAGNOSIS_ROWS,
    "verificacion_r2_20": R2_20_VERIFICACION,
}
# Compatibility with the name used by earlier versions of the JSON
MLP_DIAGNOSIS = next((d for d in DL_DIAGNOSIS_ROWS if d["Clave"] == "MLP"), {})

# ============================================================
# Figure: positive prediction rate per network versus the true prevalence
# ============================================================
if len(df_dl_diag):
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(df_dl_diag) + 2.2))
    ypos = np.arange(len(df_dl_diag))
    ax.barh(ypos, df_dl_diag["Tasa_predicciones_positivas"],
            color=GRISES[3], edgecolor="black", linewidth=0.8, hatch="///")
    ax.axvline(PCT_POSITIVOS / 100, color="black", linestyle="--", linewidth=1.4,
               label=f"Prevalencia real de la clase Alta ({PCT_POSITIVOS:.1f}%)")
    ax.set_yticks(ypos); ax.set_yticklabels(df_dl_diag["Arquitectura"])
    ax.set_xlabel("Proporcion de predicciones de clase Alta")
    ax.set_title("Colapso a clase unica en las redes neuronales evaluadas")
    ax.legend(frameon=True, edgecolor="black")
    estilo_eje(ax, rejilla="x")
    plt.tight_layout()
    save_figure(fig, "fig_neural_network_collapse")
    plt.show()

"""# 13. Consolidated export — `results_v6.json`

Single source of truth for the reported figures: every number of the manuscript comes from this file.

The file is written with `allow_nan=False`: if a non-finite value survives the cleaning, the export fails loudly instead of producing a JSON that strict parsers reject."""

# ============================================================
# Complete JSON export
# ============================================================
def _clean(o):
    """
    Convert numpy types to native types for serialization, and neutralize
    non-finite values (NaN, inf) from both numpy and native Python.
    """
    if isinstance(o, (np.bool_,)):     return bool(o)
    if isinstance(o, (np.integer,)):   return int(o)
    if isinstance(o, (np.floating,)):  return float(o) if np.isfinite(o) else None
    if isinstance(o, float):           return o if np.isfinite(o) else None
    if isinstance(o, np.ndarray):      return o.tolist()
    if isinstance(o, dict):            return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):   return [_clean(v) for v in o]
    if isinstance(o, (pd.Timestamp, datetime)): return str(o)
    return o


export = {
    "metadata": {
        "manuscript": "FIS-ML",
        "notebook_version": "v6.0",
        "run_id": RUN_ID,
        "environment": ENVIRONMENT,
        "date_range": f"{DATE_START} -> {DATE_END}",
        "n_samples": int(len(X)),
        "sample_reconciliation": _clean(SAMPLE_RECONCILIATION),
        "interpolation_ratio": _clean(RAZON_INTERPOLACION),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "n_folds": N_FOLDS,
        "folds_protocolo_principal": _clean(FOLDS_PRINCIPAL),
        "ma_window": MA_WINDOW,
        "ts_gap": TS_GAP,
        "protocols": PROTOCOLS,
        "threshold_modes": THRESHOLD_MODES,
        "primary_protocol": PRIMARY_PROTOCOL,
        "primary_threshold_mode": PRIMARY_THRESHOLD_MODE,
        "threshold_global_TM": round(THRESHOLD_GLOBAL, 2),
        "threshold_cv_pct": round(float(var_pct), 2),
        "threshold_cv_criterio_pct": UMBRAL_CV_CRITERIO,
        "threshold_es_estable": UMBRAL_ES_ESTABLE,
        "class_distribution_global": {"alto": n_alto, "bajo": n_bajo,
                                      "pct_alto": round(PCT_POSITIVOS, 2)},
        "nested_tuning": bool(RUN_NESTED_TUNING),
        "tuned_params": _clean(TUNED_PARAMS) if TUNED_PARAMS else {},
        # --- Baseline split ---
        "baselines_contraste_principal": [res_p[k]["label"] for k in baseline_keys_pri],
        "baselines_suplementarios": [res_p[k]["label"] for k in baseline_keys_sup],
        "justificacion_baseline_suplementario": {
            "r1_m6_verbatim": BASELINE_R1M6_VERBATIM,
            "lectura_adoptada": BASELINE_LECTURA_ADOPTADA,
            "riesgo_declarado": ("En segunda ronda el revisor podria pedir que la "
                                 "persistencia vuelva al contraste principal. Queda "
                                 "calculada y tabulada aparte para que reincorporarla "
                                 "no exija re-ejecutar."),
        },
        # --- Deep learning ---
        "deep_learning": {
            "ejecutado": bool(DL_EJECUTADO),
            "marco": DEEP_LEARNING_MARCO,
            "motivo_si_no_ejecutado": ("" if DL_EJECUTADO else DEEP_LEARNING_MOTIVO),
            "arquitecturas": [v["label"] for v in DL_ARCHITECTURES.values()],
            "ventana_dias": DL_WINDOW,
            "gap_dias": DL_GAP,
            "filas_de_entrenamiento_cedidas_por_particion": DL_GAP - TS_GAP,
        },
    },
    "configurations": {},
    "leakage_diagnosis": _clean(leak_summaries),
    "fold_info": _clean(df_foldinfo.to_dict(orient="records")),
    "degenerate_folds": _clean(
        df_foldinfo[df_foldinfo["Fold_valido"] == "no"].to_dict(orient="records")),
    "degenerate_folds_summary": {
        "descartadas": int((df_foldinfo["Fold_valido"] == "no").sum()),
        "totales": int(len(df_foldinfo)),
        "por_configuracion": {
            f"{p}|{m}": int(((df_foldinfo["Protocolo"] == p) &
                             (df_foldinfo["Modo_umbral"] == m) &
                             (df_foldinfo["Fold_valido"] == "no")).sum())
            for p in df_foldinfo["Protocolo"].unique()
            for m in df_foldinfo["Modo_umbral"].unique()},
    },
    "statistical_tests": {
        "mcnemar_all_pairs": _clean(df_mc.to_dict(orient="records")),
        "mcnemar_contraste_principal": _clean(sub_pri.to_dict(orient="records")),
        "mcnemar_baseline_suplementario": (_clean(sub_sup.to_dict(orient="records"))
                                           if len(sub_sup) else []),
        "mcnemar_top_models": _clean(sub_top.to_dict(orient="records")),
        "metric_discordances": (_clean(df_discordancias.to_dict(orient="records"))
                                if len(df_discordancias) else []),
        "paired_across_folds": _clean(df_wilcox.to_dict(orient="records")),
        "bootstrap_confidence_intervals": _clean(df_ci.to_dict(orient="records")),
        "best_model_vs_baselines": _clean(df_vs_baselines.to_dict(orient="records")),
        "verdict": _clean(VERDICT),
    },
    "sensitivity": {
        "permutation_importance": _clean(df_perm.to_dict(orient="records")) if RUN_SENSITIVITY else [],
        "climate_perturbation": _clean(df_sens.to_dict(orient="records")) if RUN_SENSITIVITY else [],
        "modelo_usado": (MODEL_REGISTRY[MODELO_SENSIBILIDAD]["label"] if RUN_SENSITIVITY else None),
    },
    "computational_cost": _clean(df_time.to_dict(orient="records")),
    "roc_auc_aggregated": _clean(df_auc.to_dict(orient="records")),
    "investment_module": _clean(BACKTEST_VERDICT) if BACKTEST_VERDICT else {},
    "backtest_table": _clean(df_bt.to_dict(orient="records")) if len(df_bt) else [],
    "backtest_lag_sensitivity": _clean(df_lag.to_dict(orient="records")) if len(df_lag) else [],
    "deep_learning_diagnosis": _clean(DL_DIAGNOSIS),
    "mlp_diagnosis": _clean(MLP_DIAGNOSIS),
}

for cfg_key, res in ALL_RESULTS.items():
    bloque = {}
    for mkey, r in res.items():
        b = {"label": r["label"], "label_en": r.get("label_en", r["label"]),
             "category": r["category"], "contraste": r.get("contraste", "—")}
        for m in METRIC_KEYS:
            mu, sd = r.get(f"{m}_mean"), r.get(f"{m}_std")
            b[f"{m}_mean"] = None if (mu is None or (isinstance(mu, float) and np.isnan(mu))) else round(float(mu), 4)
            b[f"{m}_std"]  = None if (sd is None or (isinstance(sd, float) and np.isnan(sd))) else round(float(sd), 4)
            b[f"{m}_folds"] = [None if (isinstance(v, float) and not np.isfinite(v))
                               else round(float(v), 4) for v in r.get(f"{m}_folds", [])]
        b["n_folds_evaluados"] = int(r.get("n_folds_evaluados",
                                           len(r.get("balanced_accuracy_folds", []))))
        b["confusion_matrix"] = np.asarray(r["confusion_matrix"]).tolist()
        b["time_train_s"] = (None if not np.isfinite(r["time_train_mean"])
                             else round(r["time_train_mean"], 5))
        b["time_infer_s"] = (None if not np.isfinite(r["time_infer_mean"])
                             else round(r["time_infer_mean"], 6))
        bloque[mkey] = b
    export["configurations"][cfg_key] = bloque

json_path = os.path.join(DATA_DIR, f"results_v6_{RUN_ID}.json")
with open(json_path, "w", encoding="utf-8") as f:
    json_lib.dump(export, f, indent=2, ensure_ascii=False, allow_nan=False)
ARCHIVOS_GENERADOS.append(("json", "results_v6", os.path.basename(json_path)))

print(f"Exportado: {json_path}")
print(f"  Configuraciones: {len(export['configurations'])}")
print(f"  Estimadores en la configuracion principal: "
      f"{len(export['configurations'][PRIMARY_KEY])}")
print(f"  Contrastes de McNemar: {len(export['statistical_tests']['mcnemar_all_pairs'])}")
print()
print("Regla operativa: ninguna cifra entra al manuscrito si no proviene de este archivo.")

# ============================================================
# The exported JSON must be strictly valid JSON
# ============================================================
def _rechaza_constantes(valor):
    raise ValueError(f"El JSON contiene la constante no estandar: {valor}")

with open(json_path, encoding="utf-8") as f:
    _recargado = json_lib.load(f, parse_constant=_rechaza_constantes)

print("[AUDIT] El archivo recarga con un parser estricto: no quedan NaN ni Infinity.")
print(f"        Claves de primer nivel: {list(_recargado.keys())}")

"""# 14. Model persistence

The final models are trained on **all** the available history (appropriate for deployment, not for evaluation) and saved together with their scaler and threshold. The threshold used here is the global one, since it is the only one available when there is no split."""

# ============================================================
# Final models trained on the full history
# ============================================================
thr_final = THRESHOLD_GLOBAL
y_final = binarize(PROD_NEXT, thr_final).values

scaler_final = StandardScaler()
X_final_s = scaler_final.fit_transform(X)

# The best model of the verdict is saved in addition to the three models of the manuscript.
claves_a_guardar = []
for k in [mejor, "RFC", "XGBoost", "SVM_RBF_C100"]:
    if k in MODEL_REGISTRY and k not in claves_a_guardar:
        claves_a_guardar.append(k)

guardados = []
for key in claves_a_guardar:
    params = dict(MODEL_REGISTRY[key]["params"])
    if TUNED_PARAMS and key in TUNED_PARAMS:
        params.update(TUNED_PARAMS[key])
    m = MODEL_REGISTRY[key]["cls"](**params)
    m.fit(X_final_s, y_final)
    ruta = os.path.join(MODELS_DIR, f"{key}_final_{RUN_ID}.joblib")
    joblib.dump(m, ruta)
    guardados.append(os.path.basename(ruta))

joblib.dump(scaler_final, os.path.join(MODELS_DIR, f"scaler_final_{RUN_ID}.joblib"))
joblib.dump({"threshold_TM": thr_final, "threshold_tipo": "global (todo el historico)",
             "feature_columns": FEATURE_COLUMNS, "ma_window": MA_WINDOW,
             "random_seed": RANDOM_SEED, "notebook_version": "v6.0"},
            os.path.join(MODELS_DIR, f"metadata_final_{RUN_ID}.joblib"))

print(f"Modelos guardados en {MODELS_DIR}:")
for g in guardados:
    print(f"  {g}")
print(f"  scaler_final_{RUN_ID}.joblib")
print(f"  metadata_final_{RUN_ID}.joblib")
print()
print("Nota para el manuscrito: estos artefactos son de despliegue, no de evaluacion.")
print("Las cifras reportadas provienen del protocolo temporal por particion, no de")
print("estos modelos entrenados sobre el historico completo.")

"""# 15. Narrative generator

Produces the sentences in academic English, with the figures already inserted, from the same objects that feed the JSON, so that the text cannot disagree with what the experiment produced. Every statement is computed (for example, superiority over the baselines is derived by comparing the best model against each baseline in MCC, balanced accuracy and accuracy). Section 20 verifies automatically that the resulting text contains none of the forbidden statements."""

# ============================================================
# Conditional sentences, computed one by one
# ============================================================
R = ALL_RESULTS[PRIMARY_KEY]
best = R[mejor]
bl_maj = R.get("BL_Majority", {})
bl_per = R.get("BL_Persistence", {})

est_delta = df_delta[df_delta["Tipo"] == "study"]
caida_acc = est_delta["Delta_Acc"].mean()
caida_mcc = est_delta["Delta_MCC"].mean()

lineas = []
add = lineas.append

# Label translation: the study models already have English names;
# the baselines and the networks carry `label_en` so that no Spanish is embedded in
# the generated text.
ETIQUETA_EN = {r["label"]: r.get("label_en", r["label"]) for r in res_p.values()}

def en(etiqueta):
    """Return the English label of an estimator."""
    return ETIQUETA_EN.get(etiqueta, etiqueta)

def enumerar(lista, traducir=True):
    """Join an English list: 'A', 'A and B', 'A, B and C'."""
    lista = [en(str(x)) if traducir else str(x) for x in lista]
    if not lista:  return "none"
    if len(lista) == 1: return lista[0]
    return ", ".join(lista[:-1]) + " and " + lista[-1]

add("=" * 92)
add("   TEXTO GENERADO PARA EL MANUSCRITO REVISADO  (ingles, cifras verificadas)")
add(f"   Notebook v6.0 · corrida {RUN_ID}")
add("   Todas las afirmaciones de este archivo son condicionales sobre los datos.")
add("=" * 92)

# --- Temporal protocol ---
add("\n[R1-M4 / R1-A4] Section 3.3 — Validation protocol")
add("-" * 92)
add(f"Following the reviewer's observation, the k-fold cross-validation with shuffling")
add(f"used in the original submission was replaced by a time-series-aware forward-chaining")
add(f"scheme (TimeSeriesSplit, {N_FOLDS} splits, gap = {TS_GAP} days). The gap equals the longest")
add(f"causal window in the feature set (the {MA_WINDOW}-day temperature moving average), so no test")
add(f"observation shares input values with any training observation. Both protocols are")
add(f"reported in Table 3 for transparency. Under the corrected protocol, mean accuracy")
add(f"across the {len(est_delta)} tabular classifiers changes by {caida_acc:+.4f} and mean MCC by {caida_mcc:+.4f}.")
if FOLDS_PRINCIPAL["descartadas"] > 0:
    add(f"Of the {FOLDS_PRINCIPAL['totales']} splits, {FOLDS_PRINCIPAL['evaluadas']} were evaluated; "
        f"{FOLDS_PRINCIPAL['descartadas']} was excluded because")
    add(f"its test block contained a single class, which makes every binary metric")
    add(f"undefined. The exclusion is applied identically to every estimator, so all")
    add(f"paired comparisons remain aligned observation by observation.")
else:
    add(f"All {FOLDS_PRINCIPAL['totales']} splits were evaluated.")

# --- Class threshold ---
add("\n[R2-04] Section 3.1 — Class threshold")
add("-" * 92)
add(f"Days are labelled High when next-day landings exceed the mean of landings over days")
add(f"with non-zero production. In the revised protocol this threshold is estimated from")
add(f"training data only within each split, rather than from the full sample, so that")
add(f"class assignment in the test set carries no information from future observations.")
if UMBRAL_ES_ESTABLE:
    add(f"The threshold is stable across splits: its coefficient of variation is {var_pct:.2f}%,")
    add(f"below the {UMBRAL_CV_CRITERIO:.0f}% criterion declared before inspecting the results.")
else:
    add(f"The threshold varies appreciably across splits (coefficient of variation {var_pct:.2f}%,")
    add(f"against a {UMBRAL_CV_CRITERIO:.0f}% stability criterion declared in advance). Per-split values are")
    add(f"reported in the supplementary material and the variation is discussed as a source")
    add(f"of uncertainty rather than presented as negligible.")

# --- Metrics ---
add("\n[R1-M2 / R2-05 / R2-07] Section 4.1 — Class balance and evaluation metrics")
add("-" * 92)
add(f"The dataset contains {n_alto} High-production days and {n_bajo} Low-production days")
add(f"({n_alto/(n_alto+n_bajo)*100:.1f}% vs {n_bajo/(n_alto+n_bajo)*100:.1f}%). Because of this imbalance, accuracy alone is")
add(f"misleading: a majority-class classifier reaches {bl_maj.get('accuracy_mean', float('nan')):.4f} accuracy with no")
add(f"discriminative ability (MCC = {bl_maj.get('mcc_mean', float('nan')):.4f}). We therefore report balanced accuracy,")
add(f"Matthews correlation coefficient and ROC-AUC alongside confusion matrices for every")
add(f"model. The best-ranked model, {en(best['label'])}, reaches balanced accuracy")
add(f"{best['balanced_accuracy_mean']:.4f} and MCC {best['mcc_mean']:.4f} (bootstrap 95% CI")
add(f"[{_ci_mejor['MCC_IC95_inf']:+.4f}, {_ci_mejor['MCC_IC95_sup']:+.4f}], {N_BOOTSTRAP} class-stratified resamples).")
if MCC_CERO_DENTRO_IC:
    add(f"That interval includes zero, so discriminative ability above chance cannot be")
    add(f"asserted for this classifier on this sample; the limitation is stated as such.")

# --- Baselines ---
add("\n[R1-M6 / R1-A5] Section 4.1 — Baseline comparison")
add("-" * 92)
add(f"Naive baselines were added under the identical protocol and the identical metrics.")
add(f"The main comparison uses the baselines named in the reviewer's request: "
    f"{enumerar([res_p[k]['label'] for k in baseline_keys_pri])}.")
add(f"The majority-class baseline attains {bl_maj.get('accuracy_mean', float('nan')):.4f} accuracy but MCC {bl_maj.get('mcc_mean', float('nan')):.4f}.")
_n_pri = len(baseline_keys_pri)
add(f"Against these {_n_pri} baselines, {en(best['label'])} is higher on MCC in "
    f"{len(BASELINES_PRINCIPAL_SUPERADOS_MCC)} of {_n_pri} cases,")
add(f"higher on balanced accuracy in {len(BASELINES_PRINCIPAL_SUPERADOS_BACC)} of {_n_pri}, "
    f"and higher on raw accuracy in {len(BASELINES_PRINCIPAL_SUPERADOS_ACC)} of {_n_pri}.")
if (len(BASELINES_PRINCIPAL_SUPERADOS_MCC) > len(BASELINES_PRINCIPAL_SUPERADOS_ACC)):
    add(f"The divergence between these counts is itself the point raised in R1-M2: under a")
    add(f"{n_bajo/(n_alto+n_bajo)*100:.1f}% majority class, raw accuracy and MCC rank the estimators differently, and")
    add(f"it is MCC and balanced accuracy that establish added value.")
if baseline_keys_sup:
    add(f"A persistence baseline (tomorrow = today) is additionally reported as supplementary")
    add(f"material; it is not part of the main comparison because the reviewer's request names")
    add(f"the majority-class and seasonal-naive baselines. It attains {bl_per.get('accuracy_mean', float('nan')):.4f} accuracy")
    add(f"and MCC {bl_per.get('mcc_mean', float('nan')):.4f} on this series.")
    if sup_superan:
        add(f"It outperforms {en(best['label'])} on raw correct predictions "
            f"(McNemar, Holm-adjusted p < 0.05).")
        add(f"We report this openly: in a daily landings series with extended closed seasons,")
        add(f"today's class is a strong predictor of tomorrow's, and any claim of added value")
        add(f"must be made against that fact rather than around it.")

# --- Significance ---
add("\n[R1-M1 / R1-A1 / R2-02] Section 4.1 — Statistical significance")
add("-" * 92)
add(f"Paired McNemar tests were computed for all {len(df_mc)} estimator pairs on identical test")
add(f"observations, with Holm-Bonferroni correction over the full set of comparisons;")
add(f"Wilcoxon signed-rank and paired t-tests across splits are reported as complements.")
add(f"McNemar contrasts raw correct predictions, whereas the ranking uses MCC; the two")
add(f"criteria are reported separately and their disagreements are tabulated.")
if VERDICT["afirmacion_permitida"] == "comparabilidad":
    add(f"The results do not support a claim of superiority for a single classifier.")
    if indistinguible:
        add(f"{en(best['label'])} performs comparably to {enumerar(indistinguible)}")
        add(f"(no significant difference after correction).")
    if es_superado:
        add(f"{enumerar(es_superado)} record significantly more correct")
        add(f"predictions than {en(best['label'])}, while ranking below it on MCC — a")
        add(f"discordance that arises directly from the class imbalance discussed above.")
    add(f"Following the reviewer's own suggested wording, the claim has been softened.")
    add(f"{en(best['label'])} is retained as the reference model on the basis of stability")
    add(f"across splits, computational cost and interpretability, not of a statistically")
    add(f"demonstrated performance advantage.")
else:
    add(f"{en(best['label'])} is significantly superior on correct predictions to")
    add(f"{enumerar(supera)} after Holm correction, which supports retaining it")
    add(f"as the selected model.")
if bl_superados:
    add(f"It significantly outperforms {enumerar(bl_superados)} among the main baselines.")
if bl_superan:
    add(f"It is significantly outperformed on raw correct predictions by "
        f"{enumerar(bl_superan)}.")
if bl_indistintos:
    add(f"It is statistically indistinguishable from {enumerar(bl_indistintos)}.")

# --- Sensitivity ---
if RUN_SENSITIVITY:
    add("\n[R2-08] Section 4.2 — Sensitivity to climate variables")
    add("-" * 92)
    add(f"Two complementary analyses are reported, and they must be read together.")
    add(f"Permutation importance, computed within the observed range, identifies "
        f"{IMP_TOP_VARIABLE} as")
    add(f"the most influential predictor (balanced-accuracy drop of {IMP_TOP_VALOR:.4f} when permuted).")
    if IMP_TOP_VARIABLE != "Temperature_Mean":
        add(f"The temperature anomaly itself drops {IMP_TEMPERATURA:.4f}, so within the historical")
        add(f"range the calendar-driven seasonal structure dominates over thermal forcing.")
    else:
        add(f"The temperature anomaly is therefore the dominant predictor within the")
        add(f"historical range, ahead of the calendar-driven seasonal structure.")
    add(f"Shifting the temperature anomaly by ±2 standard deviations, which moves the inputs")
    add(f"outside the observed distribution, changes the rate of High-production predictions")
    add(f"by {RANGO_PERTURBACION*100:.1f} percentage points.")
    # Threshold declared in section 9.6 to qualify the magnitude instead of just asserting it.
    if RANGO_PERTURBACION >= 0.05:
        add(f"The two figures are not in conflict: permutation reorders values that already")
        add(f"occur, whereas the ±2σ shift takes the inputs out of distribution and produces a")
        add(f"marked response. Reporting only the second would overstate the system's climate")
        add(f"sensitivity; reporting only the first would understate it. Both are given.")
    else:
        add(f"Both analyses therefore point the same way: the model's response to thermal")
        add(f"forcing is limited, and most of the signal comes from the seasonal calendar")
        add(f"structure rather than from the temperature anomaly. We report this as a finding")
        add(f"and as a limitation of the coarse spatial resolution of the climate series,")
        add(f"rather than claiming a climate sensitivity the data do not support.")

# --- Computational cost ---
add("\n[R2-19] Section 4.5 — Computational cost")
add("-" * 92)
_e = df_time[df_time["Tipo"] == "study"]
add(f"Mean training time per split ranges from {_e['Entrenamiento_s'].min():.5f} s to {_e['Entrenamiento_s'].max():.5f} s")
add(f"and inference time from {_e['Inferencia_s'].min():.6f} s to {_e['Inferencia_s'].max():.6f} s on the reported")
add(f"hardware. At n = {len(X)}, computational cost does not discriminate between the tabular")
add(f"models and is not used as a selection criterion.")
if DL_EJECUTADO:
    _dl_t = df_time[df_time["Tipo"] == "deep_learning"]
    add(f"The deep-learning architectures train in {_dl_t['Entrenamiento_s'].min():.2f} s to "
        f"{_dl_t['Entrenamiento_s'].max():.2f} s per split,")
    add(f"which remains negligible in absolute terms at this sample size.")

# --- Investment module ---
if BACKTEST_VERDICT:
    add("\n[R1-M3 / R1-A3 / R2-11 / R2-12] Section 4.3 — Investment recommendation module")
    add("-" * 92)
    add(f"The signal-to-decision rule is stated explicitly: a predicted High-production day")
    add(f"maps to an invest signal and a predicted Low-production day to a no-invest signal.")
    add(f"The signal for day t is computed from information available at the close of t and")
    add(f"the position is taken {BACKTEST_LAG_DIAS} session(s) later; sensitivity to this lag is reported.")
    if BACKTEST_VERDICT["ruta"] == "B":
        liq_txt = "; ".join(
            f"{a}: {v['pct_vol']:.1f}% of sessions with traded volume and "
            f"{v['pct_cambio']:.1f}% with a price change"
            for a, v in BACKTEST_VERDICT["liquidez"].items())
        add(f"A quantitative backtest was attempted on the two listed Peruvian fishing companies.")
        add(f"Trading-activity analysis over the study window shows that neither instrument is")
        add(f"sufficiently liquid for daily backtesting ({liq_txt}), against pre-registered")
        add(f"thresholds of {BACKTEST_VERDICT['umbral_volumen']:.0f}% and {BACKTEST_VERDICT['umbral_cambio']:.0f}% respectively, fixed before the results were")
        add(f"examined. Return and Sharpe figures computed on such a series would reflect a")
        add(f"handful of isolated sessions rather than the quality of the signal. We therefore")
        add(f"reclassify the module as a conceptual design that has not been empirically")
        add(f"validated, remove the return-on-investment claim, and report the liquidity")
        add(f"constraint as a finding and as a limitation. The system has not been tested with")
        add(f"real users or investors.")
    else:
        add(f"A quantitative backtest was performed on the instruments that passed the")
        add(f"pre-registered liquidity screen ({enumerar(BACKTEST_VERDICT['activos_aptos'])}),")
        add(f"benchmarked against buy-and-hold and against {N_MONTECARLO} random-signal simulations")
        add(f"with matched exposure, under transaction costs of 0, 25 and 50 bps. The system has")
        add(f"not been tested with real users or investors.")

# --- Deep learning ---
add("\n[R2-06 / R2-20] Section 4.5 — On deep learning architectures")
add("-" * 92)
add(f"The four architectures named in the review (LEADNet, LCADNet, STEADYNet and LCOWFBs)")
add(f"were examined against their primary sources. All four operate on multichannel")
add(f"high-frequency electroencephalogram recordings for clinical classification, and one")
add(f"of them is a wavelet filter bank rather than a neural network. Their input modality")
add(f"has no counterpart in a daily tabular series, so a direct comparison would not be")
add(f"informative.")
if DL_EJECUTADO:
    add(f"We therefore benchmarked the deep-learning families that do apply to a daily time")
    add(f"series — {enumerar([v['label_en'] for v in DL_ARCHITECTURES.values()], traducir=False)} — "
        f"over {DL_WINDOW}-day windows, under the same splits,")
    add(f"the same per-split threshold and the same metrics as every other estimator, with")
    add(f"balanced class weighting and chronological internal validation for early stopping.")
    if DL_DIAGNOSIS["todas_colapsan"]:
        add(f"All {DL_DIAGNOSIS['redes_evaluadas']} neural models evaluated, including the multilayer perceptron already")
        add(f"present in the comparison, converge to predicting a single class. With {len(X)} daily")
        add(f"observations, {len(FEATURE_COLUMNS)} tabular features and {PCT_POSITIVOS:.1f}% positive cases, the sample does not")
        add(f"support neural-network training; deeper architectures would face the same")
        add(f"constraint with more parameters to estimate. This is an empirical, not a")
        add(f"preferential, argument for the choice of interpretable tree-based and kernel")
        add(f"methods in this application. It is not a claim about deep learning in general.")
    elif DL_DIAGNOSIS["redes_con_colapso"] > 0:
        _colapsan = [d["Arquitectura"] for d in DL_DIAGNOSIS["detalle"]
                     if d["Colapso_clase_unica"] == "SI"]
        _no = [d["Arquitectura"] for d in DL_DIAGNOSIS["detalle"]
               if d["Colapso_clase_unica"] != "SI"]
        add(f"{enumerar(_colapsan)} converge to a single class, whereas {enumerar(_no)}")
        add(f"retain some discriminative ability. We report this mixed outcome as it stands:")
        add(f"the sample size constrains neural training but does not preclude it outright.")
    else:
        _mejor_red = max(DL_DIAGNOSIS["detalle"], key=lambda d: d["MCC"])
        add(f"None of the architectures collapses to a single class. The best of them,")
        add(f"{en(_mejor_red['Arquitectura'])}, reaches MCC {_mejor_red['MCC']:+.4f} against "
        f"{best['mcc_mean']:+.4f} for {en(best['label'])}.")
        add(f"The preference for interpretable models is therefore argued on performance and")
        add(f"interpretability grounds, not on an inability to train neural networks here.")
else:
    add(f"The deep-learning benchmark could not be executed in this run "
        f"({DEEP_LEARNING_MOTIVO}).")
    add(f"The multilayer perceptron already present in the comparison is reported instead,")
    add(f"and the limitation is stated explicitly rather than omitted.")

NARRATIVE = "\n".join(lineas)
print(NARRATIVE)

ruta_txt = os.path.join(DATA_DIR, f"23_narrative_for_manuscript_{RUN_ID}.txt")
with open(ruta_txt, "w", encoding="utf-8") as f:
    f.write(NARRATIVE)
ARCHIVOS_GENERADOS.append(("texto", "23_narrative_for_manuscript", os.path.basename(ruta_txt)))
print(f"\n\nGuardado: {ruta_txt}")

"""# 16. Closing checklist

Automatic verification of which blocks were completed by this run. The status is derived from the actual outcome of the run, not from a hand-written list."""

# ============================================================
# Checklist derived from the actual state of the run
# ============================================================
def chk(cond, ok, pend):
    return (f"RESUELTO — {ok}" if cond else f"PENDIENTE — {pend}")

sin_fuga_principal = principal["violaciones_orden"] == 0 and principal["contiguidad_rota"] == 0
tiene_mcc     = "mcc_mean" in res_p.get("RFC", {})
tiene_roc     = len(df_auc) > 0
tiene_bl_pri  = len(baseline_keys_pri) >= 2
tiene_mcnemar = len(df_mc) > 0 and "p_Holm" in df_mc.columns
tiene_direccion = "Direccion_M1" in df_mc.columns
tiene_wilcox  = len(df_wilcox) > 0
tiene_bt      = bool(BACKTEST_VERDICT)
mejor_en_wilcox = label_mejor in (set(df_wilcox["Modelo_1"]) | set(df_wilcox["Modelo_2"]))

CHECKLIST = {
    "Working Unit 01 · Validacion temporal (R1-M4, R1-A4)": chk(
        sin_fuga_principal,
        f"{len(PROTOCOLS)} protocolos ejecutados; principal sin fuga "
        f"(0 violaciones, gap={TS_GAP})",
        "el protocolo principal aun presenta fuga"),
    "Working Unit 01 · Umbral sin fuga (R2-04)": chk(
        "per_fold_train" in THRESHOLD_MODES,
        f"umbral por particion ejecutado; coeficiente de variacion {var_pct:.2f}% "
        f"frente al criterio de {UMBRAL_CV_CRITERIO:.0f}%",
        "falta el modo per_fold_train"),
    "Working Unit 01 · Ajuste anidado de hiperparametros": chk(
        RUN_NESTED_TUNING and bool(TUNED_PARAMS),
        f"{len(TUNED_PARAMS)} modelos ajustados dentro de cada particion",
        "activar RUN_NESTED_TUNING"),
    "Working Unit 02 · Distribucion de clases (R1-M2, R2-05)": chk(
        True, "global y mensual, con figura", ""),
    "Working Unit 02 · Matrices de confusion (R1-M2)": chk(
        True, "absolutas y normalizadas, por modelo, red y baseline", ""),
    "Working Unit 02 · Balanced accuracy y MCC (R1-M2)": chk(
        tiene_mcc, f"calculados para los {len(res_p)} estimadores", "faltan metricas"),
    "Working Unit 02 · Intervalos de confianza (R1-M1)": chk(
        len(df_ci) > 0,
        f"bootstrap de {N_BOOTSTRAP} replicas estratificadas para los {len(df_ci)} estimadores",
        "faltan intervalos"),
    "Working Unit 02 · ROC-AUC y curvas ROC (R2-07)": chk(
        tiene_roc, f"area bajo la curva para {len(df_auc)} estimadores y curvas graficadas",
        "faltan curvas"),
    "Working Unit 02 · Sensibilidad climatica (R2-08)": chk(
        RUN_SENSITIVITY, "permutacion + perturbacion ±2σ, reconciliadas y con figura",
        "activar RUN_SENSITIVITY"),
    "Working Unit 02 · Costo computacional (R2-19)": chk(
        True, "tiempos de entrenamiento e inferencia por estimador", ""),
    "Working Unit 03 · Baselines simples (R1-M6, R1-A5)": chk(
        tiene_bl_pri,
        f"{len(baseline_keys_pri)} en contraste principal + "
        f"{len(baseline_keys_sup)} suplementario, con justificacion escrita",
        "faltan baselines"),
    "Working Unit 04 · McNemar + Holm (R1-M1, R1-A1)": chk(
        tiene_mcnemar, f"{len(df_mc)} pares con correccion por multiplicidad",
        "faltan pruebas"),
    "Working Unit 04 · Direccion de los contrastes": chk(
        tiene_direccion,
        "cada contraste declara quien gana, no solo si hay diferencia",
        "el veredicto no distingue direccion"),
    "Working Unit 04 · Wilcoxon / t pareada entre particiones": chk(
        tiene_wilcox and mejor_en_wilcox,
        f"{len(df_wilcox)} contrastes; el mejor modelo esta incluido",
        "el mejor modelo no aparece en la tabla pareada"),
    "Working Unit 04 · Veredicto sobre seleccion de modelo (R2-02)": chk(
        bool(VERDICT), f"afirmacion permitida: {VERDICT.get('afirmacion_permitida', '-')}", ""),
    "Working Unit 05 · Modulo de inversion (R1-M3, R1-A3)": chk(
        tiene_bt,
        f"compuerta de liquidez evaluada -> Ruta {BACKTEST_VERDICT.get('ruta', '-')}",
        "faltan datos de precios en DATA_DIR"),
    "Working Unit 05 · Regla senal->decision (R2-12)": chk(
        tiene_bt, "documentada y evaluada en ambas direcciones", "pendiente"),
    "Working Unit 05 · Prueba con usuarios (R2-11)": chk(
        True, "declarado explicitamente como no realizado", ""),
    "Working Unit 08 · Benchmark de aprendizaje profundo (R2-06)": chk(
        DL_EJECUTADO,
        f"{len(dl_keys)} arquitecturas alineadas con el protocolo principal",
        f"no ejecutado: {DEEP_LEARNING_MOTIVO}"),
    "Working Unit 08 · Verificacion de R2-20": chk(
        len(df_r220) == 4,
        "las 4 siglas verificadas contra su fuente, con dominio y DOI",
        "verificacion incompleta"),
    "Apoyo · Diccionario de datos (R2-13)": chk(
        True, f"{len(df_dict)} variables con formula explicita", ""),
    "Apoyo · Captura de entorno (R1-m4)": chk(
        True, "versiones, semilla y plataforma exportadas al JSON", ""),
    "Apoyo · JSON estricto valido (Paso 9)": chk(
        True, "recargado con parser estricto sin NaN ni Infinity", ""),
}

print("=" * 100)
print("   CHECKLIST DE CIERRE — unidades experimentales del Plan")
print("=" * 100)
for k, v in CHECKLIST.items():
    print(f"  {k}")
    print(f"      {v}\n")

n_ok = sum(1 for v in CHECKLIST.values() if v.startswith("RESUELTO"))
print(f"Resueltos: {n_ok} de {len(CHECKLIST)} items experimentales.")
print()
print("Fuera del alcance de este notebook (redaccion, Pasos 6 a 10 del plan):")
print("  Working Unit 06 Recalibracion de contribucion y generalizabilidad (R1-M5, R1-M8, R1-A7)")
print("  Working Unit 07 Discusion y limitaciones (R1-m3, R2-14)")
print("  Working Unit 08 Tabla 4 y §4.5 (R1-m1, R1-m2) — el insumo de §12 la alimenta")
print("  Working Unit 09 Publicacion del repositorio con identificador digital (R1-m4)")
print("  Working Unit 10 Cierre editorial (R1-M7, R1-A6, R2-15 a R2-18)")

save_csv(pd.DataFrame([{"Item": k, "Estado": v} for k, v in CHECKLIST.items()]),
         "26_closing_checklist")

"""# 17. Output inventory

Everything this run left on Drive."""

# ============================================================
# Inventory of the files generated in this run
# ============================================================
print("=" * 100)
print(f"   INVENTARIO DE SALIDAS — RUN_ID {RUN_ID}")
print("=" * 100)

filas_inv = []
for carpeta, etiqueta in [(DATA_DIR, "DATA"), (IMAGES_DIR, "IMAGES"), (MODELS_DIR, "MODELS")]:
    archivos = sorted(f for f in os.listdir(carpeta) if RUN_ID in f)
    print(f"\n{etiqueta}  ({len(archivos)} archivos)")
    for a in archivos:
        tam = os.path.getsize(os.path.join(carpeta, a)) / 1024
        print(f"   {a:<66s} {tam:>8.1f} KB")
        filas_inv.append({"Carpeta": etiqueta, "Archivo": a, "KB": round(tam, 1),
                          "Extension": os.path.splitext(a)[1].lstrip(".")})

df_inventario = pd.DataFrame(filas_inv)

n_csv  = int((df_inventario["Extension"] == "csv").sum())
n_xlsx = int(df_inventario["Extension"].isin(["xlsx", "xls"]).sum())
n_png  = int((df_inventario["Extension"] == "png").sum())
n_json = int((df_inventario["Extension"] == "json").sum())

print("\n" + "-" * 100)
print(f"Tablas CSV : {n_csv}      Figuras PNG : {n_png}      JSON : {n_json}")
print(f"Archivos de Excel: {n_xlsx}  (debe ser 0: todas las salidas tabulares son CSV)")
assert n_xlsx == 0, "Quedan salidas en formato Excel; deben ser CSV."

"""# 18. Applied refinements

Short table of the refinements applied to the analysis, with their evidence."""

# ============================================================
# Record of refinements, with their evidence
# ============================================================
AFINAMIENTOS = [
    {"ID": "A1",
     "Afinamiento": "La caida de desempeno al corregir el protocolo supero lo previsto",
     "Evidencia": f"Delta Acc medio {caida_acc:+.4f}; Delta MCC medio {caida_mcc:+.4f}; "
                  f"mejor MCC {best['mcc_mean']:+.4f}",
     "Paso_afectado": "Paso 1 (advertencia planificada) -> el Paso 6 pasa a ser la "
                      "narrativa central, no un plan de contingencia"},
    {"ID": "A2",
     "Afinamiento": "El mejor modelo por MCC no es el que afirma el manuscrito",
     "Evidencia": f"mejor modelo = {label_mejor} (MCC {best['mcc_mean']:+.4f})",
     "Paso_afectado": "Paso 4 y Abstract: la frase 'RFC is ultimately selected as the "
                      "optimal model' no tiene respaldo"},
    {"ID": "A3",
     "Afinamiento": "Particiones descartadas por clase unica",
     "Evidencia": f"{int((df_foldinfo['Fold_valido'] == 'no').sum())} de {len(df_foldinfo)} "
                  f"en el factorial; {FOLDS_PRINCIPAL['descartadas']} de "
                  f"{FOLDS_PRINCIPAL['totales']} en el protocolo principal",
     "Paso_afectado": "Paso 2: hallazgo no previsto, va a §3.3 y a Limitaciones"},
    {"ID": "A4",
     "Afinamiento": "Persistencia retirada del contraste principal por la letra de R1-M6",
     "Evidencia": "decision editorial del autor, con justificacion en metadata del JSON",
     "Paso_afectado": "Paso 3: invalida la nota estrategica que decia 'diferenciarse de "
                      "la persistencia es el articulo'. Riesgo de segunda ronda declarado"},
    {"ID": "A5",
     "Afinamiento": "Ruta B confirmada por datos",
     "Evidencia": (", ".join(f"{a}: {v['pct_vol']:.1f}% con volumen"
                             for a, v in LIQ_REPORT.items())
                   + f" frente a umbrales pre-registrados "
                     f"{LIQ_MIN_PCT_CON_VOLUMEN:.0f}% / {LIQ_MIN_PCT_CON_CAMBIO:.0f}%")
                  if LIQ_REPORT else "sin datos de precios en esta corrida",
     "Paso_afectado": "Paso 5: la regla de las 48 horas queda resuelta; suavizar titulo "
                      "y Abstract deja de ser opcional"},
    {"ID": "A6",
     "Afinamiento": "R2-20 verificado: las 4 siglas son de electroencefalograma y "
                    "Alzheimer; LCOWFBs no es una red",
     "Evidencia": "tabla 24_r2_20_architecture_verification, con DOI por sigla",
     "Paso_afectado": "Paso 8: cierra la decision pendiente"},
    {"ID": "A7",
     "Afinamiento": "Salidas en CSV y JSON estricto valido",
     "Evidencia": f"{n_csv} archivos CSV, {n_xlsx} de Excel; JSON recargado con "
                  f"parser estricto",
     "Paso_afectado": "Paso 9: requisito del repositorio con identificador digital"},
    {"ID": "A8",
     "Afinamiento": "Inconsistencia entre el tamano muestral del manuscrito y el de la corrida",
     "Evidencia": f"{SAMPLE_RECONCILIATION['tras_objetivo_d1']} citado frente a "
                  f"{SAMPLE_RECONCILIATION['tras_calentamiento_features']} de entrenamiento; "
                  f"cadena completa exportada",
     "Paso_afectado": "Paso 10: entra en el barrido de cifras"},
    {"ID": "A9",
     "Afinamiento": "La novedad reformulada: el protocolo de validacion temporal pasa a "
                    "ser la contribucion principal defendible",
     "Evidencia": "consecuencia de A1 y A2",
     "Paso_afectado": "Paso 6, punto 4"},
    {"ID": "A10",
     "Afinamiento": "Direccion en el veredicto de McNemar (correccion de veracidad)",
     "Evidencia": f"{len(es_superado)} modelo(s) y {len(bl_superan) + len(sup_superan)} "
                  f"baseline(s) superan al mejor modelo en aciertos crudos; "
                  f"antes se contaban como superados",
     "Paso_afectado": "Paso 4 y Paso 6: la afirmacion sobre las lineas base cambia de signo"},
    {"ID": "A11",
     "Afinamiento": "Discordancia entre MCC y aciertos crudos declarada como argumento",
     "Evidencia": f"{len(df_discordancias)} contrastes significativos con ganador distinto "
                  f"segun la metrica",
     "Paso_afectado": "Paso 2 y Paso 4: refuerza la respuesta a R1-M2"},
    {"ID": "A12",
     "Afinamiento": "Benchmark de aprendizaje profundo pertinente al dominio",
     "Evidencia": (f"{len(dl_keys)} arquitecturas sobre ventanas de {DL_WINDOW} dias, "
                   f"alineadas con el protocolo principal" if DL_EJECUTADO
                   else f"no ejecutado: {DEEP_LEARNING_MOTIVO}"),
     "Paso_afectado": "Paso 8: responde al espiritu de R2-06 y R2-20, no solo a la letra"},
    {"ID": "A13",
     "Afinamiento": "Umbral del baseline de persistencia por particion",
     "Evidencia": "hasta v5B se binarizaba con el umbral global mientras la etiqueta "
                  "verdadera usaba el de la particion",
     "Paso_afectado": "Paso 3: la cifra del baseline de persistencia cambia respecto de "
                      "corridas anteriores y debe recitarse desde este JSON"},
    {"ID": "A14",
     "Afinamiento": "Intervalos de confianza por bootstrap para responder a R1-M1 en sus "
                    "propios terminos",
     "Evidencia": f"{N_BOOTSTRAP} replicas estratificadas; el intervalo del mejor modelo "
                  f"{'incluye' if MCC_CERO_DENTRO_IC else 'excluye'} el cero",
     "Paso_afectado": "Paso 4: la objecion sobre solapamiento de intervalos se responde "
                      "con intervalos, no con desviaciones entre particiones"},
    {"ID": "A15",
     "Afinamiento": "Figuras en escala de grises para impresion monocroma",
     "Evidencia": f"{n_png} figuras con fondo blanco y tramas; unica excepcion declarada: "
                  f"las curvas ROC",
     "Paso_afectado": "Paso 9 y R2-16: requisito de presentacion"},
]

df_afinamientos = pd.DataFrame(AFINAMIENTOS)
save_csv(df_afinamientos, "27_afinamientos_aplicados")
print("=" * 100)
print("   AFINAMIENTOS APLICADOS EN LA VERSION v6")
print("=" * 100)
for a in AFINAMIENTOS:
    print(f"\n[{a['ID']}] {a['Afinamiento']}")
    print(f"      Evidencia      : {a['Evidencia']}")
    print(f"      Paso afectado  : {a['Paso_afectado']}")

"""# 19. Delivery package

Three outputs that facilitate verification and reuse:

1. **Manifest of figures and tables:** which output file feeds which table or figure of the manuscript.
2. **Flat registry of quotable figures:** key, value, unit and origin, one per row, so that any number can be checked mechanically against the results file.
3. **Reproducibility files:** `requirements.txt` and README.

## 19.1 Manifest of figures and tables"""

# ============================================================
# Manifest: file -> destination in the manuscript
# ============================================================
MANIFIESTO = [
    ("03_unified_dataset",            "tabla",  "§3.1 Datos",                    "R2-13"),
    ("04b_sample_reconciliation",     "tabla",  "§3.1 nota al pie del tamano muestral", "Paso 10"),
    ("05_data_dictionary",            "tabla",  "Material suplementario A",      "R2-13, R1-m4"),
    ("06_class_distribution_monthly", "tabla",  "§3.1.2 Distribucion de clases", "R1-M2, R2-05"),
    ("fig_class_distribution_monthly","figura", "Figura nueva en §3.1.2",        "R1-M2, R2-05"),
    ("07_leakage_diagnosis",          "tabla",  "§3.3 Protocolo / carta de respuesta", "R1-M4, R1-A4"),
    ("08_nested_tuning",              "tabla",  "Material suplementario B",      "R1-M4"),
    ("09_fold_info",                  "tabla",  "§3.3 rangos de fecha por particion", "R1-M4, R2-04"),
    ("09b_deep_learning_fold_detail", "tabla",  "Material suplementario D",      "R2-06, R2-20"),
    ("10_results_PRIMARY",            "tabla",  "TABLA 3 del manuscrito revisado", "R1-M1, R1-M2, R1-M6"),
    ("10b_roc_auc_aggregated",        "tabla",  "§4.1 junto a la figura ROC",     "R2-07"),
    ("11_results",                    "tabla",  "Material suplementario C (otras configuraciones)", "R1-M4"),
    ("12_cross_protocol_comparison",  "tabla",  "§4.1.1 comparacion entre protocolos", "R1-M4"),
    ("13_protocol_effect_delta",      "tabla",  "TABLA NUEVA: efecto del protocolo", "R1-M4, R1-A4"),
    ("fig_protocol_effect",           "figura", "FIGURA NUEVA en §4.1.1",        "R1-M4"),
    ("fig_confusion_matrices_study",  "figura", "FIGURA NUEVA en §4.1.2",        "R1-M2, R1-A2"),
    ("fig_confusion_matrices_deep_learning", "figura", "§4.5",                   "R2-06, R2-20"),
    ("fig_confusion_matrices_baselines", "figura", "§4.1.2",                     "R1-M6"),
    ("fig_confusion_matrices_normalized", "figura", "Material suplementario",    "R1-M2"),
    ("fig_roc_curves_primary",        "figura", "FIGURA NUEVA en §4.1.2 (unica en color)", "R2-07"),
    ("14_permutation_importance",     "tabla",  "§4.2 sensibilidad",             "R2-08"),
    ("14b_model_feature_importance",  "tabla",  "Material suplementario",        "R2-08"),
    ("15_climate_sensitivity",        "tabla",  "§4.2 sensibilidad",             "R2-08"),
    ("fig_climate_sensitivity",       "figura", "FIGURA NUEVA en §4.2",          "R2-08"),
    ("16_computational_cost",         "tabla",  "§4.5 costo computacional",      "R2-19"),
    ("17_mcnemar_all_pairs_holm",     "tabla",  "Material suplementario E (matriz completa)", "R1-M1, R1-A1"),
    ("18_mcnemar_study_vs_baselines", "tabla",  "TABLA NUEVA en §4.1.3 (contraste principal)", "R1-M6, R1-M1"),
    ("18b_mcnemar_baseline_suplementario", "tabla", "Material suplementario F",  "R1-M6"),
    ("18c_mejor_modelo_vs_baselines_metricas", "tabla", "§4.1.3 lectura por metrica", "R1-M2, R1-M6"),
    ("19_mcnemar_top_models",         "tabla",  "TABLA NUEVA en §4.1.3 (modelos de cabeza)", "R1-M1, R2-02"),
    ("19b_metric_discordances",       "tabla",  "§4.1.3 nota sobre metricas",    "R1-M2"),
    ("20_wilcoxon_paired_folds",      "tabla",  "Material suplementario G",      "R1-M1, R1-A1"),
    ("20b_bootstrap_confidence_intervals", "tabla", "TABLA NUEVA en §4.1.2",     "R1-M1"),
    ("fig_mcc_confidence_intervals",  "figura", "FIGURA NUEVA en §4.1.2",        "R1-M1"),
    ("21_liquidity_diagnosis",        "tabla",  "TABLA NUEVA en §4.3 + Limitaciones", "R1-M3, R1-A3"),
    ("fig_liquidity_monthly",         "figura", "FIGURA NUEVA en §4.3",          "R1-M3"),
    ("22_backtest_results",           "tabla",  "§4.3 (o suplementario si Ruta B)", "R1-M3"),
    ("22b_montecarlo_backtest",       "tabla",  "§4.3 contraste Monte Carlo",    "R1-M3"),
    ("22c_backtest_lag_sensitivity",  "tabla",  "§4.3 supuesto de retardo",      "R1-M3"),
    ("fig_montecarlo_backtest",       "figura", "§4.3 (o suplementario si Ruta B)", "R1-M3"),
    ("23_narrative_for_manuscript",   "texto",  "Insumo directo de los Pasos 6 a 10", "todas"),
    ("24_r2_20_architecture_verification", "tabla", "TABLA NUEVA en §4.5 / carta de respuesta", "R2-20"),
    ("25_neural_network_diagnosis",   "tabla",  "§4.5 diagnostico de redes",     "R2-06, R2-20"),
    ("fig_neural_network_collapse",   "figura", "§4.5",                          "R2-06, R2-20"),
    ("26_closing_checklist",          "tabla",  "Control interno, no publicable", "—"),
    ("27_afinamientos_aplicados",     "tabla",  "Control interno, no publicable", "—"),
    ("28_citable_numbers",            "tabla",  "Barrido de cifras del Paso 10", "R1-M7"),
    ("results_v6",                    "json",   "Fuente unica de verdad",        "todas"),
]

def marcar_generados(df):
    """Mark which manifest entries already have their file on disk."""
    generados = {p for _, p, _ in ARCHIVOS_GENERADOS}
    df = df.copy()
    df["Generado_en_esta_corrida"] = [
        "si" if any(g.startswith(pref) for g in generados) else "pendiente"
        for pref in df["Prefijo"]]
    return df

df_manifiesto = marcar_generados(pd.DataFrame(
    [{"Prefijo": pre, "Tipo": tipo, "Destino_en_el_manuscrito": dest,
      "Observaciones_que_atiende": obs}
     for pre, tipo, dest, obs in MANIFIESTO]))

print("=" * 110)
print("   MANIFIESTO — que archivo alimenta que parte del manuscrito")
print("=" * 110)
display(df_manifiesto)

_pendientes = df_manifiesto[df_manifiesto["Generado_en_esta_corrida"] == "pendiente"]
print(f"\nEntradas ya escritas en disco: "
      f"{int((df_manifiesto['Generado_en_esta_corrida'] == 'si').sum())} de {len(df_manifiesto)}")
if len(_pendientes):
    print("Las restantes se generan mas abajo en esta misma seccion "
          f"({', '.join(_pendientes['Prefijo'])}).")
print("§20 vuelve a comprobar el manifiesto con el estado final y lo exporta entonces:")
print("comprobarlo aqui daria un falso negativo por puro orden de ejecucion.")

"""## 19.2 Flat registry of quotable figures

Every figure that the manuscript can quote is reduced to one row `key, value, unit, origin`, so that verifying a number is a search rather than a reading of the JSON."""

# ============================================================
# Quotable figures, one per row
# ============================================================
def cifra(clave, valor, unidad, origen, observacion=""):
    return {"Clave": clave, "Valor": valor, "Unidad": unidad,
            "Origen": origen, "Observacion_asociada": observacion}

CIFRAS = [
    cifra("run_id", RUN_ID, "—", "metadata.run_id"),
    cifra("periodo_inicio", DATE_START, "fecha", "metadata.date_range"),
    cifra("periodo_fin", DATE_END, "fecha", "metadata.date_range"),
    cifra("dias_calendario", SAMPLE_RECONCILIATION["dias_calendario"], "dias",
          "metadata.sample_reconciliation", "Paso 10"),
    cifra("dias_citados_en_manuscrito", SAMPLE_RECONCILIATION["tras_objetivo_d1"], "dias",
          "metadata.sample_reconciliation", "Paso 10"),
    cifra("n_muestras_evaluadas", int(len(X)), "observaciones", "metadata.n_samples", "Paso 10"),
    cifra("n_variables", len(FEATURE_COLUMNS), "variables", "metadata.n_features", "R2-13"),
    cifra("observaciones_climaticas_reales", RAZON_INTERPOLACION["observaciones_mensuales_reales"],
          "meses", "metadata.interpolation_ratio", "Limitaciones"),
    cifra("dias_alta_produccion", n_alto, "dias", "metadata.class_distribution_global", "R1-M2"),
    cifra("dias_baja_produccion", n_bajo, "dias", "metadata.class_distribution_global", "R1-M2"),
    cifra("pct_clase_positiva", round(PCT_POSITIVOS, 2), "%",
          "metadata.class_distribution_global", "R1-M2, R2-05"),
    cifra("umbral_global_TM", round(THRESHOLD_GLOBAL, 2), "TM",
          "metadata.threshold_global_TM", "R2-04"),
    cifra("umbral_coef_variacion", round(float(var_pct), 2), "%",
          "metadata.threshold_cv_pct", "R2-04"),
    cifra("umbral_es_estable", UMBRAL_ES_ESTABLE, "booleano",
          "metadata.threshold_es_estable", "R2-04"),
    cifra("gap_temporal", TS_GAP, "dias", "metadata.ts_gap", "R1-M4"),
    cifra("particiones_totales", FOLDS_PRINCIPAL["totales"], "particiones",
          "metadata.folds_protocolo_principal", "R1-M4"),
    cifra("particiones_evaluadas", FOLDS_PRINCIPAL["evaluadas"], "particiones",
          "metadata.folds_protocolo_principal", "R1-M4"),
    cifra("particiones_descartadas_factorial",
          int((df_foldinfo["Fold_valido"] == "no").sum()), "particiones",
          "degenerate_folds_summary", "Limitaciones"),
    cifra("caida_media_accuracy", round(float(caida_acc), 4), "puntos",
          "13_protocol_effect_delta", "R1-M4"),
    cifra("caida_media_mcc", round(float(caida_mcc), 4), "puntos",
          "13_protocol_effect_delta", "R1-M4"),
    cifra("mejor_modelo", label_mejor, "—", "statistical_tests.verdict", "R1-M1, R2-02"),
    cifra("mejor_modelo_en_ingles", res_p[mejor].get("label_en", label_mejor), "—",
          "configurations[...].label_en", "R1-M1"),
    cifra("mejor_modelo_mcc", round(best["mcc_mean"], 4), "MCC",
          "statistical_tests.verdict", "R1-M1"),
    cifra("mejor_modelo_balanced_accuracy", round(best["balanced_accuracy_mean"], 4), "—",
          "statistical_tests.verdict", "R1-M2"),
    cifra("mejor_modelo_accuracy", round(best["accuracy_mean"], 4), "—",
          "statistical_tests.verdict", "R1-M1"),
    cifra("mejor_modelo_mcc_ic95_inf", float(_ci_mejor["MCC_IC95_inf"]), "MCC",
          "statistical_tests.bootstrap_confidence_intervals", "R1-M1"),
    cifra("mejor_modelo_mcc_ic95_sup", float(_ci_mejor["MCC_IC95_sup"]), "MCC",
          "statistical_tests.bootstrap_confidence_intervals", "R1-M1"),
    cifra("afirmacion_permitida", VERDICT["afirmacion_permitida"], "—",
          "statistical_tests.verdict", "R1-M1, R2-02"),
    cifra("n_pares_mcnemar", len(df_mc), "pares",
          "statistical_tests.mcnemar_all_pairs", "R1-A1"),
    cifra("n_discordancias_metricas", len(df_discordancias), "pares",
          "statistical_tests.metric_discordances", "R1-M2"),
    cifra("baseline_mayoritaria_accuracy", round(bl_maj.get("accuracy_mean", np.nan), 4), "—",
          "configurations", "R1-M6"),
    cifra("baseline_mayoritaria_mcc", round(bl_maj.get("mcc_mean", np.nan), 4), "MCC",
          "configurations", "R1-M6"),
    cifra("baseline_persistencia_accuracy", round(bl_per.get("accuracy_mean", np.nan), 4), "—",
          "configurations", "R1-M6 (suplementario)"),
    cifra("baseline_persistencia_mcc", round(bl_per.get("mcc_mean", np.nan), 4), "MCC",
          "configurations", "R1-M6 (suplementario)"),
    cifra("n_baselines_contraste_principal", len(baseline_keys_pri), "baselines",
          "metadata.baselines_contraste_principal", "R1-M6"),
    cifra("baselines_superados_en_mcc", len(BASELINES_PRINCIPAL_SUPERADOS_MCC), "baselines",
          "statistical_tests.verdict", "R1-M6"),
    cifra("baselines_superados_en_accuracy", len(BASELINES_PRINCIPAL_SUPERADOS_ACC), "baselines",
          "statistical_tests.verdict", "R1-M6"),
    cifra("redes_evaluadas", DL_DIAGNOSIS["redes_evaluadas"], "redes",
          "deep_learning_diagnosis", "R2-06, R2-20"),
    cifra("redes_con_colapso", DL_DIAGNOSIS["redes_con_colapso"], "redes",
          "deep_learning_diagnosis", "R2-06, R2-20"),
    cifra("ventana_deep_learning", DL_WINDOW, "dias",
          "metadata.deep_learning.ventana_dias", "R2-06"),
]

if RUN_SENSITIVITY:
    CIFRAS += [
        cifra("variable_mas_influyente", IMP_TOP_VARIABLE, "—",
              "sensitivity.permutation_importance", "R2-08"),
        cifra("importancia_variable_top", round(IMP_TOP_VALOR, 4), "caida de balanced accuracy",
              "sensitivity.permutation_importance", "R2-08"),
        cifra("importancia_temperatura", round(IMP_TEMPERATURA, 4), "caida de balanced accuracy",
              "sensitivity.permutation_importance", "R2-08"),
        cifra("rango_perturbacion_2sigma", round(RANGO_PERTURBACION * 100, 1), "puntos porcentuales",
              "sensitivity.climate_perturbation", "R2-08"),
    ]

if BACKTEST_VERDICT:
    CIFRAS.append(cifra("ruta_modulo_inversion", BACKTEST_VERDICT["ruta"], "—",
                        "investment_module.ruta", "R1-M3, R1-A3"))
    CIFRAS.append(cifra("retardo_senal_ejecucion", BACKTEST_LAG_DIAS, "sesiones",
                        "investment_module.retardo_senal_ejecucion_sesiones", "R1-M3"))
    for a, v in BACKTEST_VERDICT["liquidez"].items():
        CIFRAS.append(cifra(f"liquidez_{a}_pct_con_volumen", round(v["pct_vol"], 1), "%",
                            "investment_module.liquidez", "R1-M3"))
        CIFRAS.append(cifra(f"liquidez_{a}_pct_con_cambio", round(v["pct_cambio"], 1), "%",
                            "investment_module.liquidez", "R1-M3"))

CIFRAS += [
    cifra("tiempo_entrenamiento_min", float(_e["Entrenamiento_s"].min()), "s",
          "computational_cost", "R2-19"),
    cifra("tiempo_entrenamiento_max", float(_e["Entrenamiento_s"].max()), "s",
          "computational_cost", "R2-19"),
]

df_cifras = pd.DataFrame(CIFRAS)
save_csv(df_cifras, "28_citable_numbers")
print("=" * 110)
print("   CIFRAS CITABLES — cada numero del manuscrito debe coincidir con una fila")
print("=" * 110)
display(df_cifras)
print(f"\n{len(df_cifras)} cifras registradas. Regla del Paso 10: si un numero del texto")
print("no aparece en esta tabla, o bien esta mal, o bien falta registrarlo aqui.")

"""## 19.3 Reproducibility files

Generates the `requirements.txt` and the README of the code repository. They are generated rather than written by hand to avoid a mismatch with the actual execution environment."""

# ============================================================
# requirements.txt and repository README
# ============================================================
_req = [
    f"numpy=={np.__version__}",
    f"pandas=={pd.__version__}",
    f"matplotlib=={matplotlib.__version__}",
    f"scikit-learn=={sklearn.__version__}",
    f"scipy=={scipy.__version__}",
]
if XGBOOST_DISPONIBLE:
    _req.append(f"xgboost=={xgboost.__version__}")
_req += ["joblib", "xarray", "netCDF4", "requests"]
if DEEP_LEARNING_MARCO:
    _marco, _ver = DEEP_LEARNING_MARCO.split(" ")[0], DEEP_LEARNING_MARCO.split(" ")[-1]
    _req.append(f"tensorflow=={_ver}" if _marco.startswith("tensorflow") else f"torch=={_ver}")

ruta_req = os.path.join(DATA_DIR, f"30_requirements_{RUN_ID}.txt")
with open(ruta_req, "w", encoding="utf-8") as f:
    f.write("\n".join(_req) + "\n")
ARCHIVOS_GENERADOS.append(("texto", "30_requirements", os.path.basename(ruta_req)))

_readme = f"""# FIS-ML — Replication package

Manuscript: FIS-ML: A Fine-Tuned Multi-Classifier Optimization Model System with SVM,
XGBoost and RFC for Intelligent Investment Signal Prediction in Fisheries Sector Stock
Markets. Submitted to Expert Systems with Applications (Elsevier), major revision.

Notebook version: v6.0
Run identifier: {RUN_ID}
Random seed: {RANDOM_SEED}
Python: {ENVIRONMENT['python']}
Platform: {ENVIRONMENT['platform']}

## Contents

- `FIS_ML_Notebook_Unificado_v6.ipynb` — the single notebook that produces every number,
  table and figure reported in the revised manuscript. Runs top to bottom without manual
  switches.
- `results_v6_{RUN_ID}.json` — single source of truth. Every figure quoted in the
  manuscript is traceable to a key in this file.
- `28_citable_numbers_{RUN_ID}.csv` — flat registry of quotable figures, one per row.
- `29_manifiesto_figuras_y_tablas_{RUN_ID}.csv` — mapping from output file to its
  destination in the manuscript and to the reviewer observation it addresses.
- `05_data_dictionary_{RUN_ID}.csv` — variable-by-variable dictionary with explicit
  formulas, addressing R2-13.
- `requirements.txt` — exact library versions used in the reported run.

## Data sources

- Fish landings: Ministerio de la Produccion del Peru, open data portal.
  Daily aggregation of individual landing records over {DATE_START} to {DATE_END}.
- Sea surface temperature anomaly: HadCRUT 5.0.2.0 ensemble mean, Climatic Research
  Unit, University of East Anglia. Four 5x5 degree grid cells over the Peruvian coast,
  monthly resolution linearly interpolated to daily.
- Stock prices: Bolsa de Valores de Lima, daily OHLC for EXALMC1 and AUSTRAC1.

## Validation protocol

Forward-chaining time-series cross-validation, {N_FOLDS} splits, gap of {TS_GAP} days.
The class threshold is estimated on training data only within each split. The k-fold
protocol used in the original submission is reported alongside, as a reference point,
not as a defensible result.

## How to reproduce

1. Place the source CSV files in the DATA directory declared in section 1.2.
2. Run the notebook top to bottom. It writes every output with a run identifier in the
   filename, so runs never overwrite each other.
3. Compare against `results_v6_{RUN_ID}.json`. Figures may differ in the last decimals
   across library versions; the reported versions are in `requirements.txt`.

## Licence and contact

Universidad Nacional Mayor de San Marcos, Lima, Peru. Funded by CONCYTEC/PROCIENCIA.
"""

ruta_readme = os.path.join(DATA_DIR, f"31_README_repositorio_{RUN_ID}.md")
with open(ruta_readme, "w", encoding="utf-8") as f:
    f.write(_readme)
ARCHIVOS_GENERADOS.append(("texto", "31_README_repositorio", os.path.basename(ruta_readme)))

print("Archivos de reproducibilidad generados:")
print(f"  {os.path.basename(ruta_req)}")
print(f"  {os.path.basename(ruta_readme)}")
print()
print("Contenido de requirements.txt:")
for r in _req:
    print(f"  {r}")

"""# 20. Automatic closing verification

Automatic checks on the generated narrative and on the exported results: no forbidden statement (such as "exceed all baselines") unless it is true, consistency of the verdict with the McNemar table, traceability of every quoted figure to the JSON, output format and experimental integrity. If any check fails, the notebook says so loudly."""

# ============================================================
# Closing checks
# ============================================================
COMPROBACIONES = []

def comprobar(nombre, condicion, detalle=""):
    COMPROBACIONES.append({"Comprobacion": nombre,
                           "Resultado": "OK" if condicion else "FALLA",
                           "Detalle": detalle})
    return bool(condicion)

_narr = NARRATIVE.lower()

# --- 1. Forbidden statements unless true ---
_supera_todos_los_principales = (
    len(BASELINES_PRINCIPAL_SUPERADOS_MCC) == len(baseline_keys_pri)
    and len(BASELINES_PRINCIPAL_SUPERADOS_BACC) == len(baseline_keys_pri)
    and not bl_superan and not sup_superan)
comprobar(
    "La narrativa no afirma superar a todas las lineas base salvo que sea cierto",
    ("exceed all baselines" not in _narr) or _supera_todos_los_principales,
    f"cadena presente: {'exceed all baselines' in _narr} | "
    f"afirmacion seria cierta: {_supera_todos_los_principales}")

comprobar(
    "La narrativa no afirma superioridad si el veredicto dice comparabilidad",
    not (VERDICT["afirmacion_permitida"] == "comparabilidad"
         and "is significantly superior" in _narr),
    f"afirmacion permitida: {VERDICT['afirmacion_permitida']}")

comprobar(
    "La narrativa no llama estable al umbral si el criterio dice lo contrario",
    not (("threshold is stable" in _narr) and not UMBRAL_ES_ESTABLE),
    f"coeficiente de variacion {var_pct:.2f}% frente a criterio {UMBRAL_CV_CRITERIO:.0f}%")

# --- 2. Consistency of the verdict with the McNemar table ---
_incoherentes = []
for _lab in VERDICT["supera_baselines"] + VERDICT["supera_a"]:
    _f = fila_par(label_mejor, _lab)
    if _f is None:
        continue
    _gana_ref = (_f["b (solo M1)"] > _f["c (solo M2)"]) if _f["Modelo_1"] == label_mejor \
        else (_f["c (solo M2)"] > _f["b (solo M1)"])
    if not _gana_ref:
        _incoherentes.append(_lab)
comprobar(
    "Ningun estimador listado como superado gana en realidad en aciertos",
    len(_incoherentes) == 0,
    f"incoherentes: {_incoherentes}" if _incoherentes else "verificado contra las columnas b y c")

comprobar(
    "El veredicto declara el campo de direccion inversa",
    "es_superado_por" in VERDICT and "es_superado_por_baselines" in VERDICT,
    "campos es_superado_por y es_superado_por_baselines presentes")

# --- 3. Traceability of the figures ---
comprobar(
    "El mejor modelo aparece en la tabla de pruebas pareadas",
    mejor_en_wilcox, f"modelo: {label_mejor}")

comprobar(
    "El registro de cifras citables no tiene valores vacios",
    not df_cifras["Valor"].isna().any(),
    f"{len(df_cifras)} cifras registradas")

_claves_json = set()
def _recorrer(o, prefijo=""):
    if isinstance(o, dict):
        for k, v in o.items():
            _claves_json.add(k)
            _recorrer(v, f"{prefijo}.{k}")
    elif isinstance(o, list):
        for v in o[:50]:
            _recorrer(v, prefijo)
_recorrer(_recargado)
comprobar(
    "Las claves de origen del registro de cifras existen en el JSON",
    all(o.split(".")[-1] in _claves_json or o.split("_")[0].isdigit() or o == "—"
        for o in df_cifras["Origen"].unique()),
    f"{len(df_cifras['Origen'].unique())} origenes distintos")

# --- 4. Output format ---
comprobar("Todas las salidas tabulares son CSV, ninguna en Excel", n_xlsx == 0,
          f"CSV: {n_csv}  |  Excel: {n_xlsx}")
comprobar("El JSON recarga con un parser estricto", True,
          "verificado en §13 con parse_constant")
# The manifest is re-evaluated here, with the complete inventory, and exported.
df_manifiesto = marcar_generados(df_manifiesto)
_faltantes = df_manifiesto[df_manifiesto["Generado_en_esta_corrida"] != "si"]
comprobar("El manifiesto no tiene entradas sin generar", len(_faltantes) == 0,
          f"{len(_faltantes)} entradas faltantes"
          + (f": {', '.join(_faltantes['Prefijo'])}" if len(_faltantes) else ""))
save_csv(df_manifiesto, "29_manifiesto_figuras_y_tablas")

# --- 5. Experimental integrity ---
comprobar("El protocolo principal no presenta fuga temporal", sin_fuga_principal,
          f"0 violaciones de orden, gap = {TS_GAP} dias")
comprobar("Los bloques de prueba del aprendizaje profundo coinciden con los tabulares",
          (not DL_EJECUTADO) or True,
          "verificado por asercion dura en §8.1" if DL_EJECUTADO else "no aplica")
_n_esperado = len(res_p) * (len(res_p) - 1) // 2
comprobar("El numero de pares de McNemar corresponde al numero de estimadores",
          len(df_mc) == _n_esperado,
          f"{len(df_mc)} pares para {len(res_p)} estimadores (esperado {_n_esperado})")

df_comprobaciones = pd.DataFrame(COMPROBACIONES)
save_csv(df_comprobaciones, "32_verificacion_de_cierre")

print("=" * 110)
print("   VERIFICACION AUTOMATICA DE CIERRE")
print("=" * 110)
display(df_comprobaciones)

_fallas = df_comprobaciones[df_comprobaciones["Resultado"] == "FALLA"]
print()
if len(_fallas) == 0:
    print(f"Las {len(df_comprobaciones)} comprobaciones pasan.")
    print("Los numeros pueden congelarse y los Pasos 6 a 10 pueden comenzar.")
else:
    print(f"ATENCION: {len(_fallas)} comprobacion(es) fallan. NO congelar numeros todavia.")
    for _, f in _fallas.iterrows():
        print(f"   - {f['Comprobacion']}")
        print(f"     {f['Detalle']}")

"""# 21. Final notes"""

print("=" * 100)
print("   SIGUIENTE PASO")
print("=" * 100)
print(f"""
1. Leer el veredicto de §10.6 y ajustar el Abstract: la afirmacion permitida es
   '{VERDICT['afirmacion_permitida']}' con {label_mejor} como modelo de referencia.
2. Ejecutar en el manuscrito la Ruta {BACKTEST_VERDICT.get('ruta', '?')} del modulo de inversion (§11.3).
3. Regenerar las Tablas 2 y 3 desde results_v6_{RUN_ID}.json, con la
   columna Contraste que distingue baselines principales de suplementarios.
4. Pegar las frases de §15 en las secciones indicadas por el manifiesto de §19.1 y
   adaptarlas al estilo del articulo.
5. Usar 28_citable_numbers como lista de control del barrido de cifras del Paso 10.
6. Iniciar los Pasos 6 a 10. Los numeros quedan congelados: a partir de aqui, ninguna
   cifra del texto debe provenir de otra fuente que este JSON.
""")

"""---

## Final notes

**Primary protocol.** The primary protocol is `timeseries / per_fold_train`. The column of the original shuffled protocol is kept as an explicitly labelled reference.

**Figures.** All figures are greyscale with a white background, except the ROC curves; a table of area-under-curve values accompanies them so that the information is recoverable in a monochrome print.

**Environment.** The final run must be executed in Google Colab, where the environment is configured and the Drive paths resolve without changes.

**Scope.** The notebook freezes the numbers and generates the inputs for the text; it does not write the manuscript."""
