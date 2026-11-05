# -*- coding: utf-8 -*-

"""# FIS-ML — Regeneration of the figures with English labels (no retraining)

This notebook does NOT train anything and does NOT change any result. It reads the CSV files and the JSON file of the frozen run `20260903_19h_54m_23s`, stored in `DATA/`, and redraws the figures with English labels. The figures are saved in `IMAGES/` with the suffix `_EN.png`.

Run top to bottom in Colab (Runtime → Run all). If a CSV file is missing, the corresponding figure is skipped with a warning and the others are generated anyway.

The only figure that is not rebuilt is the set of ROC curves, because it needs the per-observation probabilities, which the run did not export. The ROC-AUC values are reported in Table 3 of the manuscript."""

"""## 1. Paths and constants"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from google.colab import drive
    drive.mount("/content/drive")
    EN_COLAB = True
except Exception:
    EN_COLAB = False
    print("Entorno local (no Colab).")

# --- Same paths as the unified notebook v7 ---
BASE_DIR   = "/content/drive/My Drive/Colab Notebooks/2024-2.SVM.Pesca"
if not EN_COLAB and not os.path.isdir(BASE_DIR):
    BASE_DIR = os.getcwd()          # for local tests
DATA_DIR   = os.path.join(BASE_DIR, "DATA")
IMAGES_DIR = os.path.join(BASE_DIR, "IMAGES")
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- Frozen run: set RUN_ID to use the files of another run ---
RUN_ID = "20260903_19h_54m_23s"
SUFFIX = "_EN"                      # suffix of the regenerated figures

DATE_START, DATE_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-03-14")

print("DATA:  ", DATA_DIR)
print("IMAGES:", IMAGES_DIR)

"""## 2. Monochrome style (same criterion as the run: white background, hatching, font >= 10 pt)"""

GREYS  = ["#ffffff", "#e6e6e6", "#cccccc", "#b3b3b3", "#999999", "#808080", "#666666", "#4d4d4d"]
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "|||", "---", "ooo"]

def grey(i):   return GREYS[i % len(GREYS)]
def hatch(i):  return HATCHES[i % len(HATCHES)]

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.edgecolor": "black",
})

def style_axis(ax, grid="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, color="#cccccc", linewidth=0.6)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, color="#cccccc", linewidth=0.6)
    ax.set_axisbelow(True)

def save_figure(fig, prefix, dpi=300):
    fpath = os.path.join(IMAGES_DIR, f"{prefix}{SUFFIX}.png")
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"   Figure saved: {os.path.basename(fpath)}")
    plt.close(fig)
    return fpath

def csv_path(prefix):
    return os.path.join(DATA_DIR, f"{prefix}_{RUN_ID}.csv")

def read_csv_or_none(prefix):
    p = csv_path(prefix)
    if not os.path.exists(p):
        print(f"   [!] Missing: {os.path.basename(p)} -> figure skipped")
        return None
    return pd.read_csv(p, encoding="utf-8-sig")

"""## 3. Label dictionary: Spanish (run) → English (manuscript)"""

LABEL_EN = {
    "SVM (RBF, C=100)":                  "SVM (RBF, C = 100)",
    "Gradient Boosting Tree":            "Gradient Boosting Tree",
    "XGBoost":                           "XGBoost",
    "Random Forest Classifier":          "Random Forest Classifier",
    "Decision Tree Classifier":          "Decision Tree Classifier",
    "Logistic Regression":               "Logistic Regression",
    "Naive Bayes":                       "Gaussian Naive Bayes",
    "MLP (Neural Network)":              "Multi-Layer Perceptron",
    "Baseline: Clase Mayoritaria":       "Majority-class baseline",
    "Baseline: Estacional Ingenuo":      "Seasonal-naive baseline",
    "Baseline: Estratificado Aleatorio": "Stratified-random baseline",
    "Baseline: Persistencia (D+1 = D)":  None,      # not plotted in any figure
    "1D-CNN (Deep Learning)":            "1D-CNN",
    "LSTM (Deep Learning)":              "LSTM",
    "GRU (Deep Learning)":               "GRU",
}
KEY_EN = {  # keys of the JSON
    "SVM_RBF_C100": "SVM (RBF, C = 100)", "GBT": "Gradient Boosting Tree", "XGBoost": "XGBoost",
    "RFC": "Random Forest Classifier", "DTC": "Decision Tree Classifier", "LogReg": "Logistic Regression",
    "NaiveBayes": "Gaussian Naive Bayes", "MLP": "Multi-Layer Perceptron",
    "BL_Majority": "Majority-class baseline", "BL_Seasonal": "Seasonal-naive baseline",
    "BL_Stratified": "Stratified-random baseline", "BL_Persistence": None,
    "DL_CNN1D": "1D-CNN", "DL_LSTM": "LSTM", "DL_GRU": "GRU",
}
EXCLUDED = {"Baseline: Persistencia (D+1 = D)", "BL_Persistence"}

def en(label):
    """Translate a label; return None if the estimator is not plotted."""
    if label in LABEL_EN:
        return LABEL_EN[label]
    return label

"""## 4. JSON of the run (confusion matrices)"""

JSON_PATH = os.path.join(DATA_DIR, f"results_v6_{RUN_ID}.json")
with open(JSON_PATH, encoding="utf-8") as fh:
    RESULTS = json.load(fh)
PRIMARY = RESULTS["configurations"]["timeseries|per_fold_train"]
print(f"Estimators in the primary protocol: {len(PRIMARY)}")

"""## 5. Figure 6 — Monthly share of High-production days (`06_class_distribution_monthly`)"""

df = read_csv_or_none("06_class_distribution_monthly")
if df is not None:
    pct_global = 100 * df["Altos"].sum() / df["Dias"].sum()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(df["YearMonth"], df["Pct_Alto"], color=grey(3), edgecolor="black",
           linewidth=0.7, hatch=hatch(2))
    ax.axhline(pct_global, color="black", linestyle="--", linewidth=1.4,
               label=f"Overall share ({pct_global:.1f}%)")
    ax.set_ylabel("High-production days (%)")
    ax.set_xlabel("Month")
    ax.set_title("Monthly share of High-production days")
    ax.legend(frameon=True, edgecolor="black")
    style_axis(ax, "y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_figure(fig, "fig_class_distribution_monthly")

"""## 6. Figure 8 — Effect of the validation scheme (`13_protocol_effect_delta`, study models only)"""

df = read_csv_or_none("13_protocol_effect_delta")
if df is not None:
    d = df[df["Tipo"] == "study"].copy()
    d["Label"] = d["Estimador"].map(en)
    d = d.sort_values("MCC_temporal")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    panels = [("Acc_kfold", "Acc_temporal", "Accuracy"),
              ("BalAcc_kfold", "BalAcc_temporal", "Balanced accuracy"),
              ("MCC_kfold", "MCC_temporal", "MCC")]
    ypos = np.arange(len(d))
    for ax, (c_old, c_new, title) in zip(axes, panels):
        ax.barh(ypos - 0.2, d[c_old], height=0.4, color=grey(1), edgecolor="black",
                linewidth=0.7, hatch=hatch(1), label="Shuffled 5-fold (cycles 1–3)")
        ax.barh(ypos + 0.2, d[c_new], height=0.4, color=grey(5), edgecolor="black",
                linewidth=0.7, label="Forward-chaining temporal (cycle 4)")
        ax.set_yticks(ypos); ax.set_yticklabels(d["Label"])
        ax.set_title(title)
        ax.axvline(0, color="black", linewidth=0.8)
        style_axis(ax, "x")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, edgecolor="black",
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    save_figure(fig, "fig_protocol_effect")

"""## 7. Figure 9 — Model-specific feature importance (`14b_model_feature_importance`)"""

df = read_csv_or_none("14b_model_feature_importance")
if df is not None:
    models = [m for m in df["Modelo"].unique() if m not in EXCLUDED]
    ncols = min(3, len(models)); nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, m in enumerate(models):
        sub = df[df["Modelo"] == m].sort_values("Importancia", ascending=False)
        axes[i].barh(range(len(sub)), sub["Importancia"], color=grey(4),
                     edgecolor="black", linewidth=0.6)
        axes[i].set_yticks(range(len(sub))); axes[i].set_yticklabels(sub["Variable"])
        axes[i].invert_yaxis()
        axes[i].set_title(en(m)); axes[i].set_xlabel("Importance")
        style_axis(axes[i], "x")
    for j in range(len(models), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    save_figure(fig, "fig_feature_importance_primary")

"""## 8. Supplementary — Permutation importance of the reference classifier (`14_permutation_importance`)"""

df = read_csv_or_none("14_permutation_importance")
if df is not None:
    d = df.sort_values("Importancia", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(d["Variable"], d["Importancia"], xerr=d["Desv_est"], color=grey(3),
            edgecolor="black", linewidth=0.6, ecolor="black", capsize=3)
    ax.set_xlabel("Drop in balanced accuracy when the variable is permuted")
    ax.set_title("Permutation importance — Logistic Regression, last temporal split")
    ax.axvline(0, color="black", linewidth=0.8)
    style_axis(ax, "x")
    plt.tight_layout()
    save_figure(fig, "fig_permutation_importance")

"""## 9. Figure 10 — Directed perturbation of the temperature anomaly (`15_climate_sensitivity`)"""

df = read_csv_or_none("15_climate_sensitivity")
if df is not None:
    d = df.sort_values("Perturbacion_sigma")
    fig, ax1 = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(d))
    ax1.bar(x, 100 * d["Tasa_predicciones_Alto"], color=grey(3), edgecolor="black",
            linewidth=0.7, hatch=hatch(2), label="Rate of High predictions (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s:+.1f}σ\n({t:+.2f} °C)" for s, t in
                         zip(d["Perturbacion_sigma"], d["Delta_Temperatura_C"])])
    ax1.set_xlabel("Shift applied to the temperature anomaly")
    ax1.set_ylabel("High predictions (%)")
    ax1.set_ylim(0, 105)
    ax2 = ax1.twinx()
    ax2.plot(x, d["Balanced_Acc"], color="black", marker="o", linewidth=1.4,
             label="Balanced accuracy")
    ax2.set_ylabel("Balanced accuracy")
    ax2.set_ylim(0, 1.05)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, edgecolor="black")
    ax1.set_title("Sensitivity of the reference classifier to the temperature anomaly")
    style_axis(ax1, "y")
    ax2.spines["top"].set_visible(False)
    plt.tight_layout()
    save_figure(fig, "fig_climate_sensitivity")

"""## 10. Confusion matrices (JSON, pooled over the evaluated splits) — letter figures"""

def confusion_panel(ax, cm, title="", normalize=False):
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    ax.imshow(np.ones_like(shown), cmap="gray", vmin=0, vmax=1)   # white cells
    labels = ("Low", "High")
    for i in range(2):
        for j in range(2):
            txt = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11, color="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("black")
    ax.axhline(0.5, color="black", linewidth=0.8); ax.axvline(0.5, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=10)

def panel_matrices(keys, prefix, heading, normalize=False):
    keys = [k for k in keys if k not in EXCLUDED and KEY_EN.get(k)]
    if not keys:
        return
    ncols = min(4, len(keys)); nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 4.1 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, k in enumerate(keys):
        r = PRIMARY[k]
        confusion_panel(axes[i], r["confusion_matrix"],
                        title=f"{KEY_EN[k]}\nMCC = {r['mcc_mean']:+.3f}  BalAcc = {r['balanced_accuracy_mean']:.3f}",
                        normalize=normalize)
    for j in range(len(keys), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(heading, fontsize=12, y=1.01)
    plt.tight_layout(h_pad=2.5)
    save_figure(fig, prefix)

keys_study = [k for k, v in PRIMARY.items() if v["category"] == "study"]
keys_bl    = [k for k, v in PRIMARY.items() if v["category"] == "baseline"]
keys_dl    = [k for k, v in PRIMARY.items() if v["category"] == "deep_learning"]

panel_matrices(keys_study, "fig_confusion_matrices_study",
               "Confusion matrices — classifiers of the study (pooled test observations, n = 288)")
panel_matrices(keys_bl, "fig_confusion_matrices_baselines",
               "Confusion matrices — naive baselines (pooled test observations, n = 288)")
panel_matrices(keys_dl, "fig_confusion_matrices_deep_learning",
               "Confusion matrices — deep-learning benchmark (pooled test observations, n = 288)")
panel_matrices(keys_study + keys_bl + keys_dl, "fig_confusion_matrices_normalized",
               "Row-normalized confusion matrices — all reported estimators", normalize=True)

"""## 11. Letter figure — Pooled MCC with bootstrap 95 % CI (`20b_bootstrap_confidence_intervals`)"""

df = read_csv_or_none("20b_bootstrap_confidence_intervals")
if df is not None:
    d = df[~df["Estimador"].isin(EXCLUDED)].copy()
    d["Label"] = d["Estimador"].map(en)
    d = d.sort_values("MCC_agrupado")
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(d) + 2.2))
    ypos = np.arange(len(d))
    marker = {"study": "o", "deep_learning": "s", "baseline": "^"}
    for i, (_, row) in enumerate(d.iterrows()):
        err_lo = row["MCC_agrupado"] - row["MCC_IC95_inf"]
        err_hi = row["MCC_IC95_sup"] - row["MCC_agrupado"]
        ax.errorbar(row["MCC_agrupado"], i, xerr=[[err_lo], [err_hi]],
                    fmt=marker.get(row["Tipo"], "o"), markersize=7, color="black",
                    markerfacecolor=grey(2 if row["Tipo"] == "study" else 4),
                    markeredgecolor="black", ecolor="black", elinewidth=1.0, capsize=3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(ypos); ax.set_yticklabels(d["Label"])
    ax.set_xlabel("Pooled MCC with 95 % bootstrap confidence interval (2,000 class-stratified resamples)")
    ax.set_title("Matthews correlation coefficient on pooled test observations\n"
                 "circle = classifier of the study · square = deep learning · triangle = naive baseline")
    style_axis(ax, "x")
    plt.tight_layout()
    save_figure(fig, "fig_mcc_confidence_intervals")

"""## 12. Letter figure — Single-class collapse diagnosis of neural networks (`25_neural_network_diagnosis`)"""

df = read_csv_or_none("25_neural_network_diagnosis")
if df is not None:
    d = df.copy(); d["Label"] = d["Arquitectura"].map(en)
    x = np.arange(len(d)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(x - w/2, 100 * d["Tasa_predicciones_positivas"], width=w, color=grey(2),
           edgecolor="black", linewidth=0.7, hatch=hatch(1), label="High predictions (% of test days)")
    ax.bar(x + w/2, 100 * d["Recall"], width=w, color=grey(5), edgecolor="black",
           linewidth=0.7, label="Recall of the High class (%)")
    ax.set_xticks(x); ax.set_xticklabels(d["Label"])
    ax.set_ylabel("%")
    ax.set_title("Neural architectures: positive-prediction rate and recall (pooled test observations)")
    ax.legend(frameon=True, edgecolor="black")
    style_axis(ax, "y")
    plt.tight_layout()
    save_figure(fig, "fig_neural_network_collapse")

"""## 13. Letter figure — Monthly trading activity of the two listed instruments (price CSVs in `DATA/`)"""

PRICE_FILES = {
    "EXALMC1 (Pesquera Exalmar)": os.path.join(DATA_DIR, "EXALMAR_EXALMC1_LM_OHLC_diario.csv"),
    "AUSTRAC1 (Austral Group)":   os.path.join(DATA_DIR, "AUSTRAL_AUSTRAC1_LM_OHLC_diario.csv"),
}
prices = {}
for name, path in PRICE_FILES.items():
    if os.path.exists(path):
        d = pd.read_csv(path, encoding="utf-8-sig"); d["Date"] = pd.to_datetime(d["Date"])
        prices[name] = d.sort_values("Date").reset_index(drop=True)
    else:
        print(f"   [!] Missing price file: {os.path.basename(path)}")
if prices:
    fig, axes = plt.subplots(len(prices), 1, figsize=(12, 3.6 * len(prices)))
    axes = np.atleast_1d(axes).ravel()
    for i, (name, d) in enumerate(prices.items()):
        w = d[(d["Date"] >= DATE_START) & (d["Date"] <= DATE_END)].copy()
        w["ym"] = w["Date"].dt.to_period("M").astype(str)
        g = w.groupby("ym").agg(sessions=("Date", "size"),
                                with_volume=("Volume", lambda s: int((s > 0).sum())))
        axes[i].bar(g.index, g["sessions"], color="white", edgecolor="black",
                    linewidth=0.8, label="Trading sessions in the month")
        axes[i].bar(g.index, g["with_volume"], color=grey(6), edgecolor="black",
                    linewidth=0.8, label="Sessions with traded volume")
        axes[i].set_ylabel("Sessions")
        axes[i].set_title(f"{name} — monthly trading activity, study window")
        axes[i].legend(loc="upper right", frameon=True, edgecolor="black")
        style_axis(axes[i], "y")
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    save_figure(fig, "fig_liquidity_monthly")

"""## 14. Letter figure — Exploratory backtest against random signals (`22b_montecarlo_backtest`, `22_backtest_results`)"""

mc = read_csv_or_none("22b_montecarlo_backtest")
bt = read_csv_or_none("22_backtest_results")
if mc is not None:
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(mc)); w = 0.32
    ax.bar(x - w/2, mc["Retorno_estrategia_pct"], width=w, color=grey(5), edgecolor="black",
           linewidth=0.7, label="Rule-based strategy (25 bps cost)")
    ax.bar(x + w/2, mc["Media_aleatorias_pct"], width=w, color=grey(2), edgecolor="black",
           linewidth=0.7, hatch=hatch(1), label="Random signals: mean (bars) and P5–P95 (whiskers)")
    ax.errorbar(x + w/2, mc["Media_aleatorias_pct"],
                yerr=[mc["Media_aleatorias_pct"] - mc["P5_aleatorias_pct"],
                      mc["P95_aleatorias_pct"] - mc["Media_aleatorias_pct"]],
                fmt="none", ecolor="black", elinewidth=1.0, capsize=4)
    if bt is not None:
        bh = bt[bt["Regla"].str.contains("Buy")].set_index("Activo")["Retorno_total_pct"]
        for i, a in enumerate(mc["Activo"]):
            if a in bh.index:
                ax.plot([i - 0.45, i + 0.45], [bh[a], bh[a]], color="black", linestyle="--", linewidth=1.2,
                        label="Buy & hold" if i == 0 else None)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(mc["Activo"])
    ax.set_ylabel("Total return over the study window (%)")
    ax.set_title("Exploratory backtest (not a validation: instruments fail the liquidity gate)")
    ax.legend(frameon=True, edgecolor="black", loc="lower right")
    style_axis(ax, "y")
    plt.tight_layout()
    save_figure(fig, "fig_montecarlo_backtest")

"""## 15. Inventory of the generated figures"""

generated = sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith(f"{SUFFIX}.png"))
print(f"{len(generated)} figures with suffix {SUFFIX}:")
for f in generated:
    print("  ", f)
print("\nFiguras que van al ARTÍCULO: fig_class_distribution_monthly, fig_protocol_effect, "
      "fig_feature_importance_primary, fig_climate_sensitivity")
print("Figuras que van a la CARTA:   fig_confusion_matrices_study, fig_confusion_matrices_baselines, "
      "fig_confusion_matrices_deep_learning, fig_confusion_matrices_normalized, "
      "fig_neural_network_collapse, fig_mcc_confidence_intervals, fig_liquidity_monthly, fig_montecarlo_backtest")
print("Figura suplementaria:         fig_permutation_importance")
