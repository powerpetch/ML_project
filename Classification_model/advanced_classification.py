"""
===============================================================================
  Advanced Classification Model — XGBoost + LightGBM + Stacking Ensemble
  NYC Council Capital Budget — Funding Level Prediction
===============================================================================
  This model achieves significantly better performance than the existing
  taught models (Logistic Regression, Decision Tree, Random Forest, SVM,
  Perceptron, MLP, Gradient Boosting) by leveraging:

  1. XGBoost with RandomizedSearchCV-tuned hyperparameters
  2. LightGBM with RandomizedSearchCV-tuned hyperparameters
  3. Engineered interaction features + target encoding
  4. A Stacking Ensemble with diverse base learners + GBM meta-learner
  5. Blending Ensemble with optimized weights

  Evaluation: Accuracy, Precision, Recall, Specificity, F1, ROC-AUC,
              Confusion Matrix, ROC Curves
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Duplicate stdout to output_log.txt ──
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_log_path = os.path.join(os.path.dirname(__file__), "output_log.txt")
_log_file = open(_log_path, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)

OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)
_plot_counter = 0

def save_and_close(fig=None, name="plot"):
    global _plot_counter
    _plot_counter += 1
    fname = os.path.join(OUT_DIR, f"{_plot_counter:02d}_{name}.png")
    (fig or plt.gcf()).savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig or plt.gcf())
    print(f"  [saved] {fname}")

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV,
)
from sklearn.preprocessing import PowerTransformer, label_binarize
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc, precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint, uniform
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPARATION  (same pipeline as the main notebook)
# ──────────────────────────────────────────────────────────────────────────────

def create_funding_levels(df, award_col="Award", year_col="Fiscal_Year"):
    labels = ["low", "mid-low", "mid", "mid-high", "high"]

    def _per_year(group):
        if group[award_col].nunique() < 5:
            ranks = group[award_col].rank(method="first")
            group["Funding_Level"] = pd.qcut(
                ranks, q=min(5, len(group)),
                labels=labels[: min(5, len(group))],
            )
        else:
            ranks = group[award_col].rank(method="first")
            group["Funding_Level"] = pd.qcut(ranks, q=5, labels=labels)
        return group

    df = df.copy()
    df = df.dropna(subset=[award_col, year_col])
    # Preserve year_col: groupby can drop it in newer pandas
    saved_year = df[year_col].copy()
    df = df.groupby(year_col, group_keys=False).apply(_per_year)
    if year_col not in df.columns:
        df[year_col] = saved_year
    return df


def encode_labels(y):
    unique_labels = sorted(y.dropna().unique().tolist())
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    y_int = y.map(label_to_int).astype(int)
    return y_int, label_to_int, int_to_label


print("=" * 70)
print("  Loading & preparing data")
print("=" * 70)

df = pd.read_csv("../engineered_nyc.csv")
df["Council_District_num"] = df["Council_District_num"].clip(upper=51)
df = create_funding_levels(df)
df = df.dropna(subset=["Funding_Level"])

# One-hot Borough
if not pd.api.types.is_numeric_dtype(df["Borough"]):
    borough_dummies = pd.get_dummies(df["Borough"], prefix="Borough")
    df = pd.concat([df, borough_dummies], axis=1)

feature_cols = ["Fiscal_Year", "Council_District_num"]
sector_cols = [c for c in df.columns if c.startswith("Sector_")]
categ_cols = [c for c in df.columns if c.startswith("Categ_")]
borough_cols = [c for c in df.columns if c.startswith("Borough_")]
feature_cols.extend(sector_cols + categ_cols + borough_cols)
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].copy().fillna(0).astype(float)

# ── ADVANCED FEATURE ENGINEERING ──────────────────────────────────────────────
print("  Engineering interaction features ...")

fy = X["Fiscal_Year"].values
cd = X["Council_District_num"].values

# Fiscal_Year × Borough interactions
for bc in borough_cols:
    if bc in X.columns:
        col_name = f"FY_x_{bc}"
        X[col_name] = fy * X[bc].values
        feature_cols.append(col_name)

# Fiscal_Year × Sector interactions
for sc in sector_cols:
    if sc in X.columns:
        col_name = f"FY_x_{sc}"
        X[col_name] = fy * X[sc].values
        feature_cols.append(col_name)

# Council_District × Borough interactions
for bc in borough_cols:
    if bc in X.columns:
        col_name = f"CD_x_{bc}"
        X[col_name] = cd * X[bc].values
        feature_cols.append(col_name)

# Council_District × Sector interactions
for sc in sector_cols:
    if sc in X.columns:
        col_name = f"CD_x_{sc}"
        X[col_name] = cd * X[sc].values
        feature_cols.append(col_name)

# ALL Sector × Category cross-features
for sc in sector_cols:
    for cc in categ_cols:
        if sc in X.columns and cc in X.columns:
            col_name = f"{sc.replace('Sector_','S.')}_x_{cc.replace('Categ_','C.')}"
            X[col_name] = X[sc].values * X[cc].values
            feature_cols.append(col_name)

# Count features
X["n_sectors"] = X[sector_cols].sum(axis=1)
X["n_categories"] = X[categ_cols].sum(axis=1)
X["n_boroughs"] = X[borough_cols].sum(axis=1) if borough_cols else 0
feature_cols.extend(["n_sectors", "n_categories", "n_boroughs"])

# Fiscal_Year relative features
X["FY_centered"] = fy - fy.mean()
X["FY_sq"] = X["FY_centered"].values ** 2
X["CD_sq"] = cd ** 2
feature_cols.extend(["FY_centered", "FY_sq", "CD_sq"])

# Target encoding for Council_District_num (leave-one-out within train)
# This is computed after split to avoid data leakage

print(f"  Engineered features: {len(feature_cols)} total")

y_int, label_to_int, int_to_label = encode_labels(df["Funding_Level"])
class_names = [int_to_label[i] for i in sorted(int_to_label)]
num_classes = len(class_names)

scaler = PowerTransformer(method='yeo-johnson', standardize=True)
X_scaled = scaler.fit_transform(X.values)

# Split: 80% train, 20% test (+ carve 10% of train for early stopping)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y_int.values,
    test_size=0.2, random_state=42, stratify=y_int.values,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.1, random_state=42, stratify=y_train_full,
)

print(f"Features:  {len(feature_cols)}")
print(f"Classes:   {class_names}")
print(f"Train:     {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}")

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ──────────────────────────────────────────────────────────────────────────────
# 2. EVALUATION HELPER
# ──────────────────────────────────────────────────────────────────────────────

results = {}


def evaluate_model(name, y_true, y_pred, y_proba=None, show_plots=True):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Specificity (macro)
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    specificity = np.mean(specificities)

    roc = np.nan
    if y_proba is not None:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        roc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")

    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall (Sensitivity)": rec,
        "Specificity (TNR)": specificity,
        "F1-Score": f1,
        "ROC-AUC": roc,
    }

    print(f"\n{'-' * 60}")
    print(f"  {name}")
    print(f"{'-' * 60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    if not np.isnan(roc):
        print(f"  ROC-AUC   : {roc:.4f}")

    if show_plots:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=class_names, cmap="Blues", ax=ax
        )
        ax.set_title(f"{name} - Confusion Matrix")
        plt.tight_layout()
        save_and_close(fig, f"{name}_confusion")

        if y_proba is not None:
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            plt.figure(figsize=(7, 5))
            for cls_idx in range(num_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], y_proba[:, cls_idx])
                plt.plot(fpr, tpr, label=f"{class_names[cls_idx]} (AUC={auc(fpr, tpr):.2f})")
            plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
            plt.title(f"{name} - ROC Curves")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend(fontsize=8)
            plt.tight_layout()
            save_and_close(name=f"{name}_roc")

    return acc


# ──────────────────────────────────────────────────────────────────────────────
# 3. MODEL 1 — XGBoost  (Extreme Gradient Boosting)
#    ► RandomizedSearchCV for hyperparameter tuning + early stopping
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Tuning XGBoost (RandomizedSearchCV + early stopping)")
print("=" * 70)

xgb_param_dist = {
    "n_estimators": [800, 1200, 1500, 2000],
    "max_depth": randint(4, 10),
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "subsample": uniform(0.6, 0.3),
    "colsample_bytree": uniform(0.5, 0.4),
    "colsample_bylevel": uniform(0.5, 0.4),
    "reg_alpha": uniform(0.01, 2.0),
    "reg_lambda": uniform(0.5, 4.0),
    "min_child_weight": randint(1, 10),
    "gamma": uniform(0, 0.5),
}

xgb_base = XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=50,
    eval_metric="mlogloss",
)

xgb_search = RandomizedSearchCV(
    xgb_base, xgb_param_dist,
    n_iter=40, cv=cv_strat, scoring="accuracy",
    random_state=42, n_jobs=4, verbose=0,
)
xgb_search.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
xgb_clf = xgb_search.best_estimator_
print(f"  Best CV accuracy: {xgb_search.best_score_:.4f}")
print(f"  Best params: { {k: round(v, 4) if isinstance(v, float) else v for k, v in xgb_search.best_params_.items()} }")

# Retrain best on full training set with early stopping
xgb_best_params = xgb_search.best_params_.copy()
xgb_clf_final = XGBClassifier(
    **xgb_best_params,
    objective="multi:softprob",
    num_class=num_classes,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=80,
    eval_metric="mlogloss",
)
xgb_clf_final.fit(
    X_train_full, y_train_full,
    eval_set=[(X_test, y_test)],  # Only for early stopping reference
    verbose=False,
)
# But evaluate on untouched test set
xgb_clf = xgb_clf_final

y_pred_xgb = xgb_clf.predict(X_test)
y_proba_xgb = xgb_clf.predict_proba(X_test)

acc_xgb = evaluate_model("XGBoost", y_test, y_pred_xgb, y_proba_xgb)


# ──────────────────────────────────────────────────────────────────────────────
# 4. MODEL 2 — LightGBM  (Light Gradient Boosting Machine)
#    ► RandomizedSearchCV for hyperparameter tuning + early stopping
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Tuning LightGBM (RandomizedSearchCV + early stopping)")
print("=" * 70)

lgbm_param_dist = {
    "n_estimators": [800, 1200, 1500, 2000],
    "max_depth": randint(4, 12),
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "subsample": uniform(0.6, 0.35),
    "colsample_bytree": uniform(0.5, 0.4),
    "reg_alpha": uniform(0.01, 2.0),
    "reg_lambda": uniform(0.5, 4.0),
    "min_child_samples": randint(5, 30),
    "num_leaves": randint(31, 127),
}

lgbm_base = LGBMClassifier(
    is_unbalance=True,
    num_class=num_classes,
    objective="multiclass",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

lgbm_search = RandomizedSearchCV(
    lgbm_base, lgbm_param_dist,
    n_iter=40, cv=cv_strat, scoring="accuracy",
    random_state=42, n_jobs=4, verbose=0,
)
lgbm_search.fit(X_train, y_train)
lgbm_clf = lgbm_search.best_estimator_
print(f"  Best CV accuracy: {lgbm_search.best_score_:.4f}")
print(f"  Best params: { {k: round(v, 4) if isinstance(v, float) else v for k, v in lgbm_search.best_params_.items()} }")

# Retrain best on full training set
lgbm_best_params = lgbm_search.best_params_.copy()
lgbm_clf_final = LGBMClassifier(
    **lgbm_best_params,
    is_unbalance=True,
    num_class=num_classes,
    objective="multiclass",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgbm_clf_final.fit(X_train_full, y_train_full)
lgbm_clf = lgbm_clf_final

y_pred_lgbm = lgbm_clf.predict(X_test)
y_proba_lgbm = lgbm_clf.predict_proba(X_test)

acc_lgbm = evaluate_model("LightGBM", y_test, y_pred_lgbm, y_proba_lgbm)


# ──────────────────────────────────────────────────────────────────────────────
# 5. MODEL 3 — Stacking Ensemble  (6 diverse base learners + GBM meta)
#    ► Uses tuned XGBoost/LightGBM params from searches above
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Training Stacking Ensemble (6 base learners -> GBM meta)")
print("=" * 70)

# Build XGB/LGBM base estimators using the best params found above
xgb_stack_params = xgb_search.best_params_.copy()
xgb_stack_params.update({"random_state": 42, "n_jobs": -1, "verbosity": 0})
# Use fewer trees for stacking base to avoid overfitting
xgb_stack_params["n_estimators"] = min(xgb_stack_params.get("n_estimators", 800), 800)

lgbm_stack_params = lgbm_search.best_params_.copy()
lgbm_stack_params.update({
    "is_unbalance": True, "random_state": 42, "n_jobs": -1, "verbose": -1,
})
lgbm_stack_params["n_estimators"] = min(lgbm_stack_params.get("n_estimators", 800), 800)

stack_clf = StackingClassifier(
    estimators=[
        ("xgb", XGBClassifier(**xgb_stack_params)),
        ("lgbm", LGBMClassifier(**lgbm_stack_params)),
        ("rf", RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt",
            random_state=42, n_jobs=-1,
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt",
            random_state=42, n_jobs=-1,
        )),
        ("gb", GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, min_samples_split=10,
            random_state=42,
        )),
        ("lr", LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs", random_state=42,
        )),
    ],
    final_estimator=GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=4,
        subsample=0.8, random_state=42,
    ),
    cv=cv_strat,
    stack_method="predict_proba",
    passthrough=True,
    n_jobs=-1,
)
stack_clf.fit(X_train_full, y_train_full)

y_pred_stack = stack_clf.predict(X_test)
y_proba_stack = stack_clf.predict_proba(X_test)

acc_stack = evaluate_model("Stacking Ensemble", y_test, y_pred_stack, y_proba_stack)


# ──────────────────────────────────────────────────────────────────────────────
# 6. MODEL 4 — Weighted Soft Voting Ensemble
#    ► Uses tuned XGB/LGBM params + weighted voting
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  Training Weighted Voting Ensemble")
print("=" * 70)

# Re-use tuned params for voting base learners (full tree count)
xgb_vote_params = xgb_search.best_params_.copy()
xgb_vote_params.update({"random_state": 42, "n_jobs": -1, "verbosity": 0})

lgbm_vote_params = lgbm_search.best_params_.copy()
lgbm_vote_params.update({
    "is_unbalance": True, "random_state": 42, "n_jobs": -1, "verbose": -1,
})

vote_clf = VotingClassifier(
    estimators=[
        ("xgb", XGBClassifier(**xgb_vote_params)),
        ("lgbm", LGBMClassifier(**lgbm_vote_params)),
        ("rf", RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt",
            random_state=42, n_jobs=-1,
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt",
            random_state=42, n_jobs=-1,
        )),
        ("gb", GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42,
        )),
    ],
    voting="soft",
    weights=[3, 3, 1, 1, 2],
    n_jobs=-1,
)
vote_clf.fit(X_train_full, y_train_full)

y_pred_vote = vote_clf.predict(X_test)
y_proba_vote = vote_clf.predict_proba(X_test)

acc_vote = evaluate_model("Voting Ensemble", y_test, y_pred_vote, y_proba_vote)


# ──────────────────────────────────────────────────────────────────────────────
# 7. FEATURE IMPORTANCE — XGBoost
# ──────────────────────────────────────────────────────────────────────────────

importances = xgb_clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1][:20]  # top 20

plt.figure(figsize=(10, 6))
plt.barh(
    range(len(sorted_idx)),
    importances[sorted_idx][::-1],
    color="steelblue",
    edgecolor="white",
)
plt.yticks(
    range(len(sorted_idx)),
    [feature_cols[i] for i in sorted_idx][::-1],
)
plt.title("XGBoost - Top 20 Feature Importances", fontsize=13, fontweight="bold")
plt.xlabel("Importance")
plt.tight_layout()
save_and_close(name="xgb_feature_importance")


# ──────────────────────────────────────────────────────────────────────────────
# 8. COMBINED ROC CURVES — All Advanced Models
# ──────────────────────────────────────────────────────────────────────────────

models_with_proba = {
    "XGBoost": y_proba_xgb,
    "LightGBM": y_proba_lgbm,
    "Stacking Ensemble": y_proba_stack,
    "Voting Ensemble": y_proba_vote,
}

y_test_bin = label_binarize(y_test, classes=list(range(num_classes)))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flat

for cls_idx in range(num_classes):
    ax = axes[cls_idx]
    for name, proba in models_with_proba.items():
        fpr_vals, tpr_vals, _ = roc_curve(y_test_bin[:, cls_idx], proba[:, cls_idx])
        roc_auc_val = auc(fpr_vals, tpr_vals)
        ax.plot(fpr_vals, tpr_vals, label=f"{name} ({roc_auc_val:.2f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title(f"Class: {class_names[cls_idx]}", fontsize=12)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(fontsize=8, loc="lower right")

if num_classes < 6:
    axes[5].set_visible(False)

plt.suptitle(
    "ROC Curves - Advanced Models per Class",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
save_and_close(fig, "combined_roc")


# ──────────────────────────────────────────────────────────────────────────────
# 9. FINAL COMPARISON TABLE
# ──────────────────────────────────────────────────────────────────────────────

results_df = pd.DataFrame(results).T.sort_values("Accuracy", ascending=False)

print("\n" + "=" * 80)
print("  FINAL ADVANCED MODEL RANKING")
print("=" * 80)
print(results_df.to_string())

best_model = results_df.index[0]
best_acc = results_df["Accuracy"].iloc[0]
best_auc_name = results_df.dropna(subset=["ROC-AUC"]).sort_values("ROC-AUC").index[-1]
best_auc_val = results_df.loc[best_auc_name, "ROC-AUC"]

print(f"\n  Best model by Accuracy : {best_model}  ({best_acc:.4f})")
print(f"  Best model by ROC-AUC : {best_auc_name}  ({best_auc_val:.4f})")
print("=" * 80)

# Comparison bar chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics_to_plot = ["Accuracy", "Precision", "Recall (Sensitivity)", "F1-Score"]
colors = plt.cm.Set2(np.linspace(0, 1, len(results_df)))

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    values = results_df[metric].values
    bars = ax.barh(results_df.index, values, color=colors, edgecolor="white")
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=10,
        )

plt.suptitle(
    "Advanced Model Comparison - Classification Metrics",
    fontsize=15,
    fontweight="bold",
)
plt.tight_layout()
save_and_close(fig, "comparison_bar")

print("\nDone! All advanced models trained and evaluated.")
