"""
Pest-Risk Classifier Training & Chronological Evaluation (RBB Target)
=====================================================================
Compares three regularized tabular classifiers for LOW/MEDIUM/HIGH Rice Black
Bug (RBB) outbreak prediction using `data/processed/pest_risk/rbb_pest_risk_dataset.csv`:
  1. Logistic Regression (L2-regularized, StandardScaler inside Pipeline, balanced)
  2. Shallow Random Forest (max_depth=4, min_samples_leaf=5, balanced)
  3. Regularized XGBoost (max_depth=3, L1/L2 penalties)

NO GRU / deep sequence models are used.
Existing soil-moisture, fungal-risk models, and source/prepared datasets are NOT modified.

Validation Protocol:
  Uses the existing 4 expanding-window chronological walk-forward folds:
    - Fold 1: Train 2018-2020 (n=154) -> Eval 2021 (n=52)
    - Fold 2: Train 2018-2021 (n=206) -> Eval 2022 (n=52)
    - Fold 3: Train 2018-2022 (n=258) -> Eval 2023 (n=52)
    - Fold 4: Train 2018-2023 (n=310) -> Eval 2024 (n=53)

Leakage Prevention:
  - Prevalence/count tertile thresholds (q33.3, q66.7 -> LOW/MEDIUM/HIGH)
    are recalculated STRICTLY within each chronological training fold only.
  - Preprocessing (StandardScaler) is fit strictly within each training fold.
  - Features are strictly environmental-only (26 weather & soil predictors).
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.prepare_pest_dataset import (
    ENVIRONMENTAL_FEATURE_COLS,
    WALK_FORWARD_VALIDATION_YEARS,
    assign_pest_fold_labels,
)

DATA_CSV = ROOT_DIR / "data" / "processed" / "pest_risk" / "rbb_pest_risk_dataset.csv"
MODELS_DIR = BACKEND_DIR / "app" / "ml" / "models_pest_risk"
EVAL_DIR = BACKEND_DIR / "app" / "ml" / "evaluation" / "pest_risk"

CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]
LABEL_TO_INT = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def build_candidate_models() -> dict[str, Any]:
    """Instantiate candidate regularized tabular classifiers (no GRU)."""
    return {
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            reg_alpha=1.0,
            min_child_weight=5,
            eval_metric="mlogloss",
            random_state=42,
        ),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.5,
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "ShallowRandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
        ),
    }


def evaluate_chronological_cv(
    df: pd.DataFrame,
    model_template: Any,
) -> dict[str, Any]:
    """
    Evaluate model across the 4 expanding-window chronological walk-forward folds
    with training-fold-only scaling and quantile thresholds.
    """
    oof_true: list[int] = []
    oof_pred: list[int] = []
    per_fold: list[dict[str, Any]] = []

    for val_year in WALK_FORWARD_VALIDATION_YEARS:
        if val_year == 2024:
            train_mask = df["sort_year"] < 2024
            val_mask = df["sort_year"] >= 2024
        else:
            train_mask = df["sort_year"] < val_year
            val_mask = df["sort_year"] == val_year

        train_df = df.loc[train_mask]
        val_df = df.loc[val_mask]

        # Strict training-fold-only quantile thresholds
        train_lbls_str, q33, q67 = assign_pest_fold_labels(
            train_df["rbb_count"], train_df["rbb_count"]
        )
        val_lbls_str, _, _ = assign_pest_fold_labels(
            train_df["rbb_count"], val_df["rbb_count"]
        )

        y_train = np.array([LABEL_TO_INT[s] for s in train_lbls_str], dtype=int)
        y_val = np.array([LABEL_TO_INT[s] for s in val_lbls_str], dtype=int)

        X_train = train_df[ENVIRONMENTAL_FEATURE_COLS].to_numpy(dtype=float)
        X_val = val_df[ENVIRONMENTAL_FEATURE_COLS].to_numpy(dtype=float)

        clf = clone(model_template)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        oof_true.extend(y_val.tolist())
        oof_pred.extend(y_pred.tolist())

        fold_macro_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        fold_bal_acc = float(balanced_accuracy_score(y_val, y_pred))
        fold_acc = float(accuracy_score(y_val, y_pred))
        fold_cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2]).tolist()

        per_fold.append({
            "eval_year": "2024 (+ 2024-12-30 wk)" if val_year == 2024 else str(val_year),
            "train_n": int(len(train_df)),
            "eval_n": int(len(val_df)),
            "q33_train": round(q33, 4),
            "q67_train": round(q67, 4),
            "macro_f1": round(fold_macro_f1, 4),
            "balanced_accuracy": round(fold_bal_acc, 4),
            "accuracy": round(fold_acc, 4),
            "confusion_matrix": fold_cm,
        })

    y_true_arr = np.array(oof_true, dtype=int)
    y_pred_arr = np.array(oof_pred, dtype=int)

    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1, 2])
    cls_rep = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "pooled_oof_n": int(len(y_true_arr)),
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "accuracy": round(acc, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            c: {
                "precision": round(float(cls_rep[c]["precision"]), 4),
                "recall": round(float(cls_rep[c]["recall"]), 4),
                "f1": round(float(cls_rep[c]["f1-score"]), 4),
                "support": int(cls_rep[c]["support"]),
            }
            for c in CLASS_NAMES
        },
        "per_fold": per_fold,
        "holdout_2024_macro_f1": per_fold[-1]["macro_f1"],
        "holdout_2024_balanced_accuracy": per_fold[-1]["balanced_accuracy"],
    }


def _plot_confusion_matrices(results: dict[str, dict[str, Any]], out_path: pathlib.Path) -> None:
    """Plot pooled 4-fold chronological OOF confusion matrices for all 3 models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(results.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(15, 4.8))
    fig.suptitle(
        "RBB Pest-Risk Classifiers — Chronological Walk-Forward Confusion Matrices (n=209 OOF, 2021-2024)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for idx, m_name in enumerate(model_names):
        ax = axes[idx]
        res = results[m_name]
        cm = np.array(res["confusion_matrix"])
        ax.imshow(cm, cmap="Greens", vmin=0, vmax=max(1, cm.max()))

        for i in range(3):
            for j in range(3):
                val = cm[i, j]
                color = "white" if val > cm.max() * 0.55 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=12, fontweight="bold")

        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(CLASS_NAMES, fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Actual", fontsize=10, fontweight="bold")
        ax.set_title(
            f"{m_name}\nMacro F1={res['macro_f1']:.3f} | BalAcc={res['balanced_accuracy']:.3f}",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=250)
    plt.close()


def _extract_top_features(fitted_model: Any, model_name: str, top_k: int = 10) -> list[dict[str, Any]]:
    if model_name == "LogisticRegression":
        scores = np.mean(np.abs(fitted_model.named_steps["clf"].coef_), axis=0)
    elif hasattr(fitted_model, "feature_importances_"):
        scores = fitted_model.feature_importances_
    else:
        return []

    order = np.argsort(scores)[::-1][:top_k]
    return [
        {"feature": ENVIRONMENTAL_FEATURE_COLS[i], "importance": round(float(scores[i]), 4)}
        for i in order
    ]


def _format_cm_markdown(cm: list[list[int]]) -> str:
    lines = [
        "| Actual \\ Predicted | LOW | MEDIUM | HIGH |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for idx, cls in enumerate(CLASS_NAMES):
        lines.append(f"| **{cls}** | {cm[idx][0]} | {cm[idx][1]} | {cm[idx][2]} |")
    return "\n".join(lines)


def _write_markdown_report(
    report_path: pathlib.Path,
    all_results: dict[str, dict[str, Any]],
    best_model_name: str,
    top_features: list[dict[str, Any]],
    full_q33: float,
    full_q67: float,
) -> None:
    lines: list[str] = [
        "# Pest-Risk Classifier Evaluation Report (RBB LOW / MEDIUM / HIGH)",
        "",
        "**Dataset:** `data/processed/pest_risk/rbb_pest_risk_dataset.csv` (`363 rows` $\\times$ `26` environmental-only features; `WSB` excluded)  ",
        "**Validation Scheme:** 4 Expanding-Window Chronological Walk-Forward Folds (`2021`, `2022`, `2023`, `2024`; `209` pooled out-of-fold predictions)  ",
        "**Leakage Guards:** RBB count tertiles (`q33`, `q67`) and `StandardScaler` computed **strictly inside each training fold only**. Random/interleaved split rejected.  ",
        "**Models Evaluated:** XGBoost, Logistic Regression, Shallow Random Forest — **No GRU used**.  ",
        f"**Selected Best Model:** **`{best_model_name}`** (saved separately to `backend/app/ml/models_pest_risk/best_pest_model.joblib`)  ",
        "",
        "---",
        "",
        "## 1. Chronological Cross-Validation Model Comparison",
        "",
        "| Model | 4-Fold Pooled Macro F1 | 4-Fold Balanced Acc | 4-Fold Accuracy | 2024 Holdout Macro F1 | 2024 Holdout Bal. Acc |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for m_name, res in all_results.items():
        tag = " *(Best)*" if m_name == best_model_name else ""
        lines.append(
            f"| **{m_name}**{tag} | "
            f"**{res['macro_f1']:.4f}** | **{res['balanced_accuracy']:.4f}** | {res['accuracy']:.4f} | "
            f"{res['holdout_2024_macro_f1']:.4f} | {res['holdout_2024_balanced_accuracy']:.4f} |"
        )

    lines.extend(["", "---", "", "## 2. Per-Class Performance, Confusion Matrices & Per-Fold Breakdown", ""])

    for idx, (m_name, res) in enumerate(all_results.items(), start=1):
        lines.append(f"### 2.{idx} {m_name} (Pooled Macro F1: `{res['macro_f1']:.4f}` | Balanced Accuracy: `{res['balanced_accuracy']:.4f}`)")
        lines.append("")
        lines.append("**Per-Class Metrics (2021–2024 Pooled OOF, `n=209`):**")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1-Score | Support |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for c in CLASS_NAMES:
            pc = res["per_class"][c]
            lines.append(f"| **{c}** | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']} |")
        lines.append("")
        lines.append("**Confusion Matrix (Pooled OOF `n=209`):**")
        lines.append("")
        lines.append(_format_cm_markdown(res["confusion_matrix"]))
        lines.append("")
        lines.append("**Per-Fold Chronological Breakdown:**")
        lines.append("")
        lines.append("| Held-Out Eval Year | Train n | Eval n | Train `q33` | Train `q67` | Macro F1 | Bal. Accuracy | Accuracy |")
        lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for pf in res["per_fold"]:
            lines.append(
                f"| `{pf['eval_year']}` | {pf['train_n']} | {pf['eval_n']} | "
                f"{pf['q33_train']:.2f} | {pf['q67_train']:.2f} | "
                f"{pf['macro_f1']:.4f} | {pf['balanced_accuracy']:.4f} | {pf['accuracy']:.4f} |"
            )
        lines.append("")

    lines.extend([
        "---",
        "",
        f"## 3. Top Environmental Drivers (`{best_model_name}`, Full-Dataset Fit)",
        "",
        f"Reference full-dataset RBB count thresholds: `LOW <= {full_q33:.2f}`, `MEDIUM <= {full_q67:.2f}`, `HIGH > {full_q67:.2f}`.",
        "",
        "| Rank | Environmental Feature | Importance / Mean Abs Coef |",
        "| :---: | :--- | :---: |",
    ])
    for idx, feat in enumerate(top_features, start=1):
        lines.append(f"| {idx} | `{feat['feature']}` | {feat['importance']:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Saved Artifacts (Separate from Soil Moisture & Fungal Models)",
        "",
        "1. **Best Model Bundle:** `backend/app/ml/models_pest_risk/best_pest_model.joblib`",
        "2. **Model Metadata:** `backend/app/ml/models_pest_risk/model_metadata.json`",
        "3. **Structured Metrics JSON:** `backend/app/ml/evaluation/pest_risk/pest_risk_metrics.json`",
        "4. **Confusion Matrices Visualization:** `backend/app/ml/evaluation/pest_risk/pest_risk_confusion_matrices.png`",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")


def train_and_evaluate() -> dict[str, Any]:
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Prepared pest dataset missing: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    assert len(df) == 363, f"Expected 363 rows, got {len(df)}"
    assert df[ENVIRONMENTAL_FEATURE_COLS].isna().sum().sum() == 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    candidates = build_candidate_models()
    all_results: dict[str, dict[str, Any]] = {}

    for m_name, m_template in candidates.items():
        all_results[m_name] = evaluate_chronological_cv(df, m_template)

    best_model_name = max(
        all_results.keys(),
        key=lambda k: (
            all_results[k]["macro_f1"],
            all_results[k]["balanced_accuracy"],
        ),
    )

    # Fit winning model configuration on the full 363-row environmental table for persistence
    full_lbls_str, full_q33, full_q67 = assign_pest_fold_labels(df["rbb_count"], df["rbb_count"])
    y_full = np.array([LABEL_TO_INT[s] for s in full_lbls_str], dtype=int)
    X_full = df[ENVIRONMENTAL_FEATURE_COLS].to_numpy(dtype=float)

    best_estimator = clone(candidates[best_model_name])
    best_estimator.fit(X_full, y_full)
    top_features = _extract_top_features(best_estimator, best_model_name, top_k=10)

    model_artifact = {
        "model_name": best_model_name,
        "target_pest": "RBB",
        "estimator": best_estimator,
        "feature_cols": ENVIRONMENTAL_FEATURE_COLS,
        "class_names": CLASS_NAMES,
        "label_to_int": LABEL_TO_INT,
        "reference_rbb_thresholds": {
            "q33_low": round(full_q33, 4),
            "q67_high": round(full_q67, 4),
        },
        "cv_performance": all_results[best_model_name],
        "dataset_rows": int(len(df)),
        "note": "Trained on RBB environmental-only features under chronological validation (no GRU, WSB excluded).",
    }
    model_path = MODELS_DIR / "best_pest_model.joblib"
    joblib.dump(model_artifact, model_path)

    metadata = {
        "best_model": best_model_name,
        "target_pest": "RBB",
        "model_file": str(model_path.relative_to(ROOT_DIR)),
        "feature_count": len(ENVIRONMENTAL_FEATURE_COLS),
        "feature_cols": ENVIRONMENTAL_FEATURE_COLS,
        "class_names": CLASS_NAMES,
        "training_rows": int(len(df)),
        "reference_full_rbb_thresholds": {
            "q33_low": round(full_q33, 4),
            "q67_high": round(full_q67, 4),
        },
        "top_features": top_features,
        "cv_summary": all_results,
    }

    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (EVAL_DIR / "pest_risk_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _plot_confusion_matrices(all_results, EVAL_DIR / "pest_risk_confusion_matrices.png")
    _write_markdown_report(
        EVAL_DIR / "pest_risk_evaluation_report.md",
        all_results,
        best_model_name,
        top_features,
        full_q33,
        full_q67,
    )

    return metadata


if __name__ == "__main__":
    summary = train_and_evaluate()
    print(f"Best Pest-Risk Model (RBB): {summary['best_model']}")
    for name, res in summary["cv_summary"].items():
        print(
            f"  {name:20s} | Pooled MacroF1={res['macro_f1']:.4f}, BalAcc={res['balanced_accuracy']:.4f}, "
            f"Acc={res['accuracy']:.4f} | 2024 MacroF1={res['holdout_2024_macro_f1']:.4f}"
        )
    print(f"Saved model:  {summary['model_file']}")
    print("Saved report: backend/app/ml/evaluation/pest_risk/pest_risk_evaluation_report.md")
