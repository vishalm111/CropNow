"""
Fungal-Risk Classifier Training & Cross-Validation Evaluation
=============================================================
Compares three lightweight, regularized tabular classifiers for LOW/MEDIUM/HIGH
fungal disease prevalence prediction on the 70-row cleaned flax rust dataset:
  1. Logistic Regression (L2-regularized, standardized, balanced class weights)
  2. Shallow Random Forest (max_depth=3, min_samples_leaf=3, balanced)
  3. Regularized XGBoost (max_depth=2, L1/L2 penalties)

NO GRU / deep sequence networks are used (n=70 tabular dataset).

Validation Protocols:
  - Leave-One-Year-Out (LOYO): 4 folds (2020, 2021, 2022, 2023)
  - Leave-One-Site-Out (LOSO): 4 folds (BT, CC, GM, HM)

Leakage Prevention:
  - Prevalence tertile thresholds (q33.3, q66.7 -> LOW/MEDIUM/HIGH) are
    recalculated strictly inside each training fold only.
  - Feature scaling is fit strictly inside each training fold via Pipeline.

Outputs:
  - Best model artifact: backend/app/ml/models_fungal_risk/best_fungal_model.joblib
  - Model metadata:     backend/app/ml/models_fungal_risk/model_metadata.json
  - Evaluation report:  backend/app/ml/evaluation/fungal_risk/fungal_risk_evaluation_report.md
  - Evaluation JSON:    backend/app/ml/evaluation/fungal_risk/fungal_risk_metrics.json
  - Confusion figure:   backend/app/ml/evaluation/fungal_risk/fungal_risk_confusion_matrices.png
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Callable

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

from app.ml.fungal_features import FEATURE_COLS
from app.ml.prepare_fungal_dataset import assign_fold_risk_labels

DATA_CSV = ROOT_DIR / "data" / "processed" / "fungal_risk" / "flax_rust_fungal_risk_dataset.csv"
MODELS_DIR = BACKEND_DIR / "app" / "ml" / "models_fungal_risk"
EVAL_DIR = BACKEND_DIR / "app" / "ml" / "evaluation" / "fungal_risk"

CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]
LABEL_TO_INT = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def build_candidate_models() -> dict[str, Any]:
    """Instantiate regularized small-N candidate classifiers (no GRU)."""
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.25,
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "ShallowRandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=80,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            reg_alpha=1.0,
            min_child_weight=3,
            eval_metric="mlogloss",
            random_state=42,
        ),
    }


def evaluate_group_cv(
    df: pd.DataFrame,
    group_col: str,
    model_builder: Callable[[], Any],
) -> dict[str, Any]:
    """
    Run group cross-validation (LOYO or LOSO) with training-fold-only
    prevalence quantile thresholds.
    """
    groups = sorted(df[group_col].unique().tolist())
    oof_true: list[int] = []
    oof_pred: list[int] = []
    per_fold: list[dict[str, Any]] = []

    for held_out in groups:
        train_mask = df[group_col] != held_out
        test_mask = df[group_col] == held_out

        train_df = df.loc[train_mask]
        test_df = df.loc[test_mask]

        # Calculate risk-label thresholds STRICTLY within training fold
        train_lbls_str, q33, q67 = assign_fold_risk_labels(
            train_df["prevalence"], train_df["prevalence"]
        )
        test_lbls_str, _, _ = assign_fold_risk_labels(
            train_df["prevalence"], test_df["prevalence"]
        )

        y_train = np.array([LABEL_TO_INT[s] for s in train_lbls_str], dtype=int)
        y_test = np.array([LABEL_TO_INT[s] for s in test_lbls_str], dtype=int)

        X_train = train_df[FEATURE_COLS].to_numpy(dtype=float)
        X_test = test_df[FEATURE_COLS].to_numpy(dtype=float)

        clf = clone(model_builder)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        oof_true.extend(y_test.tolist())
        oof_pred.extend(y_pred.tolist())

        fold_macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        fold_bal_acc = float(balanced_accuracy_score(y_test, y_pred))
        fold_acc = float(accuracy_score(y_test, y_pred))

        per_fold.append({
            "held_out_group": str(held_out),
            "train_n": int(len(train_df)),
            "test_n": int(len(test_df)),
            "q33_train": round(q33, 6),
            "q67_train": round(q67, 6),
            "macro_f1": round(fold_macro_f1, 4),
            "balanced_accuracy": round(fold_bal_acc, 4),
            "accuracy": round(fold_acc, 4),
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
    }


def _plot_confusion_matrices(results: dict[str, dict[str, Any]], out_path: pathlib.Path) -> None:
    """Render 2x3 grid of confusion matrices (rows = LOYO/LOSO, cols = 3 models)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(results.keys())
    fig, axes = plt.subplots(2, len(model_names), figsize=(15, 9))
    fig.suptitle(
        "Fungal-Risk Classifier Confusion Matrices (Fold-Only Quantile Labels, n=70)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for row_idx, (cv_key, cv_title) in enumerate([("loyo", "Leave-One-Year-Out"), ("loso", "Leave-One-Site-Out")]):
        for col_idx, m_name in enumerate(model_names):
            ax = axes[row_idx, col_idx]
            res = results[m_name][cv_key]
            cm = np.array(res["confusion_matrix"])
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))

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
            if col_idx == 0:
                ax.set_ylabel(f"{cv_title}\nActual", fontsize=10, fontweight="bold")
            ax.set_title(
                f"{m_name}\nMacro F1={res['macro_f1']:.3f} | BalAcc={res['balanced_accuracy']:.3f}",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=250)
    plt.close()


def _extract_top_features(fitted_model: Any, model_name: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Extract top predictive environmental features from the fitted winning model."""
    if model_name == "LogisticRegression":
        coefs = np.mean(np.abs(fitted_model.named_steps["clf"].coef_), axis=0)
        scores = coefs
    elif hasattr(fitted_model, "feature_importances_"):
        scores = fitted_model.feature_importances_
    else:
        return []

    order = np.argsort(scores)[::-1][:top_k]
    return [
        {"feature": FEATURE_COLS[i], "importance": round(float(scores[i]), 4)}
        for i in order
    ]


def train_and_evaluate() -> dict[str, Any]:
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Prepared fungal dataset missing: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    assert len(df) == 70, f"Expected 70 rows, got {len(df)}"
    assert df[FEATURE_COLS].isna().sum().sum() == 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    candidates = build_candidate_models()
    all_results: dict[str, dict[str, Any]] = {}

    for m_name, m_template in candidates.items():
        loyo_res = evaluate_group_cv(df, "year", m_template)
        loso_res = evaluate_group_cv(df, "site", m_template)
        mean_macro_f1 = round((loyo_res["macro_f1"] + loso_res["macro_f1"]) / 2.0, 4)
        mean_bal_acc = round((loyo_res["balanced_accuracy"] + loso_res["balanced_accuracy"]) / 2.0, 4)

        all_results[m_name] = {
            "loyo": loyo_res,
            "loso": loso_res,
            "combined_mean_macro_f1": mean_macro_f1,
            "combined_mean_balanced_accuracy": mean_bal_acc,
        }

    # Select winning model based on combined mean Macro F1 (tie-break: Balanced Accuracy)
    best_model_name = max(
        all_results.keys(),
        key=lambda k: (
            all_results[k]["combined_mean_macro_f1"],
            all_results[k]["combined_mean_balanced_accuracy"],
        ),
    )

    # Fit best model on full 70-row dataset for persistence
    full_lbls_str, full_q33, full_q67 = assign_fold_risk_labels(df["prevalence"], df["prevalence"])
    y_full = np.array([LABEL_TO_INT[s] for s in full_lbls_str], dtype=int)
    X_full = df[FEATURE_COLS].to_numpy(dtype=float)

    best_estimator = clone(candidates[best_model_name])
    best_estimator.fit(X_full, y_full)
    top_features = _extract_top_features(best_estimator, best_model_name, top_k=10)

    # Save best model bundle separately from soil-moisture models
    model_artifact = {
        "model_name": best_model_name,
        "estimator": best_estimator,
        "feature_cols": FEATURE_COLS,
        "class_names": CLASS_NAMES,
        "label_to_int": LABEL_TO_INT,
        "reference_prevalence_thresholds": {
            "q33_low": round(full_q33, 6),
            "q67_high": round(full_q67, 6),
        },
        "cv_performance": all_results[best_model_name],
        "dataset_rows": int(len(df)),
        "note": "Trained strictly on flax rust environmental features (no GRU, no soil-moisture model changes).",
    }
    model_path = MODELS_DIR / "best_fungal_model.joblib"
    joblib.dump(model_artifact, model_path)

    metadata = {
        "best_model": best_model_name,
        "model_file": str(model_path.relative_to(ROOT_DIR)),
        "feature_count": len(FEATURE_COLS),
        "feature_cols": FEATURE_COLS,
        "class_names": CLASS_NAMES,
        "training_rows": int(len(df)),
        "reference_full_prevalence_thresholds": {
            "q33_low": round(full_q33, 6),
            "q67_high": round(full_q67, 6),
        },
        "top_features": top_features,
        "cv_summary": all_results,
    }
    meta_path = MODELS_DIR / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Save evaluation JSON & plot
    metrics_json_path = EVAL_DIR / "fungal_risk_metrics.json"
    metrics_json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    cm_plot_path = EVAL_DIR / "fungal_risk_confusion_matrices.png"
    _plot_confusion_matrices(all_results, cm_plot_path)

    # Generate Markdown evaluation report
    report_path = EVAL_DIR / "fungal_risk_evaluation_report.md"
    _write_markdown_report(report_path, all_results, best_model_name, top_features, full_q33, full_q67)

    return metadata


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
        "# Fungal-Risk Classifier Evaluation Report (LOW / MEDIUM / HIGH)",
        "",
        "**Dataset:** `data/processed/fungal_risk/flax_rust_fungal_risk_dataset.csv` (`70 rows` $\\times$ `38` pure environmental features)  ",
        "**Validation Schemes:** Leave-One-Year-Out (LOYO, 4 folds: `2020–2023`) & Leave-One-Site-Out (LOSO, 4 folds: `BT, CC, GM, HM`)  ",
        "**Leakage Guard:** Prevalence tertile cutoffs (`q33`, `q67`) computed **strictly inside each training fold only**; `StandardScaler` fit inside training folds via `Pipeline`.  ",
        "**Models Evaluated:** Logistic Regression, Shallow Random Forest (`max_depth=3`), Regularized XGBoost (`max_depth=2`) — **No GRU used**.  ",
        f"**Selected Best Model:** **`{best_model_name}`** (saved separately to `backend/app/ml/models_fungal_risk/best_fungal_model.joblib`)  ",
        "",
        "---",
        "",
        "## 1. Cross-Validation Model Comparison",
        "",
        "| Model | LOYO Macro F1 | LOYO Bal. Acc | LOSO Macro F1 | LOSO Bal. Acc | Mean Macro F1 | Mean Bal. Acc |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for m_name, res in all_results.items():
        tag = " *(Best)*" if m_name == best_model_name else ""
        lines.append(
            f"| **{m_name}**{tag} | "
            f"{res['loyo']['macro_f1']:.4f} | {res['loyo']['balanced_accuracy']:.4f} | "
            f"{res['loso']['macro_f1']:.4f} | {res['loso']['balanced_accuracy']:.4f} | "
            f"**{res['combined_mean_macro_f1']:.4f}** | **{res['combined_mean_balanced_accuracy']:.4f}** |"
        )

    lines.extend(["", "---", "", "## 2. Per-Class Performance & Confusion Matrices", ""])

    for m_name, res in all_results.items():
        lines.append(f"### 2.{list(all_results.keys()).index(m_name) + 1} {m_name}")
        lines.append("")
        for cv_key, cv_title in [("loyo", "Leave-One-Year-Out (LOYO)"), ("loso", "Leave-One-Site-Out (LOSO)")]:
            cv = res[cv_key]
            lines.append(f"#### {cv_title} — Macro F1: `{cv['macro_f1']:.4f}` | Balanced Accuracy: `{cv['balanced_accuracy']:.4f}` | Accuracy: `{cv['accuracy']:.4f}`")
            lines.append("")
            lines.append("**Per-Class Metrics:**")
            lines.append("")
            lines.append("| Class | Precision | Recall | F1-Score | Support |")
            lines.append("| :--- | :---: | :---: | :---: | :---: |")
            for c in CLASS_NAMES:
                pc = cv["per_class"][c]
                lines.append(f"| **{c}** | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']} |")
            lines.append("")
            lines.append("**Confusion Matrix:**")
            lines.append("")
            lines.append(_format_cm_markdown(cv["confusion_matrix"]))
            lines.append("")

            lines.append("**Per-Fold Breakdown:**")
            lines.append("")
            lines.append("| Held-Out Fold | Train n | Test n | Train `q33` | Train `q67` | Macro F1 | Bal. Accuracy |")
            lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for pf in cv["per_fold"]:
                lines.append(
                    f"| `{pf['held_out_group']}` | {pf['train_n']} | {pf['test_n']} | "
                    f"{pf['q33_train']:.4f} | {pf['q67_train']:.4f} | "
                    f"{pf['macro_f1']:.4f} | {pf['balanced_accuracy']:.4f} |"
                )
            lines.append("")

    lines.extend([
        "---",
        "",
        f"## 3. Top Predictive Features (`{best_model_name}`, Full-Dataset Fit)",
        "",
        f"Reference full-dataset prevalence thresholds: `LOW <= {full_q33:.4f}`, `MEDIUM <= {full_q67:.4f}`, `HIGH > {full_q67:.4f}`.",
        "",
        "| Rank | Feature | Importance / Mean Abs Coef |",
        "| :---: | :--- | :---: |",
    ])
    for idx, feat in enumerate(top_features, start=1):
        lines.append(f"| {idx} | `{feat['feature']}` | {feat['importance']:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Saved Artifacts",
        "",
        "1. **Best Model Artifact:** `backend/app/ml/models_fungal_risk/best_fungal_model.joblib`",
        "2. **Model Metadata:** `backend/app/ml/models_fungal_risk/model_metadata.json`",
        "3. **Structured Metrics JSON:** `backend/app/ml/evaluation/fungal_risk/fungal_risk_metrics.json`",
        "4. **Confusion Matrices Visualization:** `backend/app/ml/evaluation/fungal_risk/fungal_risk_confusion_matrices.png`",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    summary = train_and_evaluate()
    print(f"Best Fungal-Risk Model: {summary['best_model']}")
    for name, res in summary["cv_summary"].items():
        print(
            f"  {name:20s} | LOYO MacroF1={res['loyo']['macro_f1']:.4f}, BalAcc={res['loyo']['balanced_accuracy']:.4f} | "
            f"LOSO MacroF1={res['loso']['macro_f1']:.4f}, BalAcc={res['loso']['balanced_accuracy']:.4f}"
        )
    print(f"Saved model:  {summary['model_file']}")
    print("Saved report: backend/app/ml/evaluation/fungal_risk/fungal_risk_evaluation_report.md")
