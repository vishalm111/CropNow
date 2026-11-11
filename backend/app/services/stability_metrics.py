"""
Quantitative Evaluation Metrics for Soil Moisture Stability Conditioning.

Separates genuine physical dynamic range retention (evaluated against the clean
signal excluding isolated sensor faults) from sensor spike / outlier suppression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_range_and_outlier_metrics(
    raw_avg: np.ndarray,
    filt_avg: np.ndarray,
    fault_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """
    Compute genuine physical dynamic range retention separately from outlier suppression.

    Args:
        raw_avg: Raw dual-probe soil moisture average array.
        filt_avg: Filtered / stabilized soil moisture average array.
        fault_mask: Optional boolean mask where True indicates known injected faults.

    Returns:
        Dictionary containing:
          - raw_total_range: Peak-to-peak range of raw signal including any spikes.
          - clean_baseline_range: Peak-to-peak range of clean baseline excluding spikes.
          - filtered_dynamic_range: Peak-to-peak range of filtered signal.
          - genuine_range_retention_pct: (filtered_dynamic_range / clean_baseline_range) * 100.
          - outlier_suppression_pct: Percentage suppression of isolated spike amplitudes.
          - outliers_detected_count: Count of isolated sensor spikes identified.
    """
    # Centered 5-point rolling median detects isolated impulse faults (e.g. 0% / 100% spikes)
    # without misclassifying genuine sustained physical steps (e.g. irrigation wetting fronts).
    centered_med = (
        pd.Series(raw_avg)
        .rolling(window=5, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    auto_outlier_mask = np.abs(raw_avg - centered_med) > 5.0

    if fault_mask is not None and np.any(fault_mask):
        outlier_mask = auto_outlier_mask | (
            np.asarray(fault_mask, dtype=bool) & (np.abs(raw_avg - centered_med) > 1.0)
        )
    else:
        outlier_mask = auto_outlier_mask

    clean_mask = ~outlier_mask
    if np.sum(clean_mask) < 2:
        clean_mask = np.ones_like(raw_avg, dtype=bool)
        outlier_mask = np.zeros_like(raw_avg, dtype=bool)

    raw_total_range = float(np.ptp(raw_avg))
    clean_baseline_range = float(np.ptp(raw_avg[clean_mask]))
    filtered_range = float(np.ptp(filt_avg))

    genuine_range_retention_pct = (
        filtered_range / max(clean_baseline_range, 1e-6)
    ) * 100.0

    outliers_count = int(np.sum(outlier_mask))
    if outliers_count > 0:
        raw_spike_err = float(np.sum(np.abs(raw_avg[outlier_mask] - centered_med[outlier_mask])))
        filt_resid_err = float(np.sum(np.abs(filt_avg[outlier_mask] - centered_med[outlier_mask])))
        outlier_suppression_pct = max(
            0.0, (1.0 - filt_resid_err / max(raw_spike_err, 1e-6)) * 100.0
        )
    else:
        outlier_suppression_pct = 0.0

    return {
        "raw_total_range": round(raw_total_range, 3),
        "clean_baseline_range": round(clean_baseline_range, 3),
        "filtered_dynamic_range": round(filtered_range, 3),
        "genuine_range_retention_pct": round(genuine_range_retention_pct, 2),
        "outlier_suppression_pct": round(outlier_suppression_pct, 2),
        "outliers_detected_count": outliers_count,
    }
