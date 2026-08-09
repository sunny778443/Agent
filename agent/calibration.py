"""
Confidence Calibration and Verification module for Project Karthikeya.
Implements computational models for:
- Brier Score calculation
- Expected Calibration Error (ECE) calculation
- Strategy Learning exploitation metrics
"""
import math
from typing import List, Dict, Any, Tuple


class ConfidenceCalibrator:
    """
    Evaluates how closely predicted planner/execution confidence scores align
    with actual sandbox success probabilities using Brier and Expected Calibration Error (ECE).
    """
    def __init__(self, num_bins: int = 5) -> None:
        self.num_bins = num_bins

    def compute_brier_score(self, predictions: List[float], outcomes: List[float]) -> float:
        """
        Computes the standard Brier Score for probability calibration:
        BS = 1/N * sum((p_t - y_t)^2)
        """
        if not predictions or not outcomes or len(predictions) != len(outcomes):
            return 0.0
        total_error = sum((p - y) ** 2 for p, y in zip(predictions, outcomes))
        return total_error / len(predictions)

    def compute_expected_calibration_error(self, predictions: List[float], outcomes: List[float]) -> float:
        """
        Computes the Expected Calibration Error (ECE) stably across partitioned bins:
        ECE = sum(|B_m|/N * |acc(B_m) - conf(B_m)|)
        """
        n = len(predictions)
        if n == 0 or len(predictions) != len(outcomes):
            return 0.0

        # Define bins (e.g. 5 bins: [0.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0])
        bins_limits = [i / self.num_bins for i in range(self.num_bins + 1)]
        bins_indices = {i: [] for i in range(self.num_bins)}

        # Assign predictions and outcomes to bins
        for idx, (p, y) in enumerate(zip(predictions, outcomes)):
            # Handle boundary case: p = 1.0 goes to the last bin
            if p >= 1.0:
                bin_idx = self.num_bins - 1
            else:
                bin_idx = int(p * self.num_bins)
            bins_indices[bin_idx].append((p, y))

        ece = 0.0
        for bin_idx, items in bins_indices.items():
            bin_size = len(items)
            if bin_size == 0:
                continue

            # Compute bin accuracy (fraction of outcomes that are positive)
            acc = sum(y for _, y in items) / bin_size

            # Compute average predicted confidence in the bin
            conf = sum(p for p, _ in items) / bin_size

            # Accumulate weighted absolute difference
            ece += (bin_size / n) * abs(acc - conf)

        return ece

    def get_calibration_summary(self, predictions: List[float], outcomes: List[float]) -> Dict[str, Any]:
        """Returns a formatted calibration metrics summary."""
        brier = self.compute_brier_score(predictions, outcomes)
        ece = self.compute_expected_calibration_error(predictions, outcomes)
        return {
            "brier_score": round(brier, 4),
            "expected_calibration_error": round(ece, 4),
            "total_eval_samples": len(predictions),
            "is_calibrated": ece < 0.15
        }
