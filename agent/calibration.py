"""
Confidence Calibration and Verification module for Project Karthikeya.
Implements computational models for:
- Brier Score calculation
- Expected Calibration Error (ECE) calculation
- Wilson Score Confidence Interval calculation
- Strategy Learning exploitation metrics
"""
import math
from typing import Any


class ConfidenceCalibrator:
    """
    Evaluates how closely predicted planner/execution confidence scores align
    with actual sandbox success probabilities using Brier, Expected Calibration Error (ECE),
    and Wilson Score Intervals.
    """
    def __init__(self, num_bins: int = 5) -> None:
        self.num_bins = num_bins

    def calculate_wilson_score_interval(self, k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
        """
        Computes the Wilson Score Interval for binomial proportion from scratch.
        Prevents Wald/normal approximation breakdown at extreme boundaries (k=0 or k=n).
        """
        if n == 0:
            return 0.0, 0.0

        z = 1.95996
        p = k / n

        denominator = 1.0 + (z ** 2) / n
        center = (p + (z ** 2) / (2 * n)) / denominator
        spread = (z / denominator) * math.sqrt((p * (1.0 - p) / n) + (z ** 2) / (4 * (n ** 2)))

        lower = max(0.0, center - spread)
        upper = min(1.0, center + spread)
        return lower, upper

    def compute_brier_score(self, predictions: list[float], outcomes: list[float]) -> float:
        """
        Computes the standard Brier Score for probability calibration:
        BS = 1/N * sum((p_t - y_t)^2)
        """
        if not predictions or not outcomes or len(predictions) != len(outcomes):
            return 0.0
        total_error = sum((p - y) ** 2 for p, y in zip(predictions, outcomes))
        return total_error / len(predictions)

    def compute_expected_calibration_error(self, predictions: list[float], outcomes: list[float]) -> float:
        """
        Computes the Expected Calibration Error (ECE) stably across partitioned bins:
        ECE = sum(|B_m|/N * |acc(B_m) - conf(B_m)|)
        """
        n = len(predictions)
        if n == 0 or len(predictions) != len(outcomes):
            return 0.0

        bins_indices = {i: [] for i in range(self.num_bins)}

        for p, y in zip(predictions, outcomes):
            if p >= 1.0:
                bin_idx = self.num_bins - 1
            else:
                bin_idx = int(p * self.num_bins)
            bins_indices[bin_idx].append((p, y))

        ece = 0.0
        for items in bins_indices.values():
            bin_size = len(items)
            if bin_size == 0:
                continue

            acc = sum(y for _, y in items) / bin_size
            conf = sum(p for p, _ in items) / bin_size
            ece += (bin_size / n) * abs(acc - conf)

        return ece

    def get_calibration_summary(self, predictions: list[float], outcomes: list[float]) -> dict[str, Any]:
        """Returns a formatted calibration metrics summary."""
        brier = self.compute_brier_score(predictions, outcomes)
        ece = self.compute_expected_calibration_error(predictions, outcomes)
        return {
            "brier_score": round(brier, 4),
            "expected_calibration_error": round(ece, 4),
            "total_eval_samples": len(predictions),
            "is_calibrated": ece < 0.15
        }
