from __future__ import annotations

import numpy as np

from smac.model.random_forest.random_forest import RandomForest


class LogRandomForest(RandomForest):
    """
    A Random Forest wrapper that trains on log-transformed targets but
    predicts in linear space.

    This is essential for Cost-Aware optimization where the acquisition
    function expects real units (seconds), not log-units, but the model
    needs log-transformation to handle the high dynamic range of runtimes.
    """

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predicts means in linear space by exponentiating the log-space predictions."""
        # Get predictions from the base RF.
        # Since we initialize with log_y=True, these are log(cost).
        log_means, log_vars = super()._predict(X, covariance_type)

        # Transform back to linear space: exp(log_cost) = cost
        # We use the median (exp(mu)) as the estimator for the cost.
        linear_means = np.exp(log_means)

        # We pass the variance through as-is or could transform it,
        # but CostAwareAcquisitionFunction primarily uses the mean.
        return linear_means, log_vars
