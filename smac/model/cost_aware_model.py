from __future__ import annotations

from typing import Any

import numpy as np

from smac.model.abstract_model import AbstractModel
from smac.model.hand_crafted_cost_model import HandCraftedCostModel
from smac.model.random_forest.random_forest import RandomForest
from smac.scenario import Scenario


class CostAwareModel(AbstractModel):
    """A wrapper model that combines a performance model and a cost model."""

    def __init__(
        self,
        scenario: Scenario,
        **kwargs: Any,
    ):
        super().__init__(configspace=scenario.configspace)
        self.performance_model = RandomForest(scenario.configspace, **kwargs)
        self.cost_model = HandCraftedCostModel(scenario.configspace)

    def _train(self, X: np.ndarray, Y: np.ndarray) -> CostAwareModel:
        """
        Trains the performance model with the first column of Y and the cost model
        with the second column of Y.
        """
        if Y.ndim != 2 or Y.shape[1] != 2:
            raise ValueError(
                "CostAwareModel expects a Y matrix with 2 columns (performance and cost), "
                f"but got a matrix with shape {Y.shape}."
            )

        # The first column is performance, the second is cost.
        # This data is prepared by the RunHistoryCostAwareEncoder.
        y_perf = Y[:, 0]
        y_cost = Y[:, 1]

        # Train performance model
        self.performance_model.train(X, y_perf)

        # Train cost model
        self.cost_model.train(X, y_cost)

        return self

    def predict_marginalized(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicts mean and variance marginalized over all instances.

        This method is intended to be used by acquisition functions like EI, which
        only care about the performance, not the cost. Therefore, this method
        delegates the prediction to the internal performance model.

        Warning
        -------
        The input data must not include any features.

        Parameters
        ----------
        X : np.ndarray [#samples, #hyperparameters]
            Input data points.

        Returns
        -------
        means : np.ndarray [#samples, 1]
            The predictive mean of the performance.
        vars : np.ndarray [#samples, 1]
            The predictive variance of the performance.
        """
        return self.performance_model.predict_marginalized(X)

    def predict_cost(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Predicts the mean and variance of the cost."""
        return self.cost_model.predict(X)

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        meta = super().meta
        meta.update(
            {
                "performance_model": self.performance_model.meta,
                "cost_model": self.cost_model.meta,
            }
        )
        return meta
