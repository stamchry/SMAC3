from __future__ import annotations

from typing import Any, Optional

import numpy as np
from ConfigSpace import ConfigurationSpace

from smac.model.abstract_model import AbstractModel
from smac.model.random_forest.random_forest import RandomForest
from smac.model.hand_crafted_cost_model import HandCraftedCostModel
from smac.runhistory import RunHistory
from smac.scenario import Scenario


class CostAwareModel(AbstractModel):
    """A wrapper model that combines a performance model and a cost model."""

    def __init__(
        self,
        scenario: Scenario,
        **kwargs: Any,
    ):
        super().__init__(
            configspace=scenario.configspace,
            seed=scenario.seed,
        )
        self.performance_model = RandomForest(scenario.configspace, seed=scenario.seed, **kwargs)
        self.cost_model = HandCraftedCostModel(scenario.configspace, seed=scenario.seed)
        self._runhistory: Optional[RunHistory] = None

    def set_runhistory(self, runhistory: RunHistory) -> None:
        """Sets the runhistory, which is required to get cost data for training."""
        self._runhistory = runhistory

    def _train(self, X: np.ndarray, Y: np.ndarray) -> CostAwareModel:
        """Trains the performance model with Y and the cost model with data from runhistory."""
        if self._runhistory is None:
            raise RuntimeError("Runhistory must be set before training the CostAwareModel.")

        # Train performance model
        self.performance_model.train(X, Y)

        # Train cost model
        cost_X, cost_Y = self._runhistory.get_cost_data()
        if len(cost_X) > 0:
            self.cost_model.train(cost_X, cost_Y.reshape(-1, 1))

        return self

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict performance."""
        return self.performance_model.predict(X, covariance_type=covariance_type)

    def predict_cost(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict cost."""
        return self.cost_model.predict(X)

    @property
    def meta(self) -> dict[str, Any]:
        meta = super().meta
        meta.update(
            {
                "performance_model": self.performance_model.meta,
                "cost_model": self.cost_model.meta,
            }
        )
        return meta
