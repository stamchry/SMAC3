from __future__ import annotations

import numpy as np
from ConfigSpace import ConfigurationSpace
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from smac.model.abstract_model import AbstractModel


class HandCraftedCostModel(AbstractModel):
    """A simple linear model for cost prediction."""

    def __init__(
        self,
        configspace: ConfigurationSpace,
        instance_features: dict[str, list[int | float]] | None = None,
    ):
        super().__init__(configspace=configspace, instance_features=instance_features)
        self._model = LinearRegression()

    def _train(self, X: np.ndarray, Y: np.ndarray) -> HandCraftedCostModel:
        self._model.fit(X, Y.flatten())
        return self

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        try:
            mean = self._model.predict(X)
        except NotFittedError:
            # If the model is not fitted, return a default cost of 1.
            mean = np.ones(X.shape[0])

        var = np.zeros(mean.shape)
        return mean.reshape(-1, 1), var.reshape(-1, 1)
