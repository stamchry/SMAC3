from __future__ import annotations

from typing import Any

import numpy as np
from ConfigSpace import Configuration
from scipy.spatial.distance import cdist

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.model.hand_crafted_cost_model import HandCraftedCostModel
from smac.scenario import Scenario


class InitialDesignCostAwareAcquisition(AbstractAcquisitionFunction):
    """
    This acquisition function implements the 'Algorithm 1 Cost-effective initial design'
    as a procedural acquisition function. It guides the acquisition maximizer to select
    the single most cost-effective configuration from a set of candidates.
    """

    def __init__(self, scenario: Scenario):
        super().__init__()
        self._scenario = scenario
        self._cost_model = HandCraftedCostModel(self._scenario.configspace)
        self._X_init_arrays: np.ndarray | None = None

    def _update(self, **kwargs: Any) -> None:
        """
        Updates the cost model with the latest data from the Y matrix and stores
        the evaluated configurations (X) for later use in _compute.
        """
        X = kwargs.get("X")
        Y = kwargs.get("Y")

        if Y is None or Y.ndim != 2 or Y.shape[1] != 2:
            return

        # The second column of Y is the cost/runtime
        y_cost = Y[:, 1]
        self._cost_model.train(X, y_cost.reshape(-1, 1))

        # Store the configurations that have been evaluated
        self._X_init_arrays = X

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """
        Takes a set of candidate configurations and returns acquisition values
        that lead to the selection of the single most cost-effective point.
        """
        if X.shape[0] <= 1:
            return np.ones((X.shape[0], 1))

        candidate_configs = [Configuration(self._scenario.configspace, vector=x) for x in X]
        candidate_arrays = np.copy(X)

        while len(candidate_configs) > 1:
            # 1. Exclude most expensive point
            costs, _ = self._cost_model.predict(candidate_arrays)
            max_cost_idx = np.argmax(costs)

            candidate_configs.pop(max_cost_idx)
            candidate_arrays = np.delete(candidate_arrays, max_cost_idx, axis=0)

            if len(candidate_configs) == 1:
                break

            # 2. Exclude point closest to X_init
            if self._X_init_arrays is not None and self._X_init_arrays.shape[0] > 0:
                distances = cdist(candidate_arrays, self._X_init_arrays)
                min_dist_to_init = np.min(distances, axis=1)
                closest_idx = np.argmin(min_dist_to_init)

                candidate_configs.pop(closest_idx)
                candidate_arrays = np.delete(candidate_arrays, closest_idx, axis=0)

        winner_config = candidate_configs[0]
        winner_vector = winner_config.get_array()

        acq_values = np.zeros(X.shape[0])
        for i, x_vec in enumerate(X):
            if np.allclose(x_vec, winner_vector):
                acq_values[i] = 1.0
                break

        return acq_values.reshape(-1, 1)
