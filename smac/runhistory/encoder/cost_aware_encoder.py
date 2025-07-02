from __future__ import annotations

from typing import Mapping

import numpy as np

from smac.runhistory.encoder import AbstractRunHistoryEncoder
from smac.runhistory.runhistory import TrialKey, TrialValue
from smac.utils.configspace import convert_configurations_to_array
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class RunHistoryCostAwareEncoder(AbstractRunHistoryEncoder):
    """
    Encoder for the CostAwareModel. It builds a data matrix `Y` with two columns:
    the first for the aggregated performance objective(s) and the second for the cost objective.
    """

    def _build_matrix(
        self,
        trials: Mapping[TrialKey, TrialValue],
        store_statistics: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if store_statistics:
            pass

        n_rows = len(trials)
        n_cols = self._n_params
        X = np.ones([n_rows, n_cols + self._n_features]) * np.nan
        y = np.ones([n_rows, 2])

        for row, (key, run) in enumerate(trials.items()):
            conf = self.runhistory.ids_config[key.config_id]
            conf_vector = convert_configurations_to_array([conf])[0]
            if self._n_features > 0 and self._instance_features is not None:
                assert isinstance(key.instance, str)
                feats = self._instance_features[key.instance]
                X[row, :] = np.hstack((conf_vector, feats))
            else:
                X[row, :] = conf_vector

            # Handle how performance and cost are extracted
            if self._n_objectives > 1:
                # Case 1: User provides a cost metric as the second objective.
                # The first objective is performance, the second is cost.
                assert (
                    isinstance(run.cost, list) and len(run.cost) == 2
                ), "Cost-aware mode expects the target function to return a list of [performance, cost]."
                y[row, 0] = run.cost[0]  # Performance
                y[row, 1] = run.cost[1]  # Cost
            else:
                # Case 2: Single performance objective, so cost is runtime.
                y[row, 0] = run.cost
                y[row, 1] = run.time

        y_transformed = self.transform_response_values(values=y)

        return X, y_transformed

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """
        Transform function response values. Transform the cost values (second column)
        by a log transformation log(1. + cost).
        """
        # We need to ensure that cost remains positive after the log transform.
        values[:, 1] = np.log(1 + values[:, 1])
        return values
