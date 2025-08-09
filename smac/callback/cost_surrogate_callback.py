from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration

from smac.callback import Callback
from smac.main.smbo import SMBO
from smac.model.abstract_model import AbstractModel
from smac.runhistory.dataclasses import TrialInfo, TrialValue


class CostSurrogateCallback(Callback):
    """
    Callback to train a cost model on completed trials. This is essential for the
    main optimization loop after the initial design is finished.
    """

    def __init__(self, cost_model: AbstractModel):
        self._cost_model = cost_model
        self._history: list[tuple[Configuration, float]] = []

    @property
    def cost_model(self) -> AbstractModel:
        """Returns the cost model."""
        return self._cost_model

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        """
        This method is called after a real trial is completed. It extracts the cost,
        updates the history, and retrains the cost model with real data.
        """
        evaluation_cost = value.time
        config = info.config

        if "evaluation_cost" in value.additional_info:
            evaluation_cost = value.additional_info["evaluation_cost"]

        self._history.append((config, evaluation_cost))

        # Retrain the model on real data
        if self._history:
            configs, costs = zip(*self._history)
            X = np.array([c.get_array() for c in configs])
            Y = np.array(costs)
            self._cost_model.train(X, Y)

        return None
