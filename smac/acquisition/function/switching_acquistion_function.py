from __future__ import annotations

from typing import Any

import numpy as np

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.logging import get_logger

logger = get_logger(__name__)


class SwitchingAcquisition(AbstractAcquisitionFunction):
    """Switches between two acquisition functions based on a budget."""

    def __init__(
        self,
        scenario: Scenario,
        initial_acquisition: AbstractAcquisitionFunction,
        main_acquisition: AbstractAcquisitionFunction,
        switch_budget_factor: float = 1 / 8,
    ):
        super().__init__()
        self._scenario = scenario
        self._initial_acquisition = initial_acquisition
        self._main_acquisition = main_acquisition

        if not self._scenario.cost_aware or self._scenario.cost_aware_budget is None:
            raise ValueError(
                "`cost_aware` must be True and `cost_aware_budget` must be set in the scenario "
                "to use SwitchingAcquisition."
            )

        self._switch_budget = self._scenario.cost_aware_budget * switch_budget_factor
        self._switched = False
        self._consumed_budget = 0.0
        self._runhistory: RunHistory | None = None

    def _update(self, **kwargs: Any) -> None:
        """
        This method is called by the abstract parent class to update the child
        acquisition functions and the consumed budget.
        """
        # Get Y from kwargs to calculate the budget.
        Y = kwargs.get("Y")

        # Update consumed budget from the cost column of the Y matrix
        # The cost is log-transformed in the encoder, so we need to reverse it.
        if Y is not None and Y.ndim == 2 and Y.shape[1] == 2:
            # The encoder does log(1 + cost), so we do exp(y) - 1
            # We calculate the budget from all trials, not just the last batch.
            self._consumed_budget = np.sum(np.exp(Y[:, 1]) - 1)
            logger.info(f"Consumed budget: {self._consumed_budget:.4f}, " f"Switch budget: {self._switch_budget:.4f}")

        # Propagate the update call to children.
        # We must explicitly pass the model and runhistory that this object holds.
        # The remaining arguments (X, Y, eta, etc.) are in `kwargs`.
        if self._model is None:
            raise ValueError("Model has not been updated yet.")

        self._initial_acquisition.update(
            model=self._model,
            consumed_budget=self._consumed_budget,
            **kwargs,
        )
        self._main_acquisition.update(
            model=self._model,
            consumed_budget=self._consumed_budget,
            **kwargs,
        )

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the acquisition value for a given point X."""
        if not self._switched and self._consumed_budget < self._switch_budget:
            return self._initial_acquisition._compute(X)
        else:
            if not self._switched:
                logger.info(
                    f"Budget for initial design ({self._switch_budget:.2f}s) exhausted. "
                    f"Switching to main acquisition function {self._main_acquisition.__class__.__name__}."
                )
                self._switched = True

            return self._main_acquisition._compute(X)
