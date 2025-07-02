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
        self._switch_budget = self._scenario.walltime_limit * switch_budget_factor
        self._switched = False
        self._consumed_budget = 0.0
        self._runhistory: RunHistory | None = None

    def _update(self, **kwargs: Any) -> None:
        """
        This method is called by the abstract parent class to update the child
        acquisition functions and the consumed budget.
        """
        # The parent `update` method has already set self._model and self._runhistory.
        if self._model is None or self._runhistory is None:
            return

        # Get Y from kwargs to calculate the budget.
        Y = kwargs.get("Y")

        # Update consumed budget from the cost column of the Y matrix
        if Y is not None and Y.ndim == 2 and Y.shape[1] == 2:
            self._consumed_budget = np.sum(Y[:, 1])

        # Propagate the update call to children.
        # We must explicitly pass the model and runhistory that this object holds.
        # The remaining arguments (X, Y, eta, etc.) are in `kwargs`.
        self._initial_acquisition.update(
            model=self._model,
            runhistory=self._runhistory,
            consumed_budget=self._consumed_budget,
            **kwargs,
        )
        self._main_acquisition.update(
            model=self._model,
            runhistory=self._runhistory,
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
