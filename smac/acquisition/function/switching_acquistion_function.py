from __future__ import annotations

from typing import Any, Optional

import numpy as np

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.logging import get_logger

logger = get_logger(__name__)


class SwitchingAcquisition(AbstractAcquisitionFunction):
    """
    A wrapper acquisition function that switches from an initial design strategy
    to a main acquisition function after a certain budget has been consumed.
    """

    def __init__(
        self,
        scenario: Scenario,
        initial_acquisition: AbstractAcquisitionFunction,
        main_acquisition: AbstractAcquisitionFunction,
        switch_budget_factor: float = 1 / 8,
    ):
        super().__init__()
        self._scenario = scenario
        self._runhistory: Optional[RunHistory] = None
        self._initial_acquisition = initial_acquisition
        self._main_acquisition = main_acquisition

        if self._scenario.walltime_limit is None:
            raise ValueError("SwitchingAcquisition requires a walltime_limit to be set in the Scenario.")

        self._switch_budget = self._scenario.walltime_limit * switch_budget_factor
        self._switched = False

    def set_runhistory(self, runhistory: RunHistory) -> None:
        """Sets the runhistory for this and the wrapped acquisition functions."""
        self._runhistory = runhistory

        # Pass the runhistory to the wrapped acquisition functions if they need it
        if hasattr(self._initial_acquisition, "set_runhistory"):
            self._initial_acquisition.set_runhistory(runhistory)  # type: ignore[attr-defined]

        if hasattr(self._main_acquisition, "set_runhistory"):
            self._main_acquisition.set_runhistory(runhistory)  # type: ignore[attr-defined]

    def _update(self, **kwargs: Any) -> None:
        """Updates both the initial and main acquisition functions."""
        # The model is set by the public `update` method before this is called.
        if self._model is None:
            raise RuntimeError("Model has not been set. Call `update` with a model first.")

        # Now, update the wrapped acquisition functions
        self._initial_acquisition.update(model=self._model, **kwargs)
        self._main_acquisition.update(model=self._model, **kwargs)

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the acquisition values. Delegates to the initial or main
        acquisition function depending on the consumed budget.
        """
        if self._runhistory is None:
            raise RuntimeError("Runhistory must be set before computing the acquisition value.")

        consumed_budget = self._runhistory.get_total_cost()

        if not self._switched and consumed_budget < self._switch_budget:
            return self._initial_acquisition._compute(X)
        else:
            if not self._switched:
                logger.info(
                    f"Budget for initial design ({self._switch_budget:.2f}s) exhausted. "
                    f"Switching to main acquisition function {self._main_acquisition.__class__.__name__}."
                )
                self._switched = True

            return self._main_acquisition._compute(X)
