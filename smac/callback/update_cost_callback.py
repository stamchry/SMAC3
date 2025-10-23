from __future__ import annotations

from smac.acquisition.function.cost_aware_acquisition_function import (
    CostAwareAcquisitionFunction,
)
from smac.callback.callback import Callback
from smac.main.smbo import SMBO

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class UpdateCostCallback(Callback):
    """Callback to update the cost-aware acquisition function with budget info."""

    def __init__(
        self,
        acquisition_function: CostAwareAcquisitionFunction,
        total_budget: float,
        initial_design_budget: float,
        cumulative_cost_tracker: list[float],
    ):
        self._acquisition_function = acquisition_function
        self._total_budget = total_budget
        self._initial_design_budget = initial_design_budget
        self._cumulative_cost_tracker = cumulative_cost_tracker

    def on_iteration_start(self, smbo: SMBO) -> None:
        """
        Called before the next run is sampled. Updates the acquisition function
        with the current cumulative cost.
        """
        self._acquisition_function.set_budget_info(
            total_budget=self._total_budget,
            cumulative_cost=self._cumulative_cost_tracker[0],
            initial_design_budget=self._initial_design_budget,
        )
