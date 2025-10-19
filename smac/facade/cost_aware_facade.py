from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import logging

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.function.cost_aware_acquisition_function import (
    CostAwareAcquisitionFunction,
)
from smac.acquisition.function.expected_improvement import EI
from smac.callback.cost_surrogate_callback import CostSurrogateCallback
from smac.facade.blackbox_facade import BlackBoxFacade
from smac.initial_design.abstract_initial_design import AbstractInitialDesign
from smac.model import AbstractModel
from smac.model.hand_crafted_cost_model import HandCraftedCostModel
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario


class CostAwareFacade(BlackBoxFacade):
    """
    A facade to configure SMAC for cost-aware black-box optimization.
    This facade encapsulates the budget-based optimization loop and provides defaults for cost-aware components.

    Parameters
    ----------
    scenario : Scenario
        The scenario object, holding all environmental information.
    target_function : Callable
        The target function to optimize. It must return a tuple of (performance, cost).
    total_resource_budget : float
        The total budget for the optimization in terms of resource/cost.
    cost_model : AbstractModel | None, defaults to None
        The cost model to predict the cost of configurations.
    cost_formula : Callable | None, defaults to None
        A callable that calculates the cost of a configuration. Used if `cost_model` is not provided.
    initial_design : AbstractInitialDesign | None, defaults to None
        The initial design strategy. If None, `CostAwareInitialDesign` is used.
    initial_design_budget_ratio : float, defaults to 0.125
        The fraction of `total_resource_budget` to be used for the initial design.
    acquisition_function : AbstractAcquisitionFunction | None, defaults to None
        The acquisition function to use. If None, `CostAwareAcquisitionFunction` wrapping `EI` is used.
    runhistory : RunHistory | None, defaults to None
        The runhistory to store the trials. If None, a new runhistory is created.
    overwrite : bool, defaults to False
        If True, the output directory will be overwritten.
    **kwargs:
        Additional keyword arguments passed to the ``BlackBoxFacade``.
    """

    def __init__(
        self,
        scenario: Scenario,
        target_function: Callable,
        total_resource_budget: float,
        cost_model: AbstractModel | None = None,
        cost_formula: Callable | None = None,
        initial_design: AbstractInitialDesign | None = None,
        initial_design_budget_ratio: float = 0.125,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ):
        self.target_function = target_function
        self._total_resource_budget = total_resource_budget
        self._logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

        # --- Handle Cost Model ---
        if cost_model is not None and cost_formula is not None:
            raise ValueError("Cannot provide both `cost_model` and `cost_formula`.")

        if cost_model is None and cost_formula is None:
            raise ValueError("Must provide either `cost_model` or `cost_formula` for cost-aware optimization.")

        if cost_model is None and cost_formula is not None:
            cost_model = HandCraftedCostModel(scenario=scenario, cost_formula=cost_formula)

        runhistory = RunHistory()

        # --- Default Component Creation ---
        assert cost_model is not None  # For mypy
        if initial_design is None:
            from smac.initial_design.cost_aware_initial_design import (
                CostAwareInitialDesign,
            )

            initial_design_budget = total_resource_budget * initial_design_budget_ratio
            initial_design = CostAwareInitialDesign(
                scenario=scenario,
                cost_model=cost_model,
                initial_budget=initial_design_budget,
                runhistory=runhistory,
            )

        # Use a local variable for the acquisition function logic
        acq_function = acquisition_function

        # Check if a CostSurrogateCallback was already provided by the user.
        # If not, create a default one.
        callbacks = kwargs.get("callbacks", [])
        cost_surrogate_callback = None
        for cb in callbacks:
            if isinstance(cb, CostSurrogateCallback):
                cost_surrogate_callback = cb
                self._logger.debug("Using user-provided CostSurrogateCallback.")
                break

        if cost_surrogate_callback is None:
            self._logger.debug("No CostSurrogateCallback found, creating a default one.")
            cost_surrogate_callback = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)
            callbacks.append(cost_surrogate_callback)

        kwargs["callbacks"] = callbacks

        if acq_function is None:
            acq_function = CostAwareAcquisitionFunction(
                acquisition_function=EI(), cost_surrogate_callback=cost_surrogate_callback
            )
        # --- End Default Component Creation ---

        super().__init__(
            scenario=scenario,
            target_function=target_function,
            initial_design=initial_design,
            acquisition_function=acq_function,
            overwrite=overwrite,
            runhistory=runhistory,
            **kwargs,
        )

    def optimize(self, *, data_to_scatter: Optional[Dict[str, Any]] = None) -> Union[Any, List[Any]]:
        """Optimizes the target function within the given total resource budget."""
        cumulative_cost = 0.0
        initial_design_budget = 0.0

        self._logger.info("\n--- Starting Budget-Based Optimization Loop ---")

        while cumulative_cost < self._total_resource_budget:
            # Set the budget info on our acquisition function directly
            if isinstance(self._acquisition_function, CostAwareAcquisitionFunction):
                self._acquisition_function.set_budget_info(
                    total_budget=self._total_resource_budget,
                    cumulative_cost=cumulative_cost,
                    initial_design_budget=initial_design_budget,
                )

            trial_info: TrialInfo | None = self.ask()

            if trial_info is None:
                self._logger.info("SMAC has no more configurations to suggest. Stopping.")
                break

            # The target function for cost-aware optimization returns (cost, time)
            performance, cost = self.target_function(trial_info.config)

            if cumulative_cost + cost > self._total_resource_budget:
                self._logger.info(f"Evaluation cost ({cost:.2f}) would exceed total budget. Stopping.")
                break

            cumulative_cost += cost
            self._logger.info(
                f"Origin: {trial_info.config.origin}, Cost: {cost:.2f}, "
                f"Cumulative Cost: {cumulative_cost:.2f}/{self._total_resource_budget:.2f}"
            )

            self.tell(trial_info, TrialValue(cost=performance, time=cost))

        self._logger.info("\n--- Total resource budget exhausted. ---")
        return self.intensifier.get_incumbent()
