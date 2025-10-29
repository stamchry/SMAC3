from __future__ import annotations

from typing import Any, Callable, Optional

import logging
from functools import wraps

from ConfigSpace import Configuration

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.function.cost_aware_acquisition_function import (
    CostAwareAcquisitionFunction,
)
from smac.acquisition.function.expected_improvement import EI
from smac.callback.budget_exhausted_callback import BudgetExhaustedCallback
from smac.callback.cost_surrogate_callback import CostSurrogateCallback
from smac.callback.update_cost_callback import UpdateCostCallback
from smac.facade.blackbox_facade import BlackBoxFacade
from smac.initial_design.abstract_initial_design import AbstractInitialDesign
from smac.model import AbstractModel
from smac.model.hand_crafted_cost_model import HandCraftedCostModel
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
        The target function to optimize. It must return a dictionary with two keys:
        `"performance"` for the performance value to be minimized, and `"cost"` for the resource cost.
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
        The acquisition function to use. If None, `EI` is used.
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
        *,
        cost_model: AbstractModel | None = None,
        cost_formula: Callable | None = None,
        initial_design: AbstractInitialDesign | None = None,
        initial_design_budget_ratio: float = 0.125,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ):
        self._total_resource_budget = total_resource_budget
        self._initial_design_budget = self._total_resource_budget * initial_design_budget_ratio
        self._logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

        # --- Wrap target function ---
        wrapped_target_function = self._wrap_target_function(target_function)

        # --- Handle Cost Model ---
        if cost_model is not None and cost_formula is not None:
            raise ValueError("Cannot provide both `cost_model` and `cost_formula`.")

        if cost_model is None:
            if cost_formula is not None:
                cost_model = HandCraftedCostModel(scenario=scenario, cost_formula=cost_formula)
            else:
                # We need a model for the cost, so we create a default one.
                from smac.facade.hyperparameter_optimization_facade import (
                    HyperparameterOptimizationFacade,
                )

                cost_model = HyperparameterOptimizationFacade.get_model(scenario)

        runhistory: Optional[RunHistory] = kwargs.get("runhistory")
        if runhistory is None:
            runhistory = RunHistory()
            kwargs["runhistory"] = runhistory

        # --- Default Component Creation ---
        if initial_design is None:
            from smac.initial_design.cost_aware_initial_design import (
                CostAwareInitialDesign,
            )

            initial_design = CostAwareInitialDesign(
                scenario=scenario,
                cost_model=cost_model,
                initial_budget=self._initial_design_budget,
                runhistory=runhistory,
            )

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
            assert cost_model is not None
            cost_surrogate_callback = CostSurrogateCallback(cost_model=cost_model, scenario=scenario)
            callbacks.append(cost_surrogate_callback)

        # --- Set up cost-aware acquisition and callbacks ---
        if acquisition_function is None:
            acquisition_function = CostAwareAcquisitionFunction(
                acquisition_function=EI(), cost_surrogate_callback=cost_surrogate_callback
            )

        # We use a list as a mutable tracker that can be shared between callbacks
        cumulative_cost_tracker = [0.0]

        # Add budget exhausted callback
        budget_callback = BudgetExhaustedCallback(
            total_resource_budget=self._total_resource_budget, cumulative_cost_tracker=cumulative_cost_tracker
        )
        callbacks.append(budget_callback)

        # If using the cost-aware acquisition function, add a callback to update it
        if isinstance(acquisition_function, CostAwareAcquisitionFunction):
            update_cost_callback = UpdateCostCallback(
                acquisition_function=acquisition_function,
                total_budget=self._total_resource_budget,
                initial_design_budget=self._initial_design_budget,
                cumulative_cost_tracker=cumulative_cost_tracker,
            )
            callbacks.append(update_cost_callback)

        kwargs["callbacks"] = callbacks

        super().__init__(
            scenario=scenario,
            target_function=wrapped_target_function,
            initial_design=initial_design,
            acquisition_function=acquisition_function,
            overwrite=overwrite,
            **kwargs,
        )

    def _wrap_target_function(self, target_function: Callable) -> Callable:
        @wraps(target_function)
        def wrapper(config: Configuration, **kwargs: Any) -> tuple[float, dict[str, float]]:
            result = target_function(config, **kwargs)
            performance, cost = result["performance"], result["cost"]
            additional_info = {"resource_cost": cost}

            return performance, additional_info

        return wrapper
