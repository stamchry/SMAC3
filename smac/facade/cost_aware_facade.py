from __future__ import annotations

from typing import Any

from smac.acquisition.function.expected_improvement import EICool
from smac.acquisition.function.initial_design_cost_aware_acquisition import (
    InitialDesignCostAwareAcquisition,
)
from smac.acquisition.function.switching_acquistion_function import SwitchingAcquisition
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade,
)
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.model.cost_aware_model import CostAwareModel
from smac.model.hand_crafted_cost_model import HandCraftedCostModel
from smac.model.random_forest.random_forest import RandomForest
from smac.runhistory.encoder.cost_aware_encoder import RunHistoryCostAwareEncoder
from smac.scenario import Scenario
from smac.utils.logging import get_logger

logger = get_logger(__name__)


class CostAwareHyperparameterOptimizationFacade(HyperparameterOptimizationFacade):
    """
    Facade to configure SMAC for cost-aware hyperparameter optimization.

    This facade overrides the default components with cost-aware alternatives:
    - A CostAwareModel that models both performance and cost.
    - A SwitchingAcquisition function that uses a cost-effective strategy
      for the initial design phase.
    - A minimal initial design, as the main initial design logic is now
      handled by the acquisition function.
    """

    @staticmethod
    def get_model(  # type: ignore[override]
        scenario: Scenario,
        *,
        n_trees: int = 10,
        ratio_features: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: int = 2**20,
        bootstrapping: bool = True,
    ) -> CostAwareModel:
        """Returns a cost-aware model that wraps a RandomForest and a cost model.

        Parameters
        ----------
        n_trees : int, defaults to 10
            The number of trees in the random forest.
        ratio_features : float, defaults to 1.0
            The ratio of features that are considered for splitting.
        min_samples_split : int, defaults to 2
            The minimum number of data points to perform a split.
        min_samples_leaf : int, defaults to 1
            The minimum number of data points in a leaf.
        max_depth : int, defaults to 2**20
            The maximum depth of a single tree.
        bootstrapping : bool, defaults to True
            Enables bootstrapping.
        """
        performance_model = RandomForest(
            log_y=True,
            n_trees=n_trees,
            bootstrapping=bootstrapping,
            ratio_features=ratio_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            configspace=scenario.configspace,
            instance_features=scenario.instance_features,
            seed=scenario.seed,
        )
        cost_model = HandCraftedCostModel(
            configspace=scenario.configspace,
            instance_features=scenario.instance_features,
        )

        return CostAwareModel(
            scenario=scenario,
            performance_model=performance_model,
            cost_model=cost_model,
        )

    @staticmethod
    def get_acquisition_function(  # type: ignore[override]
        scenario: Scenario,
        *,
        xi: float = 0.0,
    ) -> SwitchingAcquisition:
        """Returns a switching acquisition function that handles the cost-effective initial design."""
        initial_acquisition = InitialDesignCostAwareAcquisition(scenario)
        main_acquisition = EICool(scenario=scenario, xi=xi)

        return SwitchingAcquisition(
            scenario=scenario,
            initial_acquisition=initial_acquisition,
            main_acquisition=main_acquisition,
        )

    @staticmethod
    def get_initial_design(
        scenario: Scenario,
        *,
        n_configs: int | None = None,
        n_configs_per_hyperparamter: int = 10,
        max_ratio: float = 0.25,
        additional_configs: list[Any] | None = None,
    ) -> SobolInitialDesign:
        """
        The main "initial design" is now handled by the SwitchingAcquisition.
        We only need one configuration to start the main Bayesian optimization loop.
        """
        logger.info("Cost-aware mode: Using Sobol design to generate 1 initial point.")
        return SobolInitialDesign(
            scenario=scenario,
            n_configs=1,
            max_ratio=max_ratio,
        )

    @staticmethod
    def get_runhistory_encoder(scenario: Scenario) -> RunHistoryCostAwareEncoder:  # type: ignore[override]
        """Returns the runhistory encoder for the cost-aware model."""
        return RunHistoryCostAwareEncoder(scenario=scenario)
