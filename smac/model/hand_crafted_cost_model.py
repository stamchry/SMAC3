from __future__ import annotations

from typing import Callable

import numpy as np
from ConfigSpace import Configuration

from smac.model.abstract_model import AbstractModel
from smac.scenario import Scenario
from smac.utils.logging import get_logger


class HandCraftedCostModel(AbstractModel):
    """
    A dummy cost model that uses a fixed formula instead of learning from data.

    The `train` method is a no-op. The `predict` method applies the given
    formula to the input configurations.
    """

    def __init__(
        self,
        scenario: Scenario,
        cost_formula: Callable[[Configuration], float],
    ):
        super().__init__(configspace=scenario.configspace, seed=scenario.seed)
        self._cost_formula = cost_formula
        self._logger = get_logger(self.__class__.__name__)

    def _train(self, X: np.ndarray, y: np.ndarray) -> HandCraftedCostModel:
        """The model is formulaic, so training is a no-op."""
        self._logger.info("HandCraftedCostModel does not learn from data. Skipping training.")
        return self

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Predicts costs by applying the formula.
        Variance is always zero as the model is deterministic.
        """
        # Convert numpy arrays back to Configuration objects to use the formula
        costs = np.array([self._cost_formula(Configuration(self._configspace, vector=x)) for x in X])

        # Variance is zero because the model is deterministic
        variances = np.zeros_like(costs)

        return costs, variances


class RLHandCraftedCostModel(HandCraftedCostModel):
    """
    A hand-crafted cost model for reinforcement learning hyperparameter optimization.

    This model computes the cost based on:
    - total timesteps (n_total_timesteps)
    - minibatch size (btchsz)
    - number of parallel environments (num_envs)
    - update epochs
    - MLP depth (num_mlp_layers)
    """

    def __init__(
        self,
        scenario: Scenario,
        cores: int | None = None,
        step_cost_per_depth: float = 1.0,
        grad_update_cost_per_depth: float = 1.0,
    ):
        """
        Parameters
        ----------
        scenario : Scenario
            The SMAC scenario
        cores : int | None, optional
            Number of CPU cores available for parallel environment execution.
            If None, automatically detects available cores.
        step_cost_per_depth : float, optional
            Cost multiplier per unit of MLP depth for environment steps
        grad_update_cost_per_depth : float, optional
            Cost multiplier per unit of MLP depth for gradient updates
        """
        # Automatically detect cores if not specified
        if cores is None:
            import os

            try:
                cores = len(os.sched_getaffinity(0))
            except AttributeError:
                cores = os.cpu_count() or 1
                self._logger.warning(f"Could not determine number of cores, using {cores}.")

        self._cores = cores
        self._step_cost_per_depth = step_cost_per_depth
        self._grad_update_cost_per_depth = grad_update_cost_per_depth

        # Initialize with the cost formula
        super().__init__(scenario=scenario, cost_formula=self._compute_cost)

    def _step_cost(self, mlp_depth: float) -> float:
        """Compute the cost of a single environment step."""
        return self._step_cost_per_depth * mlp_depth

    def _grad_update_cost(self, mlp_depth: float) -> float:
        """Compute the cost of a single gradient update."""
        return self._grad_update_cost_per_depth * mlp_depth

    def _compute_cost(self, config: Configuration) -> float:
        """
        Compute the cost for a given configuration using the algorithm.

        Cost formula:
        total_steps * grad_update_cost(mlp_depth * hidden_size) * epochs / batch_size
        + (total_steps / num_envs) * max(1, num_envs / cores) * step_cost(mlp_depth
        * hidden_size)

        Parameters
        ----------
        config : Configuration
            Configuration containing the hyperparameters

        Returns
        -------
        float
            The estimated cost
        """
        try:
            total_steps = config["environment.n_total_timesteps"]
            batch_size = config["hp_config.minibatch_size"]
            parallel_envs = config["environment.n_envs"]
            epochs = config["hp_config.update_epochs"]
            mlp_depth = config["nas_config.num_mlp_layers"]
            hidden_size = config["nas_config.hidden_size"]

        except KeyError as e:
            self._logger.error(f"Missing required hyperparameter in config: {e}")
            self._logger.error(f"Full config: {config}")
            self._logger.error(f"Config keys: {list(config.keys())}")
            for k in config.keys():
                self._logger.error(f"{k}: {config[k]}")
            raise

        nn_complexity = mlp_depth * hidden_size

        # Compute update cost
        update_cost = (total_steps * epochs / batch_size) * self._grad_update_cost(nn_complexity)

        # Compute collection cost
        collection_cost = (
            (total_steps / parallel_envs) * max(1, parallel_envs / self._cores) * self._step_cost(nn_complexity)
        )

        # Return total cost
        return update_cost + collection_cost
