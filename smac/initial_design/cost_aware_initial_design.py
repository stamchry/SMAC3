#WORKS AND IS FAST

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from ConfigSpace import Configuration
from scipy.spatial.distance import cdist
from collections import OrderedDict

from smac.initial_design.abstract_initial_design import AbstractInitialDesign
from smac.model.abstract_model import AbstractModel
from smac.scenario import Scenario


class CostAwareInitialDesign(AbstractInitialDesign):
    """
    Selects initial configurations by balancing exploration and cost.

    This strategy aims to pick a set of starting configurations that are diverse
    (spread out in the configuration space) and cheap, staying within a given
    initial budget. It implements the "Cost-effective initial design" algorithm.

    The algorithm works by running a "pruning tournament" in each round to select
    one new configuration. This tournament repeatedly eliminates candidate
    configurations that are either too expensive or too similar to points already
    chosen, until only the most suitable candidate remains.
    """

    def __init__(
        self,
        scenario: Scenario,
        cost_model: AbstractModel,
        initial_budget: float,
        candidate_pool_size: int = 1000,
        **kwargs: Any,
    ):
        super().__init__(scenario=scenario, n_configs=None, **kwargs)
        self._cost_model = cost_model
        self._initial_budget = initial_budget
        self._candidate_pool_size = candidate_pool_size
        self._logger = logging.getLogger(self.__class__.__name__)

    def _select_configurations(self) -> list[Configuration]:
        """
        Generates a set of cost-aware initial configurations.

        This method contains the main logic:
        1.  A large pool of random candidate configurations is sampled from the space.
        2.  It then loops until the initial budget is spent:
            a. A "pruning tournament" is run on the set of available candidates.
               This tournament repeatedly removes the most expensive candidate and
               the candidate closest to the already-selected design points.
            b. The last candidate standing in the tournament is chosen for the initial design.
            c. The cost of this new configuration is added to the total, and the
               cost model is retrained with the new information.
        3.  The final list of chosen configurations is returned.
        """
        # Step 1: Create a pool of unique candidate configurations from the search space.
        candidate_pool_raw = self._configspace.sample_configuration(size=self._candidate_pool_size)
        if not isinstance(candidate_pool_raw, list):
            candidate_pool_raw = [candidate_pool_raw]

        # Use OrderedDict to efficiently find and keep only the unique configurations.
        discretized_space = list(OrderedDict(
            (tuple(config.get_array()), config) for config in candidate_pool_raw
        ).values())

        self._logger.info(
            f"Generated {len(candidate_pool_raw)} candidates, resulting in "
            f"{len(discretized_space)} unique configurations."
        )

        if not discretized_space:
            return []

        # To speed things up, convert all configuration objects to numpy arrays at once.
        all_config_arrays = np.array([c.get_array() for c in discretized_space])
        
        # Initialize our budget, the list of configurations we'll select, and a history for retraining the cost model.
        cumulative_time = 0.0
        selected_configs: list[Configuration] = []
        selected_arrays: list[np.ndarray] = []
        simulated_history: list[tuple[np.ndarray, np.ndarray]] = []
        
        # We'll use a set of indices to efficiently track which candidates are still available.
        remaining_indices = set(range(len(discretized_space)))

        # Main loop: keep selecting configurations until we run out of budget or candidates.
        iteration = 0
        while cumulative_time < self._initial_budget and remaining_indices:
            iteration += 1
            
            if iteration % 10 == 0:  # Log progress periodically.
                self._logger.info(
                    f"Iteration {iteration}: Budget used {cumulative_time:.1f}/{self._initial_budget:.1f}, "
                    f"Candidates remaining: {len(remaining_indices)}"
                )

            # --- The Pruning Tournament ---
            # This is the core of the algorithm where we select the best candidate.
            
            # Before the tournament, predict the cost for all currently available candidates.
            if len(remaining_indices) > 1:
                current_indices_list = list(remaining_indices)
                current_arrays = all_config_arrays[current_indices_list]
                current_costs, _ = self._cost_model.predict(current_arrays)
                costs_dict = dict(zip(current_indices_list, current_costs.flatten()))
                
                # Also, calculate how close each candidate is to the configurations we've already selected.
                if selected_arrays:
                    selected_matrix = np.array(selected_arrays)
                    distances = cdist(current_arrays, selected_matrix)
                    min_distances = np.min(distances, axis=1)
                    distances_dict = dict(zip(current_indices_list, min_distances))
                else:
                    distances_dict = {}

            # Create a copy of the available candidates to prune in the tournament.
            pruning_candidates = set(remaining_indices)
            while len(pruning_candidates) > 1:
                # Step 6: Remove the most expensive candidate from the tournament.
                if len(pruning_candidates) > 1:
                    most_expensive_idx = max(pruning_candidates, key=lambda i: costs_dict[i])
                    pruning_candidates.remove(most_expensive_idx)
                
                if len(pruning_candidates) <= 1:
                    break
                    
                # Step 7: Remove the candidate closest to our already-selected points.
                if distances_dict:
                    # Consider only the candidates still in the tournament.
                    filtered_distances = {k: v for k, v in distances_dict.items() if k in pruning_candidates}
                    if filtered_distances:
                        closest_idx = min(filtered_distances, key=filtered_distances.get)
                        pruning_candidates.remove(closest_idx)
                    else:
                        # Fallback: if no candidates with distances are left, remove an arbitrary one.
                        pruning_candidates.pop()
                else:
                    # If we haven't selected any points yet, just remove an arbitrary candidate.
                    pruning_candidates.pop()

            # --- Tournament End ---
            # The last remaining candidate is our winner.
            if pruning_candidates:
                chosen_idx = next(iter(pruning_candidates))
                chosen_config = discretized_space[chosen_idx]
                
                # Get the predicted cost for the winning configuration.
                chosen_array = all_config_arrays[chosen_idx]
                cost_pred, _ = self._cost_model.predict(chosen_array.reshape(1, -1))
                simulated_cost = float(cost_pred[0])
                
                # Stop if adding this configuration would exceed our budget.
                if cumulative_time + simulated_cost > self._initial_budget and selected_configs:
                    self._logger.info(
                        f"Next configuration cost ({simulated_cost:.2f}) would exceed budget. Stopping."
                    )
                    break
                
                # Add the winner to our initial design.
                selected_configs.append(chosen_config)
                selected_arrays.append(chosen_array)
                
                # Permanently remove the chosen configuration from the pool of candidates.
                remaining_indices.remove(chosen_idx)
                
                # Step 10: Update the total cost and retrain the cost model.
                cumulative_time += simulated_cost
                
                # Add the new data point to our history for retraining.
                simulated_history.append((chosen_array, np.array([simulated_cost])))
                
                # Retrain the cost model with the new data point after each selection.
                if len(simulated_history) >= 2:
                    X_hist, Y_hist = zip(*simulated_history)
                    self._cost_model.train(np.array(X_hist), np.array(Y_hist))

            else:
                self._logger.warning("Tournament ended with no candidates left. Stopping.")
                break

        self._logger.info(
            f"Cost-aware initial design finished. Selected {len(selected_configs)} configurations "
            f"with a total cost of {cumulative_time:.2f}/{self._initial_budget:.2f}."
        )
        return selected_configs