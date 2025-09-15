from __future__ import annotations

from typing import Any

import logging
from collections import OrderedDict

import numpy as np
from ConfigSpace import Configuration
from scipy.spatial.distance import cdist

from smac.initial_design.abstract_initial_design import AbstractInitialDesign
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.model.abstract_model import AbstractModel
from smac.scenario import Scenario


class CostAwareInitialDesign(AbstractInitialDesign):
    """
    Selects initial configurations by balancing exploration and cost.

    This strategy aims to pick a set of starting configurations that are diverse
    (spread out in the configuration space) and cheap, staying within a given
    initial budget. It implements the "Cost-effective initial design" algorithm.

    The algorithm works by running an elimination process in each round to select
    one new configuration. This process repeatedly eliminates candidate
    configurations that are either too expensive or too similar to points already
    chosen, until only one suitable candidate remains.
    """

    def __init__(
        self,
        scenario: Scenario,
        cost_model: AbstractModel,
        initial_budget: float,
        candidate_pool_size: int = 1000,
        candidate_generator: type[AbstractInitialDesign] = SobolInitialDesign,
        **kwargs: Any,
    ):
        super().__init__(scenario=scenario, n_configs=None, **kwargs)
        self._scenario = scenario
        self._cost_model = cost_model
        self._initial_budget = initial_budget
        self._candidate_pool_size = candidate_pool_size
        self._logger = logging.getLogger(self.__class__.__name__)
        self._candidate_generator = candidate_generator

        """if candidate_generator is None:
            self._candidate_generator = SobolInitialDesign
        else:
            self._candidate_generator = candidate_generator"""

    def _select_configurations(self) -> list[Configuration]:
        """
        Generates a set of cost-aware initial configurations following Algorithm 1.

        This method contains the main logic:
        1.  (Step 2) Initializes cumulative time and the list for the initial design.
        2.  (Step 3) A large pool of random candidate configurations is sampled.
        3.  (Step 4) It then loops until the initial budget is spent:
            a. (Step 5) An elimination process begins on the set of available candidates.
               This process repeatedly removes the most expensive candidate and
               the candidate closest to the already-selected design points.
            b. The single remaining candidate is chosen for the initial design.
            c. (Step 10) The cost of this new configuration is added to the total, and the
               cost model is retrained with the new information.
        4.  (Step 12) The final list of chosen configurations is returned.
        """
        # Step 2: Initialize cumulative time (ct) and initial design (Xinit).
        cumulative_time = 0.0
        selected_configs: list[Configuration] = []
        selected_arrays: list[np.ndarray] = []
        simulated_history: list[tuple[np.ndarray, np.ndarray]] = []

        # Step 3: Discretize Ω into ˜Ω using the specified candidate generator.
        generator = self._candidate_generator(
            scenario=self._scenario,
            n_configs=self._candidate_pool_size,
            max_ratio=1.0,  # This ensures the pool size is not reduced.
        )
        candidate_pool_raw = generator.select_configurations()

        # Use OrderedDict to efficiently find and keep only the unique configurations.
        discretized_space = list(
            OrderedDict((tuple(config.get_array()), config) for config in candidate_pool_raw).values()
        )

        self._logger.info(
            f"Generated {len(candidate_pool_raw)} candidates, resulting in "
            f"{len(discretized_space)} unique configurations."
        )

        if not discretized_space:
            return []

        # To speed things up, convert all configuration objects to numpy arrays at once.
        all_config_arrays = np.array([c.get_array() for c in discretized_space])

        # We'll use a set of indices to efficiently track which candidates are still available.
        remaining_indices = set(range(len(discretized_space)))

        # Step 4: Main loop `while ct < τinit do`
        iteration = 0
        while cumulative_time < self._initial_budget and remaining_indices:
            iteration += 1

            if iteration % 10 == 0:  # Log progress periodically.
                self._logger.info(
                    f"Iteration {iteration}: Budget used {cumulative_time:.1f}/{self._initial_budget:.1f}, "
                    f"Candidates remaining: {len(remaining_indices)}"
                )

            # --- Candidate Elimination Process ---

            # Before starting, predict the cost for all currently available candidates.
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

            # Step 5: Inner loop `while size Xcand > 1 do`
            # Create a copy of the available candidates to prune in this round.
            pruning_candidates = set(remaining_indices)
            while len(pruning_candidates) > 1:
                # Step 6: Remove the most expensive candidate.
                if len(pruning_candidates) > 1:
                    most_expensive_idx = max(pruning_candidates, key=lambda i: costs_dict[i])
                    pruning_candidates.remove(most_expensive_idx)

                if len(pruning_candidates) <= 1:
                    break

                # Step 7: Remove the candidate closest to our already-selected points.
                if distances_dict:
                    # Consider only the candidates still in this round.
                    filtered_distances = {k: v for k, v in distances_dict.items() if k in pruning_candidates}
                    if filtered_distances:
                        # Using a lambda function is more explicit for the type checker.
                        closest_idx = min(filtered_distances, key=lambda k: filtered_distances[k])
                        pruning_candidates.remove(closest_idx)
                    else:
                        # Fallback: if no candidates with distances are left, remove an arbitrary one.
                        pruning_candidates.pop()
                else:
                    # If we haven't selected any points yet, just remove an arbitrary candidate.
                    pruning_candidates.pop()

            # Step 8: `end while`
            # The last remaining candidate is our chosen one for this iteration.
            if pruning_candidates:
                # Step 9: Add remaining point to Xinit and evaluate (predict cost).
                chosen_idx = next(iter(pruning_candidates))
                chosen_config = discretized_space[chosen_idx]

                # Get the predicted cost for the chosen configuration.
                chosen_array = all_config_arrays[chosen_idx]
                cost_pred, _ = self._cost_model.predict(chosen_array.reshape(1, -1))
                simulated_cost = float(cost_pred[0])

                # Stop if adding this configuration would exceed our budget.
                if cumulative_time + simulated_cost > self._initial_budget and selected_configs:
                    self._logger.info(f"Next configuration cost ({simulated_cost:.2f}) would exceed budget. Stopping.")
                    break

                # Add the chosen configuration to our initial design.
                selected_configs.append(chosen_config)
                selected_arrays.append(chosen_array)

                # Permanently remove the chosen configuration from the pool of candidates.
                remaining_indices.remove(chosen_idx)

                # Step 10: Update ct, cost surrogate.
                cumulative_time += simulated_cost

                # Add the new data point to our history for retraining.
                simulated_history.append((chosen_array, np.array([simulated_cost])))

                # Retrain the cost model with the new data point after each selection.
                if len(simulated_history) >= 2:
                    X_hist, Y_hist = zip(*simulated_history)
                    self._cost_model.train(np.array(X_hist), np.array(Y_hist))

            else:
                self._logger.warning("No candidates left after elimination process. Stopping.")
                break

        # Step 11: `end while`
        self._initial_budget_spent = cumulative_time
        self._logger.info(
            f"Cost-aware initial design finished. Selected {len(selected_configs)} configurations "
            f"with a total cost of {cumulative_time:.2f}/{self._initial_budget:.2f}."
        )

        # Step 12: Return Xinit.
        # Before returning, we reset the origin of the selected configurations to correctly
        # attribute them to this initial design method.
        final_configs = []
        for config in selected_configs:
            new_config = Configuration(
                configuration_space=self._configspace,
                values=dict(config),
                origin="Cost Aware Initial Design",
            )
            final_configs.append(new_config)

        return final_configs
