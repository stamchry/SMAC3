from __future__ import annotations

import logging
import time

from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter

from smac.facade.cost_aware_facade import (
    CostAwareHyperparameterOptimizationFacade,
)
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade,
)
from smac.scenario import Scenario

# Configure logging to see SMAC's output, including the acquisition function switch
logging.basicConfig(level=logging.INFO)


def my_target_function(config: Configuration) -> dict[str, float]:
    """
    A simple target function to be minimized.

    The performance (loss) is a simple quadratic function with a minimum at x=5.
    The cost (runtime) increases quadratically with x, making higher values
    of x more expensive to evaluate.
    """
    x = config["x"]

    # The cost is a function of the hyperparameter x
    cost = 0.5 + (x**2 / 10)
    loss = (x - 5) ** 2

    # Simulate the evaluation time
    time.sleep(cost)

    # SMAC expects a dictionary with at least a "cost" and a "loss" key
    # when doing cost-aware optimization.
    return {"cost": cost, "loss": loss}


if __name__ == "__main__":
    # 1. Define the hyperparameter space
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformFloatHyperparameter("x", 0, 10, default_value=5))

    # 2. Define the scenario
    # To enable cost-aware optimization, you must:
    #   - Set `cost_aware=True`
    #   - Provide a `walltime_limit` (in seconds)
    scenario = Scenario(
        configspace,
        # --- Key parameters for cost-aware optimization ---
        cost_aware=True,
        walltime_limit=45,  # Total budget in seconds
        # ---
        n_trials=50,  # Max number of evaluations
        min_trials=5,  # Min number of evaluations
        seed=1,
    )

    # 3. Select the appropriate facade
    # This demonstrates the intended usage pattern. Your script can choose the
    # facade based on whether cost-awareness is needed.
    if scenario.cost_aware:
        print("Using Cost-Aware Hyperparameter Optimization Facade.")
        smac = CostAwareHyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=my_target_function,
        )
    else:
        print("Using Standard Hyperparameter Optimization Facade.")
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=my_target_function,
        )

    # 4. Run the optimization
    # During the run, you should see a log message like:
    # "Budget for initial design (5.62s) exhausted. Switching to main acquisition function EICool."
    incumbent = smac.optimize()

    # 5. Print the results
    print("\n--- Results ---")
    print(f"Incumbent configuration: {incumbent}")
    
    # Get the cost and loss of the incumbent
    incumbent_cost = smac.runhistory.get_cost(incumbent)
    incumbent_loss = smac.runhistory.get_instance_costs_for_config(incumbent)[0]
    print(f"Incumbent cost: {incumbent_cost:.2f}s")
    print(f"Incumbent loss: {incumbent_loss:.2f}")
    print("----------------")