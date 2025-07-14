from __future__ import annotations

import logging

import numpy as np
import matplotlib.pyplot as plt
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter

from smac.acquisition.function.expected_improvement import EICool
from smac.acquisition.function.initial_design_cost_aware_acquisition import (
    InitialDesignCostAwareAcquisition,
)
from smac.acquisition.function.switching_acquistion_function import SwitchingAcquisition
from smac.facade.cost_aware_facade import (
    CostAwareHyperparameterOptimizationFacade,
)
from smac.scenario import Scenario

# Configure logging to see SMAC's output
logging.basicConfig(level=logging.INFO)


def target_function(config: Configuration, seed: int | None = None) -> dict[str, float]:
    """
    A 2D target function where the cost is a complex landscape and the loss is constant.
    """
    x, y = config["x"], config["y"]

    # The cost is the function you provided, normalized to be in [0, 1]
    cost_unnormalized = (
        np.exp(-((x - 2) ** 2 + (y - 2) ** 2))
        + np.exp(-((x + 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x - 2) ** 2 + (y + 2) ** 2))
        - np.exp(-((x + 2) ** 2 + (y - 2) ** 2))
    )
    cost = (cost_unnormalized + 1) / 2

    # The loss is constant, so optimization focuses purely on cost
    loss = 1.0

    return {"loss": loss, "cost": cost}


if __name__ == "__main__":
    # 1. Define the hyperparameter space
    configspace = ConfigurationSpace()
    configspace.add(UniformFloatHyperparameter("x", -3.5, 3.5, default_value=0))
    configspace.add(UniformFloatHyperparameter("y", -3.5, 3.5, default_value=0))

    # 2. Define the scenario
    scenario = Scenario(
        configspace,
        name="test_initial_design_acq_isolated",
        objectives="loss",
        cost_aware=True,
        cost_aware_objective="cost",
        cost_aware_budget=50,  # A small budget for demonstration
        n_trials=150,  # We only run a few trials to see the initial design
        seed=3,
    )

    # 3. Manually create a switching acquisition function that will never switch
    # By setting the factor > 1, the consumed budget will never reach the switch budget.
    initial_acquisition = InitialDesignCostAwareAcquisition(scenario)
    main_acquisition = EICool(scenario=scenario)
    acquisition_function = SwitchingAcquisition(
        scenario=scenario,
        initial_acquisition=initial_acquisition,
        main_acquisition=main_acquisition,
        switch_budget_factor=1.1,  # This ensures we NEVER switch
    )

    # 4. Use the Cost-Aware Facade with our custom acquisition function
    smac = CostAwareHyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=target_function,
        acquisition_function=acquisition_function,
        overwrite=True,
    )

    # 5. Run the optimization
    incumbent = smac.optimize()

    # 6. Plot the results
    # Create a grid for the contour plot
    grid_res = 100
    x_grid = np.linspace(-3.5, 3.5, grid_res)
    y_grid = np.linspace(-3.5, 3.5, grid_res)
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Evaluate cost on the grid
    cost_grid = np.zeros_like(xx)
    for i in range(grid_res):
        for j in range(grid_res):
            res = target_function(Configuration(configspace, {"x": xx[i, j], "y": yy[i, j]}))
            cost_grid[i, j] = res["cost"]

    # Get evaluated points from runhistory
    eval_x, eval_y, eval_costs = [], [], []
    for k, v in smac.runhistory.items():
        config = smac.runhistory.get_config(k.config_id)
        eval_x.append(config["x"])
        eval_y.append(config["y"])
        eval_costs.append(v.time)  # The cost is stored in the `time` field

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("InitialDesignCostAwareAcquisition Test (No Switch)", fontsize=16)

    # Plot Cost Landscape
    contour = ax1.contourf(xx, yy, cost_grid, levels=20, cmap="viridis")
    fig.colorbar(contour, ax=ax1, label="Cost")
    ax1.scatter(eval_x, eval_y, c="red", edgecolor="white", s=50, label="Evaluated Points")
    ax1.set_title("Cost Landscape with Evaluated Points")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True)

    # Plot Cumulative Cost
    cumulative_cost = np.cumsum(eval_costs)
    ax2.plot(range(len(cumulative_cost)), cumulative_cost, "-o", label="Cumulative Cost")
    ax2.axhline(y=scenario.cost_aware_budget, color="r", linestyle="--", label="Total Budget")
    ax2.set_title("Cumulative Cost Over Trials")
    ax2.set_xlabel("Trial Number")
    ax2.set_ylabel("Cumulative Cost")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("initial_design_acq_isolated_test.png")
    plt.show()