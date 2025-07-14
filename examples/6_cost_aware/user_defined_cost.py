from __future__ import annotations

import logging
import time

import numpy as np
import matplotlib.pyplot as plt
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter

from smac.facade.cost_aware_facade import (
    CostAwareHyperparameterOptimizationFacade,
)
from smac.scenario import Scenario

# Configure logging to see SMAC's output, including the acquisition function switch
logging.basicConfig(level=logging.INFO)


def my_target_function(config: Configuration, seed: int | None = None) -> dict[str, float]:
    """
    A simple target function to be minimized. It returns a dictionary
    containing the loss and the cost.
    """
    x = config["x"]

    # The cost is a function of the hyperparameter x
    cost = 0.5 + (x / 10)
    #cost = (3-x)**2
    loss = (x - 5) ** 2

    # Simulate the evaluation time
    time.sleep(0.1)  # We sleep for a short time to not slow down the example

    return {"loss": loss, "cost": cost}


if __name__ == "__main__":
    # 1. Define the hyperparameter space
    configspace = ConfigurationSpace()
    configspace.add(UniformFloatHyperparameter("x", 0, 6, default_value=5))

    # 2. Define the scenario
    scenario = Scenario(
        configspace,
        name="new_code4",
        objectives="loss",  # Name of the objective
        # --- Key parameters for cost-aware optimization ---
        cost_aware=True,
        cost_aware_objective="cost",  # Name of the cost objective
        cost_aware_budget=300,  # Total budget in seconds
        # ---
        n_trials=np.inf,
        seed=1,
    )

    # 3. Select the appropriate facade
    smac = CostAwareHyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=my_target_function,
        overwrite=True
    )

    # 4. Run the optimization
    incumbent = smac.optimize()

    # 5. Print the results
    print("\n--- Results ---")
    print(f"Incumbent configuration: {incumbent}")

    # In SMAC, "cost" is the objective value, which we named "loss".
    incumbent_loss = smac.runhistory.get_cost(incumbent)
    print(f"Incumbent loss: {incumbent_loss:.2f}")

    # To get the runtime, we must iterate through the runhistory and find all
    # trials (TrialValue objects) for the incumbent configuration.
    incumbent_id = smac.runhistory.get_config_id(incumbent)
    incumbent_trial_values = [
        v for k, v in smac.runhistory.items() if k.config_id == incumbent_id
    ]
    incumbent_runtimes = [v.time for v in incumbent_trial_values]
    average_runtime = np.mean(incumbent_runtimes)
    print(f"Incumbent's average measured runtime: {average_runtime:.2f}s")
    print("----------------")

    # 6. Plot the results
    # We iterate through the runhistory to get paired data for each trial
    all_costs = []
    all_losses = []
    all_xs = []
    for k, v in smac.runhistory.items():
        if v.time > 0:  # Ensure the trial was actually run
            config = smac.runhistory.get_config(k.config_id)
            all_costs.append(v.time)
            all_losses.append(v.cost)
            all_xs.append(config["x"])

    # Convert to numpy arrays for plotting
    all_costs = np.array(all_costs)
    all_losses = np.array(all_losses)
    all_xs = np.array(all_xs)
    trial_indices = np.arange(len(all_costs))

    # Find the point where the switch happened
    switch_budget_factor = 1 / 8  # Default from SwitchingAcquisition
    switch_budget = scenario.cost_aware_budget * switch_budget_factor
    cumulative_costs = np.cumsum(all_costs)
    # Handle cases where the run is too short to switch
    switch_indices = np.where(cumulative_costs >= switch_budget)[0]
    switch_index = switch_indices[0] if len(switch_indices) > 0 else -1

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("SMAC Cost-Aware Optimization Analysis", fontsize=16)

    # Plot 1: Loss vs. Cost (Trade-off)
    ax = axs[0, 0]
    ax.scatter(all_costs, all_losses, alpha=0.5, label="Evaluated Trials")
    ax.scatter(
        [average_runtime],
        [incumbent_loss],
        color="red",
        marker="*",
        s=200,
        zorder=10,
        label="Final Incumbent",
    )
    ax.set_title("Performance vs. Cost Trade-off")
    ax.set_xlabel("Cost")
    ax.set_ylabel("Loss (Objective Value)")
    ax.legend()
    ax.grid(True)

    # Plot 2: Loss vs. Trial Number (Convergence)
    ax = axs[0, 1]
    ax.plot(trial_indices, all_losses, "-o", alpha=0.5, label="Loss per Trial")
    if switch_index != -1:
        ax.axvline(
            x=switch_index,
            color="r",
            linestyle="--",
            label=f"Switch to EICool (Trial {switch_index})",
        )
    ax.set_title("Convergence Over Time")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Loss (Objective Value)")
    ax.legend()
    ax.grid(True)

    # Plot 3: Cost vs. Trial Number (Cost Strategy)
    ax = axs[1, 0]
    ax.plot(trial_indices, all_costs, "-o", alpha=0.5, label="Cost per Trial")
    if switch_index != -1:
        ax.axvline(
            x=switch_index,
            color="r",
            linestyle="--",
            label=f"Switch to EICool (Trial {switch_index})",
        )
    ax.set_title("Cost Strategy Over Time")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Cost")
    ax.legend()
    ax.grid(True)

    # Plot 4: Loss vs. x with the real loss function
    ax = axs[1, 1]
    x_vals = np.linspace(0, 6, 100)  # Range of x values
    loss_vals = (x_vals - 5) ** 2  # Calculate loss for each x
    ax.plot(x_vals, loss_vals, label="True Loss Function", color="blue")
    ax.scatter(all_xs, all_losses, color="red", label="Evaluated Configurations", alpha=0.5)
    ax.scatter(
        [incumbent["x"]],
        [incumbent_loss],
        color="green",
        marker="*",
        s=200,
        zorder=10,
        label="Final Incumbent",
    )
    ax.set_xlabel("Hyperparameter 'x'")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs. Hyperparameter 'x'")
    ax.legend()
    ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("detailed_analysis_plots.png")

    plt.show()