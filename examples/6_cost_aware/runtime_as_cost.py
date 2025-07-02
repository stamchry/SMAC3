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


def my_target_function(config: Configuration, seed: int | None = None) -> float:
    """
    A simple target function to be minimized. It returns only the loss.
    SMAC will automatically measure the execution time as the cost.

    The performance (loss) is a simple quadratic function with a minimum at x=5.
    The cost (runtime) increases quadratically with x, making higher values
    of x more expensive to evaluate.
    """
    x = config["x"]

    # The cost is a function of the hyperparameter x
    cost = 0.5 + (x / 10)
    loss = (x - 5) ** 2

    # Simulate the evaluation time
    time.sleep(cost)

    # In a real scenario, you only need to return the loss.
    return loss


if __name__ == "__main__":
    # 1. Define the hyperparameter space
    configspace = ConfigurationSpace()
    configspace.add(UniformFloatHyperparameter("x", 0, 6, default_value=5))

    # 2. Define the scenario
    scenario = Scenario(
        configspace,
        name="new_code3",
        objectives="loss",  # Name of the objective
        # --- Key parameters for cost-aware optimization ---
        cost_aware=True,
        walltime_limit=90,  # Total budget in seconds
        # ---
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
    all_runtimes = []
    all_losses = []
    all_xs = []
    for k, v in smac.runhistory.items():
        if v.time > 0:  # Ensure the trial was actually run
            config = smac.runhistory.get_config(k.config_id)
            all_runtimes.append(v.time)
            all_losses.append(v.cost)
            all_xs.append(config["x"])

    # Convert to numpy arrays for plotting
    all_runtimes = np.array(all_runtimes)
    all_losses = np.array(all_losses)
    all_xs = np.array(all_xs)
    trial_indices = np.arange(len(all_runtimes))

    # Find the point where the switch happened
    switch_budget = scenario.walltime_limit * 0.125  # Default fraction
    cumulative_runtimes = np.cumsum(all_runtimes)
    # Handle cases where the run is too short to switch
    switch_indices = np.where(cumulative_runtimes >= switch_budget)[0]
    switch_index = switch_indices[0] if len(switch_indices) > 0 else -1

    # Plot 1: Loss vs. Runtime (Trade-off)
    plt.figure(figsize=(10, 6))
    plt.scatter(all_runtimes, all_losses, alpha=0.5, label="All Evaluated Trials")
    plt.scatter(
        [average_runtime],
        [incumbent_loss],
        color="red",
        marker="*",
        s=200,
        zorder=10,
        label="Final Incumbent",
    )
    plt.title("Performance vs. Runtime Trade-off")
    plt.xlabel("Measured Runtime (s)")
    plt.ylabel("Loss (Objective Value)")
    plt.legend()
    plt.grid(True)
    plt.savefig("tradeoff_plot.png")

    # Plot 2: Loss vs. Trial Number (Convergence)
    plt.figure(figsize=(10, 6))
    plt.plot(trial_indices, all_losses, "-o", alpha=0.5)
    if switch_index != -1:
        plt.axvline(
            x=switch_index,
            color="r",
            linestyle="--",
            label=f"Switch to EICool (Trial {switch_index})",
        )
    plt.title("Convergence Over Time")
    plt.xlabel("Trial Number")
    plt.ylabel("Loss (Objective Value)")
    plt.legend()
    plt.grid(True)
    plt.savefig("convergence_plot.png")

    # Plot 3: Runtime vs. Trial Number (Cost Strategy)
    plt.figure(figsize=(10, 6))
    plt.plot(trial_indices, all_runtimes, "-o", alpha=0.5)
    if switch_index != -1:
        plt.axvline(
            x=switch_index,
            color="r",
            linestyle="--",
            label=f"Switch to EICool (Trial {switch_index})",
        )
    plt.title("Runtime Cost Strategy Over Time")
    plt.xlabel("Trial Number")
    plt.ylabel("Measured Runtime (s)")
    plt.legend()
    plt.grid(True)
    plt.savefig("runtime_strategy_plot.png")

    # Plot 4: Loss vs. x with the real loss function
    x_vals = np.linspace(0, 10, 100)  # Range of x values
    loss_vals = (x_vals - 5) ** 2  # Calculate loss for each x

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, loss_vals, label="True Loss Function", color="blue")
    plt.scatter(all_xs, all_losses, color="red", label="Evaluated Configurations", alpha=0.5)
    plt.scatter(
        [incumbent["x"]],
        [incumbent_loss],
        color="green",
        marker="*",
        s=200,
        zorder=10,
        label="Final Incumbent",
    )
    plt.xlabel("x")
    plt.ylabel("Loss")
    plt.title("Loss vs. x with Evaluated Configurations")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_vs_x_plot.png")

    plt.show()