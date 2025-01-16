# -*- coding: utf-8 -*-
#!/usr/bin/env python

import subprocess
import itertools
import os

def run_experiment(batch_size, lr, momentum, lambda_coral, epochs):
    """
    Run the training script with the specified hyperparameters.
    """
    command = [
        "python", "train.py",  # Path to your training script
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--momentum", str(momentum),
        "--lambda_coral", str(lambda_coral),
        "--epochs", str(epochs),
    ]

    # Run the training script
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Save the logs of the current run
    experiment_name = f"batch{batch_size}_lr{lr}_momentum{momentum}_coral{lambda_coral}_epochs{epochs}"
    log_dir = "experiment_logs"
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, f"{experiment_name}.log"), "w") as f:
        f.write(result.stdout)
        f.write("\n\n")
        f.write("=== STDERR ===\n")
        f.write(result.stderr)

    print(f"Experiment {experiment_name} completed. Logs saved in {log_dir}")


def main():
    """
    Run multiple experiments with different hyperparameter combinations.
    """

    # Define hyperparameter ranges
    batch_sizes = [8]
    learning_rates = [1e-3, 1e-4, 5e-4]
    momentums = [0.9, 0.95]
    lambda_coral_weights = [0.1, 0.5, 1.0]
    num_epochs = [10]

    # Iterate over all combinations of hyperparameters
    hyperparam_combinations = itertools.product(batch_sizes, learning_rates, momentums, lambda_coral_weights, num_epochs)

    for batch_size, lr, momentum, lambda_coral, epochs in hyperparam_combinations:
        run_experiment(batch_size, lr, momentum, lambda_coral, epochs)


if __name__ == "__main__":
    main()
