#!/bin/bash

# Create the logs directory if it doesn't exist
mkdir -p logs

# Define ranges for the hyperparameters
gamma_values=(1 10 15 20)
k_clf_values=(1 5 10 20)
wd_clf_values=(0.01 0.05 0.1 0.5)

# Iterate over all combinations of hyperparameters
for gamma in "${gamma_values[@]}"; do
  for k_clf in "${k_clf_values[@]}"; do
    for wd_clf in "${wd_clf_values[@]}"; do
      
      # Construct a descriptive log filename
      log_file="logs/gamma_${gamma}_kclf_${k_clf}_wdclf_${wd_clf}.log"
      
      # Run the script with the current combination of hyperparameters
      python wdgrl.py \
        trained_models/source.pt \
        --gamma $gamma \
        --k-clf $k_clf \
        --wd-clf $wd_clf > $log_file 2>&1

      echo "Ran wdgrl.py with gamma=$gamma, k-clf=$k_clf, wd-clf=$wd_clf. Log saved to $log_file."

    done
  done
done

echo "All experiments completed. Logs are stored in the 'logs' directory."
