# Unsupervised domain adaptation

## Overview

This folder contains the replication code to train the data augmentation methods on BDAPPV. DeepCORAL is separated from the other methods. 

The model weights obtained with the code are accessible in the repostory `weights-da` in our Zenodo repository. 

## Retraining

If you want to retrain DeepCORAL, specify the paths to BDAPPV on your machine and run `train.py` in `deepcoral`. If you want to optimize the hyperparameters, run `run_coral.sh`.

If you want to retrain the adversarial DA methods, run `adda.py`, `wdgrl.py` or `revgrad.py` from `adversarial-da`. All these methods rely on a source encoder trained on the source domain. This encoder can be found in the folder `models-da` (`source.pt`) or can be retrained using `train_source.py`. The script `wdgrl_hyperparam.sh` optimizes the hyperparameters of the WDGRL method.

## Sources

The source code contained in the folder `deepcoral` comes from [https://github.com/DenisDsh/PyTorch-Deep-CORAL](https://github.com/DenisDsh/PyTorch-Deep-CORAL) and the source code contained in `adversarial-da` comes from [https://github.com/jvanvugt/pytorch-domain-adaptation](https://github.com/jvanvugt/pytorch-domain-adaptation).