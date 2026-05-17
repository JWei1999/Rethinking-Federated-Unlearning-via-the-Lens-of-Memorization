# Rethinking Federated Unlearning via the Lens of Memorization
[![DOI](https://zenodo.org/badge/1241284109.svg)](https://doi.org/10.5281/zenodo.20258278)        

Official implementation for the paper:

Rethinking Federated Unlearning via the Lens of Memorization (KDD 2026)

## Overview

This repository studies federated unlearning from a memorization perspective.

The core idea is:
1. Identify how strongly target examples are memorized.
2. Use gradient-based tracing to locate memorization/redundant neurons.
3. Apply fine-tuning with selective pruning/re-initialization to forget target-client data.
4. Compare against retraining and standard baselines.

The codebase is notebook-driven and includes reusable Python libraries for datasets, models, federated training, unlearning, and evaluation.

## Highlights

- Federated training with customizable server aggregation and client optimization.
- Memorization score estimation for target datasets.
- Gradient-based tracing for memorization neuron discovery.
- Unlearning via fine-tuning while pruning/re-initializing selected neurons.
- Evaluation against retraining on:
	- Unlearning accuracy
	- Target/test performance
	- Memorization-group behavior

## Repository Structure

- Core notebooks
	- [Federated Unlearning Based on Memorization.ipynb](Federated%20Unlearning%20Based%20on%20Memorization.ipynb): end-to-end training and unlearning experiments.
	- [Memorization Evaluation.ipynb](Memorization%20Evaluation.ipynb): memorization-aware evaluation and figure generation.
- Federated learning and unlearning
	- [Federated_Learning_lib.py](Federated_Learning_lib.py): Server and Client abstractions.
	- [FL_unlearning_fine_tune_by_tracing_gradient_stat.py](FL_unlearning_fine_tune_by_tracing_gradient_stat.py): gradient-stat tracing and unlearning strategies.
	- [FL_unlearning_utility.py](FL_unlearning_utility.py): unlearning/memorization evaluation utilities.
- Deep learning utilities
	- [DL_classification_lib.py](DL_classification_lib.py): training/test loops and metrics.
	- [DL_vision_model.py](DL_vision_model.py): model generators (ResNet, AlexNet, Inception, BN/GN variants).
	- [DL_vision_dataset.py](DL_vision_dataset.py): dataset wrappers (CIFAR-10/100, MNIST, TinyImageNet, etc.).
	- [DL_vision_dataset_sampling.py](DL_vision_dataset_sampling.py): client/data samplers and dataset split/combine helpers.
	- [DL_check_lib.py](DL_check_lib.py): debug and inspection helpers.
	- [General_utils.py](General_utils.py): naming, logging, serialization helpers.
	- [Image_utils.py](Image_utils.py): visualization helpers.
- Data and outputs
	- [dataset](dataset): raw/downloaded dataset root.
	- [data](data): extra data artifacts.
	- [model](model): saved model checkpoints.
	- [output](output): generated outputs.
	- [log](log): experiment logs.

## Environment Setup

Recommended:
- Python 3.9+
- CUDA-enabled PyTorch for GPU experiments

Install dependencies:

```bash
pip install torch torchvision
pip install numpy 
pip install jupyter notebook
```

## Quick Start

### 1) Run Main Federated Unlearning Experiments

Open and run [Federated Unlearning Based on Memorization.ipynb](Federated%20Unlearning%20Based%20on%20Memorization.ipynb) from top to bottom.

Typical workflow inside the notebook:
1. Load libraries and set device.
2. Build datasets (CIFAR-10/CIFAR-100/EMNIST options).
3. Create client splits (IID or Non-IID).
4. Train baseline federated model.
5. Train retraining baselines (excluding target client).
6. Run proposed unlearning method (fine-tuning + tracing/pruning).
7. Save checkpoints and intermediate artifacts.

### 2) Run Memorization-Centric Evaluation

Open and run [Memorization Evaluation.ipynb](Memorization%20Evaluation.ipynb).

This notebook loads produced checkpoints/artifacts and reports:
- Memorization-group unlearning behavior
- Retraining vs proposed method comparisons
- Plots/tables for IID and Non-IID settings

## Data Preparation

The code uses dataset wrappers under [DL_vision_dataset.py](DL_vision_dataset.py) with root path typically set in notebooks as:

- dataset_dir = ./dataset

For CIFAR datasets, torchvision download mechanisms are used by wrappers when enabled by the dataset class configuration.

## Main Method Entry Points

Important functions and classes:

- Federated training abstraction:
	- Server, Client in [Federated_Learning_lib.py](Federated_Learning_lib.py)
- Unlearning training loops:
	- fedAvg_epoch_fine_tuning_while_pruning_redundant_neurons
	- in [FL_unlearning_fine_tune_by_tracing_gradient_stat.py](FL_unlearning_fine_tune_by_tracing_gradient_stat.py)
- Memorization and unlearning evaluation:
	- measure_memorization_scores
	- RetrainingMeasurement
	- MemorizationMeasurement
	- in [FL_unlearning_utility.py](FL_unlearning_utility.py)

## Reproducibility Notes

- Keep client splits saved/loaded consistently across runs.
- Keep model naming/path rules consistent (helpers in [General_utils.py](General_utils.py)).
- Ensure the same target client index is used in training and evaluation.

## License

This project is released under the license in [LICENSE](LICENSE).

## Citation

If you use this repository, please cite the paper.
BibTeX will be added once the publication metadata is finalized.

