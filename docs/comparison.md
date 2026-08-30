# Published results vs runs on this machine

Source: arXiv:2407.14251v1, Table 1 (p.7), Table 2 (p.8), Table 3 (p.45).
All published values are validation accuracy, per cent, mean (std). Higher is better.
PM = personalised model, GM = global model.

## Table A. Published, Table 1

### MCLR, strongly convex

| Algorithm | MNIST | Synthetic | FMNIST | EMNIST-10 |
|---|---|---|---|---|
| FedAvg (GM) | 84.87 (0.054) | 84.87 (0.054) | 79.80 (0.002) | 91.60 (0.001) |
| Per-FedAvg (PM) | 94.81 (0.00) | 83.91 (0.15) | 94.75 (0.00) | 97.57 (0.0) |
| pFedMe (GM) | 75.50 (0.00) | 81.93 (0.21) | 83.45 (0.21) | 88.78 (0.01) |
| pFedMe (PM) | 88.89 (0.001) | 87.61 (0.32) | 91.32 (0.08) | 91.23 (0.01) |
| pFedBayes (PM) | 94.13 (0.27) | 87.05 (0.5) | 92.14 (0.001) | 94.13 (0.001) |
| Ditto (GM) | 84.81 (0.001) | 82.35 (0.001) | 74.02 (0.001) | 91.03 (0.0003) |
| h-SGD (GM) | 87.41 (6.35) | 84.29 (5.18) | 81.653 (1.8) | 92.33 (0.001) |
| AL2GD (PM) | 93.70 (0.13) | 84.75 (0.03) | 98.52 (0.004) | 98.72 (0.001) |
| DemLearn (GM) | 87.32 (0.002) | 67.93 (0.04) | 62.60 (0.002) | 69.09 (0.12) |
| DemLearn (PM) | 91.26 (0.01) | 81.21 (0.01) | 97.50 (0.0) | 97.24 (0.005) |
| **PerMFL (GM)** | 86.92 (0.013) | 84.92 (0.06) | 83.71 (0.001) | **91.68 (0.0)** |
| **PerMFL (PM)** | 96.87 (0.0) | 87.94 (0.001) | 96.77 (0.0) | **96.49 (0.0)** |

### DNN (Synthetic) or CNN (image), non-convex

| Algorithm | MNIST | Synthetic | FMNIST | EMNIST-10 |
|---|---|---|---|---|
| FedAvg (GM) | 93.17 (0.02) | 84.53 (0.067) | 84.14 (0.00) | 92.73 (0.003) |
| Per-FedAvg (PM) | 91.845 (0.00) | 75.93 (0.18) | 88.69 (0.269) | 97.37 (0.01) |
| pFedMe (GM) | 80.12 (0.01) | 81.23 (0.19) | 68.64 (0.009) | 91.81 (0.0002) |
| pFedMe (PM) | 97.40 (0.001) | 87.86 (0.06) | 96.30 (0.001) | 97.18 (0.0003) |
| Ditto (GM) | 87.30 (0.03) | 81.12 (0.006) | 57.80 (0.001) | 90.58 (0.004) |
| h-SGD (GM) | 86.59 (7.14) | 87.42 (5.67) | 79.84 (0.035) | 96.03 (0.001) |
| AL2GD (PM) | 91.04 (0.035) | 84.92 (0.02) | 71.32 (0.13) | 92.94 (0.14) |
| DemLearn (GM) | 90.75 (0.001) | 68.91 (0.05) | 64.84 (0.002) | 96.63 (0.005) |
| DemLearn (PM) | 97.20 (0.001) | 82.74 (0.008) | 98.64 (0.0) | 98.74 (0.0) |
| **PerMFL (GM)** | 89.39 (0.001) | 87.53 (0.0) | 79.15 (0.0) | **93.12 (0.0)** |
| **PerMFL (PM)** | 98.15 (0.0) | 87.89 (0.0) | 98.67 (0.0) | **98.79 (0.0)** |

## Table B. Published, Table 2, team formation ablation

400 global iterations, K=10, L=20, lambda=0.5, gamma=1.5, beta=0.6, alpha=0.01.
Worst case: team 1 holds labels {0-4}, team 2 holds {5-9}.
Average case: overlapping, team 1 {0-6}, team 2 {5,6,7,8,9,0,1}.

| Formation | Algorithm | MNIST MCLR | MNIST CNN | FMNIST MCLR | FMNIST CNN | EMNIST-10 MCLR | EMNIST-10 CNN |
|---|---|---|---|---|---|---|---|
| Worst | PerMFL (PM) | 96.86 | 95.80 | 97.14 | 95.62 | 96.57 | 98.13 |
| Worst | PerMFL (GM) | 80.48 | 82.21 | 76.18 | 70.28 | 88.05 | 87.05 |
| Average | PerMFL (PM) | 97.01 | 97.02 | 96.72 | 97.38 | 96.39 | 98.15 |
| Average | PerMFL (GM) | 80.86 | 83.59 | 74.45 | 74.66 | 90.36 | 87.43 |

## Table C. Published, Table 3, MCLR only. Values are fractions, not per cent

| Algorithm | FEMNIST | CIFAR100 |
|---|---|---|
| h-SGD (GM) | 0.6405 (0.005) | 0.1232 (0.001) |
| AL2GD (PM) | 0.4467 (0.01) | 0.65.87 (0.07) [sic, as printed] |
| PerMFL (GM) | 0.5757 (0.0) | 0.1368 (0.0) |
| PerMFL (PM) | 0.8129 (0.0) | 0.6695 (0.001) |

## Table D. The paper's stated setup

| Item | Value |
|---|---|
| Devices | 40, as four teams of ten |
| Participation | full, every team and every device each global round |
| Label skew | at most two classes per device |
| Split | train / validation 3:1 |
| Global rounds T | 400 |
| Team rounds K | 10 |
| Local steps L | 20 |
| lambda, gamma, beta, alpha | 0.5, 1.5, 0.6, 0.01 |
| Strongly convex model | MCLR with softmax |
| Non-convex model | two-hidden-layer DNN (Synthetic), two-layer CNN (image) |
| FEMNIST / CIFAR100 | 3,500 and 350 devices, 3 classes each, 5 teams |

## Table E. What was run on this machine

Counts from `results/`, 297 h5 files.

| Dataset | Runs | Horizons T | Best PM acc | Best GM acc | Best PM F1 | Best GM F1 | In the paper |
|---|---|---|---|---|---|---|---|
| EMNIST-10 | 30 | 5, 100, 400 | 98.49 | 90.45 | 0.9828 | 0.8966 | yes |
| CICIDS2017 | 183 | 2 to 100 | 98.17 | 95.21 | 0.9392 | 0.8622 | no |
| NSL-KDD | 41 | 1, 20 | 81.65 | 74.61 | 0.6212 | 0.4839 | no |
| TON-IoT | 39 | 1, 10 | 86.74 | 66.83 | 0.8266 | 0.5132 | no |
| hierarchical-FedAvg, Synthetic | 1 | 100 | n/a | 85.20 | n/a | n/a | as a baseline only |
| pFedMe, NSL-KDD | 2 | 2, 20 | 78.54 | 74.23 | logged | logged | no |

MNIST, FMNIST, CIFAR-10, CIFAR100, FEMNIST and Synthetic-with-PerMFL have no runs.
Of the seven baseline algorithms the paper compares against, only pFedMe and
hierarchical-FedAvg were ever executed, on datasets the paper does not use.

## Table F. EMNIST-10 runs in detail, all MCLR

Config for every row: 4 teams, 5 devices per team, 20 devices total, full
participation, K=10, L=20, lambda=0.5, gamma=1.5, beta=0.6, alpha=0.01,
two labels per device. **This is half the device count the paper states.**

| exp_no | T | Purpose | PM acc | GM acc |
|---|---|---|---|---|
| 0 | 5 | smoke | 97.92 | 12.06 |
| 1, 4 | 100 | early baseline | 97.89 | 85.29 |
| **2** | **400** | **reproduction, argument defect present** | **96.45** | **87.84** |
| 3 | 400 | same horizon, defect fixed | 98.05 | 90.45 |
| 5, 6, 10-13 | 100 | fixed baseline, seed repeats | 98.37-98.38 | 85.78-85.82 |
| 7, 8, 9 | 100 | L sweep, L = 2, 5, 10 | 95.86, 97.25, 97.96 | 75.87, 81.32, 83.99 |
| 90-93 | 100 | p_teams = 1, 2, 4, 10 | 98.37-98.39 | 85.74-85.85 |
| 700, 702, 704 | 100 | lambda = 20.0 | 96.90-97.42 | 77.20-79.77 |
| 701, 703, 705 | 100 | lambda = 0.5 | 98.35-98.49 | 85.58-86.53 |
| 2100, 2102, 2104 | 100 | PerMFL, lambda_team = lambda | 98.35-98.49 | 85.79-86.53 |
| 2101, 2103, 2105 | 100 | Fine Tuned, lambda_team = 1.5 | 98.35-98.48 | 90.20-90.37 |

## Table G. The two comparisons that matter, kept apart

| | Quantity | Baseline | Comparator | Difference |
|---|---|---|---|---|
| Reproduction | PM accuracy, T=400 | ours 96.45 | published 96.49 | **0.04 pp short** |
| Fine tuning | GM macro F1, T=100 | PerMFL 0.8513 | Fine Tuned 0.8959 | **+4.46 pp** |
| Fine tuning | GM accuracy, T=100 | PerMFL 86.19 | Fine Tuned 90.29 | **+4.10 pp** |
| Fine tuning | PM accuracy, T=100 | PerMFL 98.41 | Fine Tuned 98.41 | -0.01 pp |

The reproduction gap and the fine-tuning gain are not the same size. They differ
by a factor of about 110, and they are different quantities on different tiers.

## Table H. Our global model against the published global model, EMNIST-10 MCLR

| Configuration | GM accuracy | Gap to published 91.68 |
|---|---|---|
| Published PerMFL (GM) | 91.68 | - |
| Ours, PerMFL, T=100 | 86.19 | -5.49 |
| Ours, Fine Tuned, T=100 | 90.29 | **-1.39** |
| Ours, fixed, T=400 (exp 3) | 90.45 | -1.23 |
| Ours, defect present, T=400 (exp 2) | 87.84 | -3.84 |

Fine tuning recovers about three quarters of the shortfall to the published
global-model figure, at a quarter of the published horizon and half the
published device count.
