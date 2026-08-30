# Results

Every measurement in `results/`, 297 h5 files. Published comparisons are in
[comparison.md](comparison.md).

Metric directions: macro F1 higher is better, accuracy higher is better, macro
recall higher is better, macro FPR lower is better. PM is the personalised model,
GM is the global model. All values are the peak over the run's global rounds.

Runs are labelled `shipped` when `--lamda_team` is unset, so equation 3 weights
the member average by `--lamda`, and `decoupled` when `--lamda_team 1.5` is
passed. Neither is recorded in the h5, so pairing is by exp_no parity: even is
shipped, odd is decoupled.

## 1. Reproduction

| | Ours | Published | Gap |
|---|---|---|---|
| EMNIST-10, MCLR, T=400, PM accuracy | 96.45 | 96.49 | 0.04 pp |

The run carries the argument-binding defect, which is present in the released
code, and still lands on the published figure. Configuration was 20 devices in
four teams; the paper states 40 devices in four teams, so this is not a
like-for-like reproduction.

## 2. The team tier is inert under the shipped coupling

Changing the number of teams changes almost nothing.

| Dataset | Teams compared | Change in PM macro F1 |
|---|---|---|
| CICIDS2017 | 1 against 10 | 0.0002 |
| EMNIST-10 | 1 against 4 | 0.0000 |
| NSL-KDD | 1 against 10 | 0.0011 |

Seed-to-seed noise on the same configuration is 0.0116, so all three are below
the noise floor. Team assignment method, whether oracle, random or derived, was
null across five comparisons.

Mechanism: at lambda 0.05, gamma 1.5, eta 0.03 the team update weights the
member average at 0.0015 against a 0.045 pull toward the global model, so member
influence is 3.3 per cent. The global model advances 0.135 per cent per round.

## 3. Decoupling, headline

CICIDS2017, domainmix, benign fraction 0.25, cross-test on, five seeds, exp 906-915.

| Metric | Shipped | Decoupled | Delta |
|---|---|---|---|
| PM macro F1 | 0.2809 | 0.3154 | +0.0345 |
| PM accuracy | 0.7842 | 0.8161 | +0.0320 |
| GM macro F1 | 0.1139 | 0.2155 | +0.1016 |
| GM accuracy | 0.5412 | 0.8487 | +0.3075 |

## 4. Decoupling across the heterogeneity sweep

Twenty-seven pairs. Ordered within each dataset from most to least heterogeneous.

### CICIDS2017, exp 2000-2023, twelve pairs

| Pair | PM shipped | PM decoupled | dPM | GM shipped | GM decoupled | dGM |
|---|---|---|---|---|---|---|
| 2000/2001 | 0.2391 | 0.2764 | +0.0373 | 0.0965 | 0.2727 | +0.1762 |
| 2002/2003 | 0.2479 | 0.2712 | +0.0233 | 0.1063 | 0.2357 | +0.1294 |
| 2004/2005 | 0.2474 | 0.2688 | +0.0214 | 0.1426 | 0.1998 | +0.0572 |
| 2006/2007 | 0.4489 | 0.5720 | +0.1231 | 0.1178 | 0.5328 | +0.4150 |
| 2008/2009 | 0.4611 | 0.5742 | +0.1131 | 0.1154 | 0.5023 | +0.3868 |
| 2010/2011 | 0.4579 | 0.5697 | +0.1117 | 0.1286 | 0.5187 | +0.3901 |
| 2012/2013 | 0.4652 | 0.7063 | +0.2411 | 0.1221 | 0.6528 | +0.5306 |
| 2014/2015 | 0.4823 | 0.7137 | +0.2314 | 0.1396 | 0.6444 | +0.5048 |
| 2016/2017 | 0.4901 | 0.7093 | +0.2193 | 0.1409 | 0.6444 | +0.5035 |
| 2018/2019 | 0.5282 | 0.8464 | +0.3182 | 0.1558 | 0.8597 | +0.7039 |
| 2020/2021 | 0.5203 | 0.8473 | +0.3270 | 0.1511 | 0.8622 | +0.7111 |
| 2022/2023 | 0.5042 | 0.8358 | +0.3316 | 0.1421 | 0.8419 | +0.6997 |
| **mean** | | | **+0.1749** | | | **+0.4340** |

Wins 12/12 on both tiers.

### TON-IoT, exp 2024-2041, nine pairs

| Pair | dPM | dGM |
|---|---|---|
| 2024/2025 | +0.0374 | +0.1168 |
| 2026/2027 | +0.0515 | +0.1779 |
| 2028/2029 | +0.0285 | +0.1771 |
| 2030/2031 | +0.1213 | +0.3068 |
| 2032/2033 | +0.1069 | +0.3326 |
| 2034/2035 | +0.0999 | +0.2248 |
| 2036/2037 | +0.1455 | +0.3488 |
| 2038/2039 | +0.1718 | +0.3949 |
| 2040/2041 | +0.1668 | +0.3855 |
| **mean** | **+0.1033** | **+0.2739** |

Wins 9/9 on both tiers.

### NSL-KDD, exp 2042-2053, six pairs

| Pair | dPM | dGM |
|---|---|---|
| 2042/2043 | +0.1415 | +0.1204 |
| 2044/2045 | +0.1163 | +0.1418 |
| 2046/2047 | +0.1517 | +0.2155 |
| 2048/2049 | +0.1217 | +0.1012 |
| 2050/2051 | +0.1041 | +0.0863 |
| 2052/2053 | +0.1284 | +0.1511 |
| **mean** | **+0.1273** | **+0.1360** |

Wins 6/6 on both tiers.

The benefit grows as heterogeneity falls. That is the opposite of what was
predicted before the sweep ran.

## 5. The one regression

TON-IoT under the `domain` partition, exp 916-925, five pairs. This is the most
heterogeneous configuration measured, mean pairwise Jensen-Shannon divergence
0.6879.

| Pair | GM shipped | GM decoupled | dGM |
|---|---|---|---|
| 916/917 | 0.1068 | 0.0483 | -0.0585 |
| 918/919 | 0.0822 | 0.0383 | -0.0439 |
| 920/921 | 0.1188 | 0.0383 | -0.0805 |
| 922/923 | 0.0767 | 0.0383 | -0.0384 |
| 924/925 | 0.0740 | 0.0383 | -0.0357 |
| **mean** | **0.0917** | **0.0403** | **-0.0514** |

Zero wins out of five. Decoupled GM accuracy is 0.2369 in four of the five,
which is exactly the majority-class floor, so the global model has collapsed to
predicting `normal`. The personalised model still improved in all five pairs,
0.1561 to 0.1701 and similar.

## 6. EMNIST-10 decoupling, exp 2100-2105

MCLR, T=100, 20 devices in four teams, three seeds paired.

| Metric | Direction | Shipped | Decoupled | Delta | t | Wins |
|---|---|---|---|---|---|---|
| PM macro F1 | up | 0.9818 +- 0.0008 | 0.9818 +- 0.0007 | -0.0000 | -0.54 | 1/3 |
| PM accuracy | up | 0.9841 +- 0.0007 | 0.9841 +- 0.0007 | -0.0001 | -1.73 | 0/3 |
| PM recall | up | 0.9811 +- 0.0008 | 0.9811 +- 0.0008 | +0.0001 | 3.22 | 3/3 |
| PM FPR | down | 0.0018 +- 0.0001 | 0.0018 +- 0.0001 | +0.0000 | 0.95 | 1/3 |
| GM macro F1 | up | 0.8513 +- 0.0041 | 0.8959 +- 0.0006 | +0.0446 | 20.12 | 3/3 |
| GM accuracy | up | 0.8619 +- 0.0037 | 0.9029 +- 0.0008 | +0.0410 | 22.14 | 3/3 |
| GM recall | up | 0.8578 +- 0.0039 | 0.9014 +- 0.0005 | +0.0435 | 19.76 | 3/3 |
| GM FPR | down | 0.0156 +- 0.0004 | 0.0110 +- 0.0001 | -0.0046 | -23.59 | 3/3 |

The personalised model does not move. The global model improves on all four
metrics on all three seeds. The PM recall row has t = 3.22 but a delta of
0.0001; it is not an effect.

This is the mechanism the code predicts. `lamda_team` appears only in equation 3,
the team update, and reaches the global model through equations 4 and 5. It never
enters equation 1, which governs the device parameters.

## 7. Variance collapse

The decoupled global model is far more stable, and this holds even where the mean
regressed.

| Dataset | GM accuracy sd, shipped | GM accuracy sd, decoupled |
|---|---|---|
| CICIDS2017 | 0.1885 | 0.0041 |
| TON-IoT, victim-IP | 0.0928 | 0.0004 |
| NSL-KDD | 0.0619 | 0.0063 |
| EMNIST-10 | 0.0037 | 0.0008 |

## 8. Sweeps

| Sweep | Finding |
|---|---|
| Local steps L, EMNIST-10, L in {2,5,10,20} | monotone, L=20 best. PM accuracy 95.86, 97.25, 97.96, 98.38 |
| Teams p_teams, EMNIST-10, {1,2,4,10} | null. PM macro F1 0.9815, 0.9815, 0.9815, 0.9816 |
| Horizon T, EMNIST-10, {5,100,400} | PM accuracy 97.92, 98.38, 98.05 |
| lambda value, EMNIST-10, 20.0 against 0.5 | PM macro F1 0.9670 +- 0.0031 against 0.9819 +- 0.0008 |
| Percentile clipping, CICIDS2017 | +0.028 macro F1 |
| `OMP_NUM_THREADS=1` | 1.28x wall-clock, not the 2 to 4x predicted |

## 9. Floors and ceilings

| Quantity | Value |
|---|---|
| CICIDS2017 accuracy floor, majority class | 0.8171 |
| CICIDS2017 macro F1 floor | 0.0999 |
| CICIDS2017 centralised ceiling, this config | 0.8994 |
| NSL-KDD normal share | 0.535 |
| TON-IoT normal share | 0.237 |
| Structural macro F1 ceiling under cross-test, 4 of 20 clients per class | 0.4074 |
| Achieved against that ceiling | 0.3435, or 84 per cent of it |

The structural ceiling is the mean over classes of `2(h/N) / (1 + h/N)` where `h`
is the number of clients holding the class. It is why 0.6 macro F1 was
unreachable in the cross-test configuration.

## 10. Local-only baseline

A client trained only on its own data beats the federated personalised model on
every IDS dataset, including under cross-test. It has no global model, so it
cannot detect anything absent from its own training set and cannot be deployed
as a shared detector.

## 11. Coverage gaps

| Missing | Detail |
|---|---|
| Published datasets not run | MNIST, FMNIST, CIFAR-10, CIFAR100, FEMNIST, Synthetic with PerMFL |
| Baselines not run | FedAvg, Ditto, PerAvg, AL2GD, pFedBayes, Hier-Local-QSGD on any dataset |
| Baselines partially run | pFedMe on NSL-KDD only, two runs. hierarchical-FedAvg on Synthetic, one run |
| Configuration mismatch | every EMNIST run used 20 devices; the paper states 40 |
| Unattributed | exp 900-905 on NSL-KDD do not follow the parity convention and are not labelled |
| Single-run results | the L sweep and the communication sweep have no seed replication |
