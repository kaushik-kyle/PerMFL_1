# Results

Every measurement in `results/`, 297 h5 files. Published comparisons are in
[comparison.md](comparison.md).

## 0. Conventions

**The two arms.**

| Name | Meaning |
|---|---|
| **PerMFL** | the method as published. `--lamda_team` is unset, so the team update weights the member average by `--lamda`, the same parameter that governs the device update |
| **Fine Tuned** | `--lamda_team 1.5`. The team update's weight on the member average is set independently of the device proximal coefficient |

Neither arm is recorded in the h5 file, so pairing is by exp_no parity: even is
PerMFL, odd is Fine Tuned. See defect 21 in [defects.md](defects.md).

**The two tiers**, following the paper's own notation.

| Symbol | Meaning |
|---|---|
| PM | personalised model, the device parameters, one per client |
| GM | global model, the top-tier parameters, one shared |

**The four metrics.** Every table states which one it reports.

| Metric | Direction | What it measures |
|---|---|---|
| macro F1 | higher is better | unweighted mean of per-class F1. Ignores the true-negative cell, which is why it survives an 81.7 per cent benign corpus |
| accuracy | higher is better | fraction correct. Dominated by the true-negative cell on imbalanced data |
| macro recall | higher is better | unweighted mean per-class detection rate |
| macro FPR | lower is better | unweighted mean per-class false-positive rate, the alert-fatigue proxy |

**Aggregation.** Every figure is the **peak over a run's global rounds**. The
final-round value is never used. Paired t-statistics use ddof=1 over the
per-seed differences. Using final-round values instead would move the headline
t from 21.25 to 11.35, so the choice matters and is stated here once.

**Verification.** Every value below was recomputed from the h5 files on
30 August 2026 and matches the report chapters. Nine t-statistics in Chapter 7
drifted in the last significant figure against a fresh recomputation and were
corrected; the largest was the CICIDS k=8 global row, 213.4 against 212.62.
Every effect size was already exact and no conclusion changed.

## 1. Reproduction

| Dataset, model, horizon | Metric | Ours | Published | Gap |
|---|---|---|---|---|
| EMNIST-10, MCLR, T=400 | PM accuracy, higher better | 96.45 | 96.49 | 0.04 pp |

The run carries the argument-binding defect, which is present in the released
code, and still lands on the published figure. Configuration was 20 devices in
four teams; the paper states 40 devices in four teams, so this is not a
like-for-like reproduction.

## 2. The team tier is inert under the published coupling

Changing the number of teams changes almost nothing. Metric is **PM macro F1**,
higher is better.

| Dataset | Teams compared | Change |
|---|---|---|
| CICIDS2017 | 1 against 10 | 0.0002 |
| EMNIST-10 | 1 against 4 | 0.0000 |
| NSL-KDD | 1 against 4 | 0.0011 |

Seed-to-seed noise on the same configuration is 0.0116 in the same metric, so
all three are below the noise floor. Team assignment method, whether oracle,
random or derived, was null across five comparisons.

Mechanism: at lambda 0.05, gamma 1.5, eta 0.03 the team update weights the
member average at 0.0015 against a 0.045 pull toward the global model, so member
influence is 3.3 per cent. The global model advances 0.135 per cent per round.

## 3. Headline comparison

CICIDS2017, domainmix partition, benign fraction 0.25, cross-test on, five seeds,
exp 906-915. Critical value at four degrees of freedom is 2.776.

| Metric | Direction | PerMFL | Fine Tuned | Delta | t |
|---|---|---|---|---|---|
| PM macro F1 | higher better | 0.2809 | 0.3154 | +0.0346 | 21.25 |
| PM accuracy | higher better | 0.7842 | 0.8161 | +0.0320 | 10.61 |
| GM macro F1 | higher better | 0.1139 | 0.2155 | +0.1016 | 4.35 |
| GM accuracy | higher better | 0.5412 | 0.8487 | +0.3075 | 3.60 |

All four clear the critical value and win on every seed. The floors are 0.8171
accuracy and 0.0999 macro F1. The PerMFL global model sits below the accuracy
floor; the Fine Tuned one sits above it.

## 4. Heterogeneity sweep

Twenty-seven pairs, three seeds per point. **All values are macro F1, higher is
better.** Ordered within each dataset from most to least heterogeneous.

### CICIDS2017, exp 2000-2023, twelve pairs

| Pair | PerMFL (PM) | Fine Tuned (PM) | Delta PM | PerMFL (GM) | Fine Tuned (GM) | Delta GM |
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
| **mean** | 0.4244 | 0.5993 | **+0.1749** | 0.1299 | 0.5639 | **+0.4340** |

Fine Tuned wins 12/12 on both tiers.

### TON-IoT, exp 2024-2041, nine pairs

| Pair | PerMFL (PM) | Fine Tuned (PM) | Delta PM | PerMFL (GM) | Fine Tuned (GM) | Delta GM |
|---|---|---|---|---|---|---|
| 2024/2025 | 0.1471 | 0.1845 | +0.0374 | 0.1111 | 0.2279 | +0.1168 |
| 2026/2027 | 0.1757 | 0.2272 | +0.0515 | 0.0318 | 0.2096 | +0.1779 |
| 2028/2029 | 0.1772 | 0.2057 | +0.0285 | 0.0851 | 0.2622 | +0.1771 |
| 2030/2031 | 0.2093 | 0.3305 | +0.1213 | 0.1063 | 0.4131 | +0.3068 |
| 2032/2033 | 0.2106 | 0.3175 | +0.1069 | 0.0511 | 0.3837 | +0.3326 |
| 2034/2035 | 0.2115 | 0.3114 | +0.0999 | 0.1001 | 0.3249 | +0.2248 |
| 2036/2037 | 0.2757 | 0.4212 | +0.1455 | 0.1063 | 0.4551 | +0.3488 |
| 2038/2039 | 0.2724 | 0.4442 | +0.1718 | 0.0678 | 0.4627 | +0.3949 |
| 2040/2041 | 0.2588 | 0.4256 | +0.1668 | 0.1277 | 0.5132 | +0.3855 |
| **mean** | 0.2154 | 0.3187 | **+0.1033** | 0.0875 | 0.3614 | **+0.2739** |

Fine Tuned wins 9/9 on both tiers.

### NSL-KDD, exp 2042-2053, six pairs

| Pair | PerMFL (PM) | Fine Tuned (PM) | Delta PM | PerMFL (GM) | Fine Tuned (GM) | Delta GM |
|---|---|---|---|---|---|---|
| 2042/2043 | 0.3325 | 0.4739 | +0.1415 | 0.3635 | 0.4839 | +0.1204 |
| 2044/2045 | 0.3316 | 0.4479 | +0.1163 | 0.3102 | 0.4519 | +0.1418 |
| 2046/2047 | 0.3172 | 0.4689 | +0.1517 | 0.2538 | 0.4693 | +0.2155 |
| 2048/2049 | 0.3428 | 0.4645 | +0.1217 | 0.3619 | 0.4631 | +0.1012 |
| 2050/2051 | 0.3415 | 0.4456 | +0.1041 | 0.3573 | 0.4435 | +0.0863 |
| 2052/2053 | 0.3356 | 0.4640 | +0.1284 | 0.3134 | 0.4645 | +0.1511 |
| **mean** | 0.3335 | 0.4608 | **+0.1273** | 0.3267 | 0.4627 | **+0.1360** |

Fine Tuned wins 6/6 on both tiers.

### Sweep summary with significance

JSD is the mean pairwise Jensen-Shannon divergence between client label
distributions, higher meaning more heterogeneous. Ceiling is the structural
macro F1 bound imposed by the evaluation protocol. All deltas are macro F1.

| Configuration | JSD | Ceiling | Delta PM | t | Delta GM | t |
|---|---|---|---|---|---|---|
| CICIDS k=1 | 0.1730 | 0.308 | +0.0273 | 5.45 | +0.1209 | 3.49 |
| CICIDS k=3 | 0.1414 | 0.596 | +0.1160 | 32.27 | +0.3973 | 44.69 |
| CICIDS k=5 | 0.0851 | 0.795 | +0.2306 | 36.52 | +0.5130 | 58.02 |
| CICIDS k=8 | 0.0001 | 1.000 | +0.3256 | 82.92 | +0.7049 | 212.62 |
| TON-IoT alpha=0.1 | 0.7591 | 0.781 | +0.0391 | 5.85 | +0.1573 | 7.77 |
| TON-IoT alpha=0.5 | 0.3909 | 0.985 | +0.1094 | 17.43 | +0.2881 | 8.87 |
| TON-IoT alpha=5.0 | 0.0646 | 1.000 | +0.1614 | 19.98 | +0.3764 | 26.75 |
| NSL-KDD alpha=0.1 | 0.5379 | 1.000 | +0.1365 | 12.95 | +0.1592 | 5.53 |
| NSL-KDD alpha=5.0 | 0.0333 | 1.000 | +0.1181 | 16.28 | +0.1129 | 5.76 |

Critical value at two degrees of freedom is 4.303. The benefit grows as
heterogeneity falls, which is the opposite of what was predicted before the
sweep ran.

## 5. The one regression

TON-IoT under the `domain` partition, exp 916-925, five pairs. This is the most
heterogeneous configuration measured, JSD 0.6879. Metric is **macro F1**, higher
is better.

| Pair | PerMFL (GM) | Fine Tuned (GM) | Delta GM |
|---|---|---|---|
| 916/917 | 0.1068 | 0.0483 | -0.0585 |
| 918/919 | 0.0822 | 0.0383 | -0.0439 |
| 920/921 | 0.1188 | 0.0383 | -0.0805 |
| 922/923 | 0.0767 | 0.0383 | -0.0384 |
| 924/925 | 0.0740 | 0.0383 | -0.0357 |
| **mean** | 0.0917 | 0.0403 | **-0.0514**, t = -6.21 |

Fine Tuned wins 0/5. Its GM accuracy is 0.2369 in four of the five, which is
exactly the majority-class floor, so the global model has collapsed to
predicting `normal`. The personalised model still improved in all five pairs,
PM macro F1 0.1561 to 0.1701 and similar.

## 6. EMNIST-10, the paper's own dataset

MCLR, T=100, 20 devices in four teams, three seeds paired, exp 2100-2105.
Critical value at two degrees of freedom is 4.303.

| Metric | Direction | PerMFL | Fine Tuned | Delta | t | Wins |
|---|---|---|---|---|---|---|
| PM macro F1 | higher better | 0.9818 +- 0.0008 | 0.9818 +- 0.0007 | -0.0000 | -0.54 | 1/3 |
| PM accuracy | higher better | 0.9841 +- 0.0007 | 0.9841 +- 0.0007 | -0.0001 | -1.73 | 0/3 |
| PM macro recall | higher better | 0.9811 +- 0.0008 | 0.9811 +- 0.0008 | +0.0001 | 3.22 | 3/3 |
| PM macro FPR | lower better | 0.0018 +- 0.0001 | 0.0018 +- 0.0001 | +0.0000 | 0.95 | 1/3 |
| GM macro F1 | higher better | 0.8513 +- 0.0041 | 0.8959 +- 0.0006 | +0.0446 | 20.12 | 3/3 |
| GM accuracy | higher better | 0.8619 +- 0.0037 | 0.9029 +- 0.0008 | +0.0410 | 22.14 | 3/3 |
| GM macro recall | higher better | 0.8578 +- 0.0039 | 0.9014 +- 0.0005 | +0.0435 | 19.76 | 3/3 |
| GM macro FPR | lower better | 0.0156 +- 0.0004 | 0.0110 +- 0.0001 | -0.0046 | -23.59 | 3/3 |

The personalised model does not move on any metric. The global model improves on
all four, on all three seeds. The PM macro recall row has t = 3.22 but a delta of
0.0001, below the critical value; it is not an effect.

This is the mechanism the code predicts. `lamda_team` appears only in the team
update and reaches the global model through the two aggregation steps. It never
enters the device update.

### Against the published figures

| Configuration | GM accuracy | Gap to published 91.68 |
|---|---|---|
| Published PerMFL (GM), MCLR | 91.68 | - |
| PerMFL, ours, T=100 | 86.19 | -5.49 |
| Fine Tuned, ours, T=100 | 90.29 | **-1.39** |

## 6b. EMNIST-10 at the paper's client count

MCLR, **40 devices in four teams of ten**, full participation, T=100, three seeds
paired, exp 2200-2205. This is the paper's stated setup. Critical value at two
degrees of freedom is 4.303.

| Metric | Direction | PerMFL | Fine Tuned | Delta | t | Wins |
|---|---|---|---|---|---|---|
| PM macro F1 | higher better | 0.9780 +- 0.0003 | 0.9786 +- 0.0007 | +0.0006 | 1.62 | 3/3 |
| PM accuracy | higher better | 0.9788 +- 0.0002 | 0.9795 +- 0.0008 | +0.0007 | 1.79 | 3/3 |
| PM macro recall | higher better | 0.9776 +- 0.0002 | 0.9784 +- 0.0008 | +0.0008 | 1.93 | 3/3 |
| PM macro FPR | lower better | 0.0024 +- 0.0000 | 0.0023 +- 0.0001 | -0.0001 | -1.84 | 3/3 |
| GM macro F1 | higher better | 0.8503 +- 0.0029 | 0.8940 +- 0.0016 | +0.0437 | 53.40 | 3/3 |
| GM accuracy | higher better | 0.8560 +- 0.0027 | 0.8978 +- 0.0014 | +0.0418 | 48.65 | 3/3 |
| GM macro recall | higher better | 0.8514 +- 0.0022 | 0.8944 +- 0.0015 | +0.0431 | 79.38 | 3/3 |
| GM macro FPR | lower better | 0.0160 +- 0.0003 | 0.0113 +- 0.0001 | -0.0046 | -52.67 | 3/3 |

The four global metrics clear the critical value by an order of magnitude. The
four personalised metrics do not clear it, every t falling between 1.6 and 1.9,
though all three seeds move the same direction on each. The correct statement is
that the personalised model is unchanged and the global model improves, not that
there is a small personalised gain.

Doubling the client count from 20 to 40 did not weaken the effect: global
accuracy gain +0.0418 against +0.0410 at 20 devices.

### Against the published figures, T=100

| | PM accuracy | vs published 96.49 | GM accuracy | vs published 91.68 |
|---|---|---|---|---|
| PerMFL | 97.88 | +1.39 | 85.60 | -6.08 |
| Fine Tuned | 97.95 | +1.46 | 89.78 | -1.90 |

### At the paper's horizon, T=400

Exp 2210-2215, three seeds paired. Critical value at two degrees of freedom
is 4.303.

| Metric | Direction | PerMFL | Fine Tuned | Delta | t | Wins |
|---|---|---|---|---|---|---|
| PM macro F1 | higher better | 0.9787 +- 0.0009 | 0.9786 +- 0.0007 | -0.0001 | -1.03 | 1/3 |
| PM accuracy | higher better | 0.9796 +- 0.0009 | 0.9795 +- 0.0008 | -0.0001 | -0.65 | 1/3 |
| PM macro FPR | lower better | 0.0023 +- 0.0001 | 0.0023 +- 0.0001 | +0.0000 | 0.62 | 1/3 |
| GM macro F1 | higher better | 0.9018 +- 0.0012 | 0.9176 +- 0.0004 | +0.0158 | 33.35 | 3/3 |
| GM accuracy | higher better | 0.9053 +- 0.0010 | 0.9206 +- 0.0004 | +0.0152 | 39.38 | 3/3 |
| GM macro recall | higher better | 0.9022 +- 0.0011 | 0.9180 +- 0.0004 | +0.0158 | 42.39 | 3/3 |
| GM macro FPR | lower better | 0.0105 +- 0.0001 | 0.0088 +- 0.0000 | -0.0017 | -41.19 | 3/3 |

### Against the published global figure of 91.68

| | Seeds | Mean | Versus published |
|---|---|---|---|
| PerMFL | 90.45, 90.65, 90.51 | 90.53 | -1.15, exceeds on 0/3 |
| Fine Tuned | 92.03, 92.09, 92.05 | 92.06 | +0.38, exceeds on 3/3 |

The arms do not overlap and the two groups fall either side of the published
number.

The defensible claim is that under the paper's own configuration, at a horizon
where both arms have converged, the decoupled variant reaches the published
global accuracy on every seed and the published coupling reaches it on none. The
shorter phrasing, that it beats the published result, invites the objection that
a peak is being compared against a figure whose horizon the paper never states.

Three observations.

The personalised model is unchanged. At T=400 the deltas are slightly negative
with t below 1 and one win in three. At T=100 they were slightly positive with t
near 1.7. Both are noise around zero, and the sign flip between horizons is
itself evidence that the small positive deltas at T=100 were not real.

The global gain shrinks with horizon but remains decisive: +0.0418 accuracy at
T=100 against +0.0152 at T=400. Longer training lets the published coupling
recover much of the gap on its own, but not all of it. The contribution is
therefore partly convergence speed and partly final quality.

Both arms have converged. The slope over the last fifty rounds is +0.0048 points
per round for PerMFL and +0.0019 for Fine Tuned, so the remaining gap is not a
horizon artefact.

Both personalised models overfit, peaking between rounds 130 and 190 and
declining by round 400. The paper reports one number per cell with no stated
horizon, so whether its figures are final or best values affects the comparison.

## 7. Variance

Metric is **GM accuracy standard deviation across seeds**, lower is better. The
Fine Tuned global model is far more stable, and this holds even on TON-IoT where
the mean regressed.

| Dataset | PerMFL | Fine Tuned |
|---|---|---|
| CICIDS2017 | 0.1885 | 0.0041 |
| TON-IoT, victim-IP | 0.0928 | 0.0004 |
| NSL-KDD | 0.0619 | 0.0063 |
| EMNIST-10 | 0.0037 | 0.0008 |

## 8. Sweeps

| Sweep | Metric | Finding |
|---|---|---|
| Local steps L, EMNIST-10, L in {2,5,10,20} | PM accuracy, higher better | monotone, L=20 best. Replicated at three seeds, see below |
| Teams p_teams, EMNIST-10, {1,2,4,10} | PM macro F1, higher better | null: 0.9815, 0.9815, 0.9815, 0.9816 |
| Horizon T, EMNIST-10, {5,100,400} | PM accuracy, higher better | 97.92, 98.38, 98.05 |
| lambda value, EMNIST-10, 20.0 against 0.5 | PM macro F1, higher better | 0.9670 +- 0.0031 against 0.9819 +- 0.0008 |
| Percentile clipping, CICIDS2017 | macro F1, higher better | +0.028 |
| `OMP_NUM_THREADS=1` | wall-clock | 1.28x, not the 2 to 4x predicted |

### Local-steps sweep, replicated

EMNIST-10, 20 devices, T=100, three seeds per point, exp 2300-2502.

| L | PM accuracy | sd | GM accuracy | sd | PM macro F1 |
|---|---|---|---|---|---|
| 2 | 95.89 | 0.184 | 76.80 | 0.819 | 0.9522 |
| 5 | 97.36 | 0.180 | 82.05 | 0.667 | 0.9698 |
| 10 | 98.02 | 0.112 | 84.44 | 0.414 | 0.9775 |
| 20 | **98.41** | 0.074 | **86.19** | 0.373 | 0.9818 |

Monotone increasing on both tiers. Every replicated mean falls within 0.11
points of the original single run, so the exploratory result was sound.

The final step is not uniformly significant. L=20 against L=10 gives +0.386 on
personalised accuracy with t = 3.91, below the critical 4.303 at two degrees of
freedom, though it wins on all three seeds. On the global model the same step
gives +1.750 with t = 4.41, which clears. The supported claim is that accuracy
increases monotonically with local depth across the range tested, with the
final step significant on the global model and marginal on the personalised one.

Variance falls as depth rises, personalised standard deviation running 0.184,
0.180, 0.112, 0.074 and global 0.819, 0.667, 0.414, 0.373. Deeper local training
makes runs more reproducible as well as more accurate.

## 8b. The lambda_team sweep

CICIDS2017, T=100, K=10, L=20, group division 3, lambda 0.05, gamma 1.5, three
seeds per point, exp 2600-2617. `lt/g` is the member-influence ratio
lambda_team / gamma.

| lambda_team | lt/g | PM macro F1 | sd | GM macro F1 | sd | PM acc | GM acc | GM FPR |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.3 | 0.4168 | 0.0206 | 0.1529 | 0.0310 | 0.8817 | 0.8312 | 0.0887 |
| 1.0 | 0.7 | 0.4383 | 0.0178 | 0.2169 | 0.0122 | 0.8860 | 0.8511 | 0.0816 |
| 1.5 | 1.0 | 0.5202 | 0.0068 | 0.2480 | 0.0072 | 0.9108 | 0.8632 | 0.0788 |
| 3.0 | 2.0 | 0.5818 | 0.0397 | 0.2603 | 0.0022 | 0.9203 | 0.8684 | 0.0757 |
| 6.0 | 4.0 | 0.6480 | 0.0046 | 0.4046 | 0.0120 | 0.9340 | 0.8859 | 0.0642 |
| **12.0** | 8.0 | **0.8440** | 0.0132 | **0.5341** | 0.0137 | 0.9576 | 0.9057 | 0.0475 |

Monotone on both tiers across the whole range, and monotone on false-positive
rate too, which falls from 0.0887 to 0.0475. The curve does not turn over. The
stability bound at eta 0.03 and gamma 1.5 permits lambda_team below 31.8, so the
sweep ran out of range rather than finding an optimum. Values of 18, 24 and 30
are untested.

Three consequences.

**Every headline result in this project is conservative.** All of them use
lambda_team = 1.5, chosen a priori with no evidence. At that point personalised
macro F1 is 0.5202; at 12.0 it is 0.8440. The reported effect sizes understate
what the parameter can do.

**The 0.6 macro F1 target is reachable.** An earlier analysis concluded 0.6 was
structurally unreachable, computing a cross-test ceiling of 0.4074. That
conclusion was correct for its configuration, cross-test on at T=10 with L=500,
and does not constrain this one.

**There is a point where tuning becomes redesign.** At lambda_team = 12.0 the
team model is weighted eight times more toward its members' average than toward
the global model. That is closer to a per-team FedAvg with a weak global anchor
than to the method the paper describes. The numbers are unambiguous, but a claim
of correcting a coupled parameter is harder to sustain at the top of this range
than at the bottom, and the write-up should choose its operating point
deliberately rather than quoting the best cell.

## 8c. What the confusion matrix shows that macro F1 does not

CICIDS2017, global model, seed 0, exp 2700 and 2701, lambda_team 1.5.

| | Classes with any true-positive rate | Attack classes sent entirely to BENIGN |
|---|---|---|
| PerMFL | 2 of 9: BENIGN 0.86, DDoS 0.69 | 3 of 8 |
| Fine Tuned | 3 of 9: BENIGN 1.00, Bot 0.39, DDoS 0.68 | 5 of 8 |

Fine Tuned wins on macro F1 and the metric is not misleading: it recovers Bot,
holds DDoS and takes BENIGN from 0.86 to 1.00.

It also sends more attack classes wholly to BENIGN, five against three. PerMFL
spreads its errors across wrong attack labels, DoS predicted as PortScan and
BruteForce as DDoS. Fine Tuned concentrates them on BENIGN. In an intrusion
detection setting these failures are not equivalent: a misclassified attack
still raises an alert, a benign classification does not. The two arms fail
differently and the better-scoring one fails more dangerously.

Neither global model is deployable at this operating point. Both miss six of
nine classes. A macro F1 of 0.15 to 0.25 always implied this; the matrix makes
it explicit, and reporting it is a stronger critical evaluation than reporting
the delta alone.

This is at lambda_team 1.5, where global macro F1 is 0.248. Section 8b shows
0.534 at lambda_team 12.0. Whether the higher setting also repairs the BENIGN
collapse or merely sharpens the same two or three classes is untested, and it
decides whether the sweep result is operationally meaningful.

## 9. Floors and ceilings

| Quantity | Metric | Value |
|---|---|---|
| CICIDS2017 majority-class floor | accuracy | 0.8171 |
| CICIDS2017 majority-class floor | macro F1 | 0.0999 |
| CICIDS2017 centralised ceiling, this config | macro F1 | 0.8994 |
| NSL-KDD normal share | proportion | 0.535 |
| TON-IoT normal share | proportion | 0.237 |
| Structural ceiling under cross-test, 4 of 20 clients per class | macro F1 | 0.4074 |
| Achieved against that ceiling | macro F1 | 0.3435, 84 per cent of it |

The structural ceiling is the mean over classes of `2(h/N) / (1 + h/N)` where `h`
is the number of clients holding the class and `N` is the client count. It is why
0.6 macro F1 was unreachable in the cross-test configuration. The bound assumes a
client cannot predict a class it never trained on, so it is conservative rather
than exact.

## 10. Local-only baseline

A client trained only on its own data beats both arms on PM macro F1 on every IDS
dataset, including under cross-test. Under cross-test it loses more than half its
score, 0.9525 to 0.4012 at one setting, because a class absent from the loss
cannot be predicted. It produces no global model at all, so it offers nothing to a
host with no local data or one that has not yet seen a given attack.

## 11. A limitation the convergence figure exposes

The IDS runs stop at T=10 (CICIDS2017, TON-IoT) and T=20 (NSL-KDD). The
convergence figure shows both arms still rising at the final round on all three,
against EMNIST-10 where the curves plateau by round 75 of 100.

The IDS comparisons therefore measure which arm rises faster over the first ten
rounds, not which converges to a better model. The direction of every IDS result
is consistent with EMNIST at a converged horizon, so the conclusion is unlikely
to reverse, but the effect sizes are not converged values and should not be
reported as though they were.

Closing this needs the IDS headline re-run at T=100 with L=20, matching EMNIST
and the paper's loop shape. See backlog B2b.

## 11b. The clustering has never been tested as designed

`--group_division 3` engages `team_former.py`, the MCTC formation with the
CFMD-i drift trigger. B7 is the first batch whose logs prove it runs at all: it
reclusters on 99 of 100 global rounds and reports adjusted Rand index against
the loader's day-domain grouping every time.

Two findings.

**The trigger is not gating.** `should_recluster` returns
`mx > eps_hi and mean < eps_lo`. With the default `eps_hi = 0.0` and
`eps_lo = infinity` this is true whenever any two clients differ, so it fires
unconditionally. No run has ever supplied real thresholds. What has been
measured is per-round reclustering, not adaptive reclustering. See defect 22.

**The partition never converges.** ARI oscillates between roughly 0.15 and 0.65
for the whole run with no trend, mean 0.366. Teams are rebuilt into a
substantially different partition every round.

This gives the earlier null result a mechanism. Oracle, random and derived team
assignment were statistically indistinguishable across five comparisons. If
derived teams are re-randomised each round then derived is closer to random each
round than to any stable structure, so the null is expected rather than
surprising.

Whether the clustering can find stable structure when the gate actually gates is
untested. `should_recluster` already returns the observed max and mean pairwise
distance so plausible thresholds can be read off a run and supplied.

## 12. Coverage gaps

| Missing | Detail |
|---|---|
| Published datasets not run | MNIST, FMNIST, CIFAR-10, CIFAR100, FEMNIST, Synthetic with PerMFL |
| Baselines not run | FedAvg, Ditto, PerAvg, AL2GD, pFedBayes on any dataset. h-QSGD cannot be run at all, defect 16 |
| Clustering | `team_former.py` has never run with a working trigger, see section 11b |
| Baselines partially run | pFedMe on NSL-KDD only, two runs. hierarchical-FedAvg on Synthetic, one run |
| Configuration mismatch | every EMNIST run used 20 devices; the paper states 40 |
| Unattributed | exp 900-905 on NSL-KDD do not follow the parity convention and are not labelled |
| Single-run results | the communication sweep has no seed replication. The local-steps sweep was replicated at three seeds, see section 8 |
