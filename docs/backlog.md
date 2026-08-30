# Run backlog

Runs not yet done, ordered by what they buy the dissertation. Nothing here has
been executed. Terminology follows [results.md](results.md): **PerMFL** is the
method as published, **Fine Tuned** is `--lamda_team 1.5`.

Time estimates are extrapolated from measured throughput, EMNIST-10 MCLR at
T=100 with 20 devices taking 2.5 minutes per run at two-way parallelism. They
are rough and I have underestimated before.

## Tier 1. Needed for claims already made in the draft

### B1. Confusion matrix persistence, then a re-run

**Blocks the confusion matrix figures entirely.** Matrices are computed in
`serverPerMFL.py:201` and `per_client_report`, printed to stdout, and never
saved. The h5 stores only the derived scalars. No run log survives that contains
a matrix.

Needs a patch to `save_results` writing `cm_global` and `cm_personal` as h5
datasets, then a re-run of the headline configuration, CICIDS2017 exp 906-915,
both arms, five seeds. Ten runs.

Without this there is no confusion matrix in the report, on any dataset.

### B2. EMNIST-10 at the paper's stated configuration

Every EMNIST run used 20 devices in four teams. The paper states **40 devices in
four teams of ten**, full participation, T=400. The current reproduction and the
Fine Tuned comparison are therefore not like-for-like against either published
figure, 96.49 PM or 91.68 GM.

Both arms, three seeds, six runs. Roughly 20 minutes each, so about 90 minutes at
two-way parallelism.

### B3. Seed replication for the single-run sweeps

The local-steps sweep and the communication sweep are reported as exploratory
because they have no replication. Chapter 7 flags this and it is highlighted
yellow in the Word draft. Three seeds each would let them be stated plainly.

Local steps L in {2,5,10,20} at three seeds is twelve runs, about 30 minutes.

## Tier 2. Fills the published comparison

### B4. Baseline algorithms on EMNIST-10 at the paper's configuration

The paper compares against seven methods. We have run two, on datasets the paper
does not use. This is the largest single gap and it is what "no comparison
against related methods" in Chapter 7 refers to.

| Algorithm | Runnable now? |
|---|---|
| `FedAvg` | yes |
| `pFedMe_original` | yes, already reports macro F1 |
| `ditto` | yes |
| `PerAvg` | yes |
| `pFedBayes` | yes |
| `AL2GD` | yes |
| `Hier-Local-QSGD` | **no**, defect 16. One-line fix needed first |

Six or seven runs, one seed each to start. These report accuracy only except
pFedMe, so a macro F1 instrumentation pass would be needed for a like-for-like
table. That is the same `confusion` method already added to `userbase.py`.

### B5. The paper's other datasets

MNIST, FMNIST and Synthetic with PerMFL are three of the four columns in the
paper's Table 1 and have zero runs. MCLR only, both arms, three seeds is
eighteen runs.

### B6. The non-convex half of Table 1

Every run in this project is MCLR. The paper's Table 1 has a second block using
CNN for image datasets and a two-hidden-layer DNN for Synthetic. EMNIST-10 CNN
was attempted once and failed on defect 9. Estimated four to eight hours on CPU
for a single arm.

Related and cheap: `HIDDEN_LAYERS=2` matches the paper's stated two-hidden-layer
DNN and has never been run on any IDS dataset. The flag exists and defaults to 1.

## Tier 3. Strengthens the contribution

### B7. lamda_team sweep

Only two values have ever been used, 0.5 and 1.5. There is no evidence 1.5 is
the right choice, and the stability bound permits up to about 31.8 at
eta 0.03 and gamma 1.5. A sweep over {0.5, 1.0, 1.5, 3.0, 6.0, 12.0} on CICIDS
at one heterogeneity setting, three seeds, is eighteen runs and would turn the
contribution from one setting into a characterised curve.

### B8. Weighted aggregation

`--weighted_agg 1` is implemented and never swept. It is not recorded in the h5,
so it cannot be confirmed whether any existing run used it. Two arms at three
seeds on CICIDS, six runs.

### B9. Team re-formation

`--group_division 3` and the whole of `FLAlgorithms/clustering/team_former.py`,
the MCTC and CFMD-i work, cannot be confirmed to appear in any recorded result.
The relevant flags are not persisted. If it was run, the runs are unlabelled; if
it was not, 161 lines of implemented code have no evidence behind them.

Needs establishing either way before Chapter 5 can claim it as delivered.

### B10. Diagnosing the TON-IoT domain regression

The one negative result, exp 916-925. Fine Tuned global accuracy pins to 0.2369,
the majority-class floor, in four of five seeds. Worth a small diagnostic set at
intermediate `lamda_team` values to find where the collapse begins. Six runs.

## Tier 4. Optional

| # | Run | Value |
|---|---|---|
| B11 | `num_teams` and K sweeps at five seeds | the existing team-tier nulls rest on single runs |
| B12 | FEMNIST and CIFAR100, the paper's Table 3 | completes the published comparison, expensive |
| B13 | Percentile clipping at three seeds | currently a single-run +0.028 |

## Summary

| Tier | Runs | Rough time |
|---|---|---|
| 1 | 28 | 3 hours |
| 2 | 25 plus the CNN block | 1 day |
| 3 | 30 | 4 hours |
| 4 | unbounded | - |

If only one thing is run, make it **B1**. It is the only item that unblocks a
figure the draft is expected to contain, and confusion matrices are standard in
every intrusion detection paper cited in Chapter 2.

If two, add **B2**, because it converts the reproduction from approximate to
like-for-like against the paper's own numbers.
