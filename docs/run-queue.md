# Experiment queue, archived

Imported from the `wip/paper-config` branch. A dated working queue, kept for the
reasoning it records rather than the schedule. It is **historical**.

Of lasting value and not repeated elsewhere: the T1 local-steps sweep with
wall-clock figures, the team-size constraint analysis in "Design constraint"
with the per-team label distributions, and the four blocked items, all four of
which were subsequently resolved.

---

# Experiment queue

Companion to `run.md`, which has the environment setup and the exact command
forms. This file tracks what has run, what runs today on the laptop, and what
was cut; all work is CPU-only.

Standard config unless a row says otherwise.

```
--dataset Emnist10 --model_name mclr --algorithm PerMFL
--lamda 0.5 --gamma 1.5 --beta 0.6 --alpha 0.01 --eta 0.03
--num_global_iters 100 --num_team_iters 10
--tot_users 20 --num_teams 4 --numusers 5 --p_teams 4
--num_labels 2 --group_division 0
OMP_NUM_THREADS=1
```

Twenty clients, not ten. Ten forces `--num_teams 2`, and the research question
is whether derived team assignment beats random, which needs more than two
teams to vary. Ten clients with four teams raises `IndexError`, see defect 15.

Always set `--exp_start N --times N+1`, and give every run a distinct N. The
output filename encodes only exp_no, lambda, gamma, beta, model and dataset, so
runs that differ only in T, client count or local_iters overwrite each other
silently. Defects 13 and 14.

---

## Done

| id | exp_no | config | PM | GM | wallclock |
|---|---|---|---|---|---|
| R0 | - | hier-FedAvg, Synthetic, dnn, shipped defaults | - | 85.18 | 299 s |
| R2 | 0 | 40 clients, T=100, **baseline arm** | 96.35 | 77.72 | 381 s |
| R3 | 1 | 40 clients, T=100, fixed arm | 97.85 | 85.29 | 335 s |
| R4 | 2 | 40 clients, T=400, **baseline arm** | peak 96.45 | 87.84 | 1542 s |
| R5 | 3 | 40 clients, T=400, fixed arm | peak 98.05 | 90.45 | 1623 s |
| A | 4 | 40 clients, T=100, threads=1 | 97.85 | 85.29 | 262 s |
| B | 5 | 20 clients, T=100, threads=14 | 98.34 | 85.79 | 239 s |
| C | 6 | 20 clients, T=100, threads=1, L=20 | 98.34 | 85.79 | 145 s |

R4 is the reproduction. The baseline arm carries the lamda bug, which is what
the released code does, and its PM peak of 96.45 sits 0.04 from the paper's
96.49. That comparison is banked and does not need repeating.

Thread count changes no result, verified bit-identical in both pairs.

---

## Today, M3 Max, nothing over 1 hour

### T1. local_iters sweep - DONE, hypothesis refuted

| L | local steps | PM final | PM peak @r | GM final | wallclock |
|---|---|---|---|---|---|
| 2 | 40,000 | 95.86 | 95.86 @99 | 75.87 | 45 s |
| 5 | 100,000 | 97.25 | 97.25 @96 | 81.32 | 63 s |
| 10 | 200,000 | 97.94 | 97.96 @93 | 83.99 | 89 s |
| **20** | 400,000 | **98.34** | **98.38 @76** | **85.79** | 145 s |

Monotonic. More local steps is better on both metrics, and L=20 is best.

**The overtraining hypothesis was wrong.** The effective-epoch arithmetic was
correct, the median 20-client really does make about 1375 passes over its own
data across T=100, but it is not hurting. The device update carries a proximal
term, `theta <- theta - alpha*grad - alpha*lambda*(theta - w)`, which pins each
device to its team model and prevents the drift that thousands of unconstrained
epochs would cause. That is exactly the claim in the paper's Remark 2, and it
holds empirically. Report it as a positive result about the method.

Caveat on the comparison. L=2, 5 and 10 peak at rounds 99, 96 and 93, so none
had converged at T=100, and this table compares total compute as much as it
compares L. A fair test would equalise total local steps, L=5 at T=400 against
L=20 at T=100. That said, in FL more local work per round means *less*
communication, so L=20 already wins on both accuracy and communication cost and
there is no free saving here to take.

Keep `--local_iters 20`.



`L in {2, 5, 10}` against the existing L=20 at exp_no 6. exp_no 7, 8, 9.

Why. `--local_iters` is minibatch steps, not epochs. At `K*L = 200` steps of
batch 124, the median 20-client holds 1803 samples and therefore makes about
13.8 passes over its own data **per global round**, 1375 over T=100. At 40
clients the median is 2887. That is memorisation, and it is why fixed-arm PM is
within one point of its peak before round 1 completes and declines after.

Decides whether the team tier still does anything once the devices stop being
overtrained, and whether we can cut 4x the compute for free.

### T2. Baseline algorithms at the standard config

Eight algorithms are reachable: `FedAvg`, `hierarchical-FedAvg`, `PerAvg`,
`pFedMe_original`, `AL2GD`, `ditto`, `pFedBayes`, `PerMFL`.

**h-QSGD is not.** `--algorithm Hier-Local-QSGD` passes argparse but the
dispatch at `main.py:115` tests `Hier_local_qsgd`, so `server` is never assigned
and `server.train()` raises `NameError`. h-SGD is the paper's headline
multi-tier baseline in Table 1 and cannot be launched from the shipped CLI.
Defect 16.

Smoke each for 3 rounds before committing to a full run. `PerAvg` is the one to
watch, it is the only algorithm reaching `MySGD` and its deprecated
`p.data.add_(-beta, d_p)` overload, which is why the environment is pinned to
torch 2.3.1.

Budget: 8 smokes at well under a minute, then roughly 5 full runs at 2 to 5 min.

### T3. Team assignment control

`--group_division 0` (sequential) against `--group_division 1` (random) at
otherwise identical settings. This is the control for the whole project, per
`02-LESSONS.md` point 1.

Note that `group_division 1` reseeds on `time.time()` at every loader, and that
clock seed then leaks into `random.sample` team selection at
`serverPerMFL.py:667`. So the random arm is a different draw every run and needs
several repeats to characterise, while the sequential arm is deterministic.
Defect 3.

### T4. CICIDS2017 loader, development only

Build and smoke `read_cicids_data`. No long run today. Design questions are
open, see Blocked below.

---

## Lab-machine section removed

The GPU queue that stood here was cut. All work is CPU-only on the Mac.

## L1. EMNIST-10 CNN, both arms

The non-convex half of Table 1. Paper targets PM 98.79, GM 93.12.

`cnn_emnist(output_dim=10)` is 2,362,874 parameters against MCLR's 7,850, so
2194 TFLOP per run against 4.67. Roughly 5 hours per arm on the M3 Max CPU,
roughly 12 minutes on a 3070.

Test `OMP_NUM_THREADS` 10 against 14 here rather than assuming. Threading is
counterproductive for MCLR because the matmuls are tiny, but the 18432x128
layer is large enough to pay. Ten keeps the parallel region on performance
cores. Do **not** carry `OMP_NUM_THREADS=1` over from the MCLR runs.

Memory: six full model copies per client in `userbase.__init__`, so 2.3 GB of
model state at 40 clients and 1.1 GB at 20. Fits 8 GB comfortably, but it
scales linearly with client count.

### L2. CNN baselines

Same eight algorithms as T2, CNN model. Only worth doing once L1 confirms the
CNN path runs clean.

### L3. Five-seed repeats

**Blocked on a code change.** `--times N` currently produces N identical runs.
`torch.manual_seed(0)` is hardcoded at `main.py:15`, the loaders fix
`random.seed(5)` and `np.random.seed(9)`, and `select_users` seeds on the round
index. With `--group_division 0` nothing varies.

This is why almost every PerMFL entry in Table 1 carries a std of exactly
`(+/-0.0)` while the baselines show real spread. Deterministic repeats produce
exactly that signature. Stated as a hypothesis, not proven.

`02-LESSONS.md` point 6 requires five seeds minimum. Needs `torch.manual_seed`
parameterised off a `--seed` flag.

---

## Design constraint: team size must be under 10, or the experiment is vacuous

Measured, not inferred. `read_EMnist10_data` assigns label sets with
`l = (user + j) % 10`, a **sliding window of period 10**, not a partition.
Client 0 holds {0,1}, client 1 holds {1,2}, client 9 holds {9,0}, and client 10
holds {0,1} again. Adjacent clients share a class, clients ten apart are
label-identical.

Sequential teams of 10 consecutive clients therefore tile the ring exactly once.

**40 clients, 4 teams of 10, the paper's own configuration:**

| team | classes covered | label distribution, classes 0-9 |
|---|---|---|
| 0 | all 10 | .08 .10 .10 .07 .07 .12 .17 .08 .09 .12 |
| 1 | all 10 | .06 .11 .06 .14 .05 .05 .16 .07 .11 .19 |
| 2 | all 10 | .10 .10 .07 .12 .16 .10 .09 .08 .07 .10 |
| 3 | all 10 | .23 .07 .08 .09 .06 .09 .07 .13 .12 .07 |

Every team covers every class at roughly uniform proportions. The teams are
statistically interchangeable, so **no team assignment can beat any other and
the research question cannot be answered at this configuration.**

**20 clients, 4 teams of 5:**

| team | classes covered | label distribution |
|---|---|---|
| 0 | 0-5 only | .08 .17 .17 .19 .23 .16 .00 .00 .00 .00 |
| 1 | 0, 5-9 | .07 .00 .00 .00 .00 .07 .12 .18 .13 .44 |
| 2 | 0-5 only | .05 .07 .15 .14 .56 .02 .00 .00 .00 .00 |
| 3 | 0, 5-9 | .17 .00 .00 .00 .00 .20 .18 .12 .12 .21 |

Teams of 5 cover half the ring, so teams 0 and 2 hold zero samples of classes
6 to 9. That is real between-team structure.

General rule: consecutive-client teams tile the ring whenever team size is at
least the period, which equals the class count. **Team size must be under 10.**

This also explains Table 2 of the paper. The authors hand-constructed the
worst case (team 1 = {0,1,2,3,4}, team 2 = {5..9}) at two teams rather than
four, because their default partitioning produces no team structure to ablate.

Consequence. Run all team-assignment work at 20 clients in 4 teams of 5, and
state in the methodology why 40 cannot work. It is also the strongest argument
for adding Dirichlet partitioning, which has no periodic structure, so team
composition stops being an artefact of client index ordering.

---

## Blocked, needs a decision

### B1. Seed parameterisation

One line at `main.py:15`. Nothing statistical is reportable until this exists.
Not yet authorised.

### B2. h-QSGD name mismatch

One line. Restores the paper's main multi-tier baseline.

### B3. CICIDS2017 temporal split

The design question that shapes the loader. Every loader in this repo does
`random.shuffle` then a flat 75/25 cut. `02-LESSONS.md` point 2 says that split
is what manufactured the previous project's reversed result, 88.1% of test rows
having a near-twin in training.

Two options, and they are not equivalent:

- split globally by timestamp, then partition clients from each side
- partition clients first, then split each client by its own timeline

These give different heterogeneity and different leakage properties. Needs a
decision before the loader is written.

### B4. Macro F1

`02-LESSONS.md` points 3 and 4. Accuracy is uninformative at 73% BENIGN, and
per-client scoring without an explicit label set inflates personalised methods,
by up to 0.37 in the previous project.

This is already visible on EMNIST-10. Each device holds 2 of 10 classes and its
personal model is scored only on its own slice, so PM 98.34 is an average of
forty 2-class problems while GM 85.79 is a genuine 10-class problem. The paper
prints them adjacent in one table. On CICIDS with 12 classes the same effect
would be far worse.

Requires changing the server `evaluate` methods, and must pass
`labels=list(range(num_classes))`.
