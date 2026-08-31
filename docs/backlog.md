# Run backlog

Every run in this project is CPU-only on the Mac. There is no GPU work and no
lab-machine queue. Items requiring hardware we do not have have been removed
rather than deferred.

Terminology follows [results.md](results.md): **PerMFL** is the method as
published, **Split-λ** sets the team coefficient independently.

## Done

52 runs, five batches, zero failures, 3.7 hours of compute. Logs in
`logs/<batch>/` with command, git SHA, environment and exit code per run.

| Batch | Runs | What it settled |
|---|---|---|
| B1 | 10 | CICIDS headline at a converged horizon, five seeds. Confusion matrices now persisted |
| B2 | 6 | EMNIST at the paper's 40 devices. Effect holds |
| B2L | 6 | EMNIST at the paper's T=400. Split-λ reaches the published global figure on 3/3 seeds, PerMFL on 0/3 |
| B3 | 12 | Local-steps sweep replicated at three seeds. Monotone confirmed |
| B7 | 18 | λ_team sweep, 0.5 to 12.0. Monotone throughout, no turnover |

B1 also delivered what was filed as B2b, since it runs at T=100 with L=20.

## Not doing

| Item | Why |
|---|---|
| CNN and non-convex half of the paper's Table 1 | four to eight hours per arm on CPU. Was queued for a lab GPU we are not using |
| FEMNIST and CIFAR100, the paper's Table 3 | 3,500 and 350 clients. Not feasible on this machine |
| Baseline algorithms across all published datasets | would need macro F1 instrumentation on six servers plus the runs |

These become stated limitations in the report rather than gaps to be filled.

## Still open, cheap, CPU-only

Ordered by what each buys. None is required for the current claims.

| # | Run | Cost | Buys |
|---|---|---|---|
| C1 | Clustering trigger with real `--eps_hi` and `--eps_lo` | 6 runs, ~20 min | Decides whether the CFMD-i trigger can be claimed as working. Currently the gate is open by default and has never been tested as designed. See defect 22 |
| C2 | λ_team above 12.0, values 18 and 24 | 6 runs, ~20 min | The sweep ran out of range rather than finding an optimum. Would establish where the curve turns |
| C3 | Confusion matrices at λ_team 12.0 | 2 runs, ~7 min | Whether the higher setting repairs the BENIGN collapse or only sharpens the same classes. Decides if the sweep result is operationally meaningful |
| C4 | TON-IoT and NSL-KDD at T=100, L=20 | 20 runs, ~70 min | Those two datasets are still on the unconverged T=10 and T=20 horizons |
| C5 | Weighted aggregation, `--weighted_agg 1` | 6 runs, ~20 min | Implemented and never swept |
| C6 | Communication sweep replicated | 9 runs, ~30 min | The last remaining single-run result |
| C7 | Class-weighted loss, `CLASS_WEIGHTS=1` | 6 runs, ~20 min | The loss is unweighted `NLLLoss` while the metric is macro F1, so the objective and the reported measure disagree on an 81.7 per cent benign corpus. Candidate explanation for the global model collapsing to BENIGN. Implemented and flag-gated; off by default so the 52 completed runs stay comparable |

Total if all seven run: 55 runs, roughly three and a half hours.

## Recommendation

**C1, C3 and C7 are worth running; the rest are optional.**

C1 because the report currently describes drift-triggered reclustering as
delivered, and no run has exercised the trigger. That is a claim the evidence
does not yet support either way.

C3 because the strongest number in the project, λ_team 12.0, has no per-class
evidence behind it, and the confusion matrix at 1.5 showed the better-scoring
arm failing more dangerously.

C7 because it tests a specific, cheap hypothesis about why both global models
collapse to BENIGN. If class weighting moves per-class recall it is a citable
future-work direction; if it does not, that is equally worth recording, since it
removes the obvious explanation and points elsewhere.

Everything else is refinement. The current results already cover four datasets,
both tiers, four metrics, five seeds on the headline, and the paper's own
configuration at its own horizon.
