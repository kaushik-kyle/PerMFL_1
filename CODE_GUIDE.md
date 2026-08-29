# PerMFL code guide

## 0. Branches

| Branch | Contents |
|---|---|
| `main` | upstream, byte-identical to `origin/main` at `85f46c6`. Never modified |
| `dev` | the two minimum fixes only: lambda argument shift, Synthetic seeding. A sibling of `dev_cicids`, not an ancestor; the only commit unique to it is a README |
| `dev_cicids` | **the proposal.** Branches off the same fix commit `02af70c` that `dev` does, plus everything in section 4 |
| `wip/paper-config` | local only. Holds `run.md` and `RUNS.md`, the run book and defect catalogue |
| `wip/cicids`, `wip/emnist10-data` | local only, superseded, kept for provenance |

`main` is a strict ancestor of `dev_cicids`, so `git diff main dev_cicids -- '*.py'`
is the complete proposal: 1,442 insertions, 24 deletions, 14 files.
`git diff dev dev_cicids -- '*.py'` isolates everything past the two fixes:
1,297 insertions, 20 deletions, 13 files.

Branch `dev_cicids`. 10,047 tracked Python lines, of which the PerMFL path is ~1,700.
Everything else is the seven baseline algorithms shipped by the original authors.

## 1. Execution path

```
main.py:__main__          argparse, _seed_everything(args.seed)
main.py:main()            builds model, dispatches on --algorithm
  utils/model_utils.py:read_data / read_user_data     dataset dispatch
    utils/cicids.py | nslkdd.py | toniot.py           our loaders
  FLAlgorithms/servers/serverPerMFL.py:PerMFL.__init__ builds teams, wraps users
    FLAlgorithms/users/userPerMFL.py:UserPerMFL        one per client
  PerMFL.train()          the T x K x L loop
```

`main.py:25` is `while i < times` seeded from `--exp_start`. `--exp_start 1 --times 1`
runs nothing and exits 0. Use `--exp_start N --times N+1`.

## 2. The algorithm, six equations

Three parameter sets, coupled by squared Euclidean penalties.
`theta` device, `w` team, `x` global.

| # | Update | Code |
|---|---|---|
| 1 | `theta <- theta - alpha*grad - alpha*lamda*(theta - w)` | `optimizers/fedoptimizer.py:89` |
| 2 | `theta_bar = mean_i(theta_i)` | `serverPerMFL.py:328` `aggregate_parameters` |
| 3 | `w <- (1 - eta*lamda_team - eta*gamma)*w + eta*gamma*x_old + lamda_team*eta*theta_bar` | `serverPerMFL.py:407` |
| 4 | `w_bar = mean_m(w_m)` | `serverPerMFL.py:424` |
| 5 | `x <- (1 - beta*gamma)*x_old + beta*gamma*w_bar` | `serverPerMFL.py:441` |
| 6 | eval on both `theta` (PM) and `x` (GM) | `serverPerMFL.py:669` |

Loop nesting in `train()` at `serverPerMFL.py:823`:

```
for t in range(num_global_iters):        # T, eq 5 fires once per t
  set_old_global_parameter()             # x_old <- x
  sub_group = random.sample(team_list, p_teams)
  for grp in sub_group:
    set_team_parameter(grp)              # w <- x, teams are reset every t
    for k in range(num_team_iters):      # K, eq 3 fires once per k
      select_users(k, numusers, grp)
      send_parameters(grp)
      for user in selected: user.train(local_iters, team_params)   # L, eq 1
      aggregate_parameters()             # eq 2
      personalized_team_aggregate_parameters(grp)                  # eq 3
  server_level_aggregate_parameters()    # eq 4
  global_update()                        # eq 5
  evaluate()
  maybe_recluster(t)                     # ours
```

Device transfers per global round = `2 * num_team_iters * numusers * p_teams`.
`num_teams` cancels out of the per-round cost. Only K and the client count move it.

**The lambda double duty.** In upstream, the same `--lamda` is both the device
proximal coefficient in eq 1 and the team's weight on its members' average in
eq 3. Theory requires `gamma > 2*lamda`, so member influence `lamda/gamma` is
capped below 50%, and at the paper defaults it is 3.3%. `--lamda_team` splits
the two roles. That split is the contribution.

## 3. Read order

| # | File | Look for |
|---|---|---|
| 1 | `serverPerMFL.py:823` `train` | the T/K/L nesting above, nothing else |
| 2 | `serverPerMFL.py:407,424,441` | eq 3, 4, 5 verbatim |
| 3 | `userPerMFL.py:37` `train` | eq 1, plus the `local_model` copy-back |
| 4 | `optimizers/fedoptimizer.py:71` `pFedMeOptimizer` | the actual proximal step |
| 5 | `serverPerMFL.py:669` `evaluate` | which metrics land in which list |
| 6 | `metrics.py` | `macro_f1`, `full_report`, 73 lines |
| 7 | `utils/cicids.py:174` `read_cicids_data` | partition, split, thinning order |
| 8 | `clustering/team_former.py` | signal, trigger, MCTC assignment |

Skip `utils/model_utils.py` (1,452 lines) except `read_user_data`. Skip every
`servers/*` file that is not `serverPerMFL.py`.

## 4. What we changed

| File | Change | Rationale |
|---|---|---|
| `users/userPerMFL.py:12` | keyword args in `super().__init__` | positional binding put `--lamda` in `self.eta` and `--local_iters` in `self.lamda`, so the optimizer received lamda=20 |
| `main.py:20` | `_seed_everything(seed)` | `torch.manual_seed(0)` was hardcoded, so `--times N` gave N identical runs |
| `serverPerMFL.py:95` | `self.lamda_team` | splits eq 1's coefficient from eq 3's |
| `serverPerMFL.py:96,424` | `weighted_agg` | eq 4 sample-proportional instead of uniform |
| `serverPerMFL.py:186` | `_pooled_f1` | confusion matrices pooled over clients |
| `serverPerMFL.py:207` | `per_client_report` | per-client breakdown at end of run |
| `serverPerMFL.py:257` | `maybe_recluster` | CFMD-i drift trigger, MCTC reassignment |
| `metrics.py` | new, 73 lines | macro F1, macro recall, macro FPR |
| `clustering/team_former.py` | new, 161 lines | `client_signal`, `should_recluster`, `form_teams`, `agreement` |
| `utils/cicids.py` | new, 414 lines | leak-free CICIDS2017 loader |
| `utils/nslkdd.py` | new, 152 lines | canonical KDDTrain+/KDDTest+ |
| `utils/toniot.py` | new, 173 lines | TON-IoT Network |
| `users/userbase.py` | `confusion`, `confusion_personalized` | vectorised confusion matrix via `np.bincount`; a Python zip over 100k rows per client per round dominated the run |
| `users/userpFedMebase.py` | same two methods | personal model is `persionalized_model_bar` here, not `local_model` |
| `servers/serverpFedMeoriginalbase.py:248` | `pooled_f1`, printed in `evaluate_personalized_model` | makes pFedMe's macro F1 directly comparable to PerMFL's |
| `trainmodel/models.py` | `hidden_layers` arg on the DNN; `output_dim` on the EMNIST CNN | the paper's Section 4 and D.3 say two hidden layers, the release has one. Default stays 1 so shipped results reproduce |
| `utils/model_utils.py` | +157 lines | dataset dispatch for `Emnist10`, `Cicids`, `Nslkdd`, `Toniot` |

Of the seven baselines only pFedMe reports macro F1. FedAvg, Ditto, PerAvg,
L2GD, h-QSGD and pFedBayes are untouched and still report accuracy only.

## 5. Glossary: CLI flags

### Upstream, live for PerMFL

| Flag | Default | Symbol | Meaning |
|---|---|---|---|
| `--alpha` | 0.01 | alpha | device learning rate, eq 1 |
| `--lamda` | 5.0 | lambda | device-to-team proximal coefficient, eq 1; also eq 3's weight on `theta_bar` unless `--lamda_team` is set |
| `--eta` | 0.03 | eta | team learning rate, eq 3 |
| `--gamma` | 5.0 | gamma | team-to-global coupling, eq 3 and eq 5. Must satisfy `gamma > 2*lamda` |
| `--beta` | 0.99 | beta | global learning rate, eq 5 |
| `--num_global_iters` | 100 | T | outer rounds |
| `--num_team_iters` | 10 | K | team rounds per outer round |
| `--local_iters` | 20 | L | local SGD steps per team round. One minibatch per step, not one epoch |
| `--num_teams` | 4 | M | teams. Cancels out of per-round communication |
| `--p_teams` | 4 | - | teams sampled per outer round. `p_teams <= num_teams` |
| `--tot_users` | 40 | N | total clients |
| `--numusers` | 10 | - | clients sampled per team round |
| `--num_labels` | 2 | - | labels per client for the built-in label-skew splitter. Ignored by our IDS loaders |
| `--batch_size` | 124 | - | minibatch |
| `--group_division` | 1 | - | 0 sequential, 1 random, 2 single team, 3 derived from client updates |
| `--model_name` | dnn | - | `dnn`, `mclr`, `cnn`, `pbnn`. No `dnn` branch exists for `Emnist10` |
| `--dataset` | Synthetic | - | `Mnist`, `Synthetic`, `Cifar10`, `FMnist`, `Emnist10`, `Cicids`, `Nslkdd`, `Toniot` |
| `--algorithm` | hierarchical-FedAvg | - | set to `PerMFL` |
| `--exp_start` / `--times` | 0 / 1 | - | loop bounds, not counts. See section 1 |
| `--gpu` | 0 | - | -1 for CPU |

### Upstream, dead for PerMFL

| Flag | Why dead |
|---|---|
| `--K` (5) | pFedMe inner steps. `self.K = K` is commented out at `userPerMFL.py:25` |
| `--personal_learning_rate` (0.005) | pFedMe only. Never read by PerMFL |
| `--selected_users` (10) | defined in argparse, never passed to `main()`. Use `--numusers` |
| `--analysis` (perf) | only builds the results directory path, `serverPerMFL.py:465` |
| `--weight_scale`, `--rho_offset`, `--zeta` | pFedBayes only |
| `--optimizer` (SGD) | string is stored, never branched on |

### Ours

| Flag | Default | Meaning |
|---|---|---|
| `--seed` | 0 | seeds torch, numpy, random, and exports `PERMFL_SEED` for the loaders |
| `--lamda_team` | None (falls back to `--lamda`) | eq 3's weight on `theta_bar`, decoupled from eq 1 |
| `--weighted_agg` | 0 | 0 uniform as Algorithm 1 specifies, 1 sample-proportional |
| `--team_signal` | residual | `residual` uses `theta - w`, `grad` uses the raw update as SCMoE-PFL does |
| `--recluster_from` | 1 | first outer round at which reclustering is considered |
| `--recluster_every` | 1 | 1 reform every round, 0 form once at `--recluster_from` then hold |
| `--eps_hi` | 0.0 | CFMD-i: recluster only if max pairwise client difference exceeds this |
| `--eps_lo` | inf | CFMD-i: and only if mean pairwise difference stays below this |
| `--pca_dim` | 8 | MCTC: PCA components before cosine similarity |

## 6. Glossary: environment variables

Loaders read env vars, not flags, because `read_data` has a fixed signature.

### CICIDS2017, `utils/cicids.py`

| Var | Default | Meaning |
|---|---|---|
| `CICIDS_DIR` | see :41 | `TrafficLabelling` directory |
| `CICIDS_CACHE` | `data/cicids_clean.npz` | parsed cache. Delete to force a re-parse |
| `CICIDS_PARTITION` | domain | `domain`, `domainmix`, `labelskew`, `dirichlet` |
| `CICIDS_SPLIT` | temporal | `temporal` or `random`. Temporal splits within day, no leakage |
| `CICIDS_DOMAIN_MIX` | 1.0 | fraction of a client's rows drawn from its own day. 1.0 is pure domain |
| `CICIDS_BENIGN_FRACTION` | 1.0 | benign thinning, **train only**. Thinning test moves the accuracy floor |
| `CICIDS_MIN_SAMPLES` | 2000 | drop classes below this |
| `CICIDS_MAX_PER_CLIENT` | 20000 | cap per client |
| `CICIDS_RARE_FLOOR` / `_RARE_BELOW` | 2000 / 20000 | rare-class replication band |
| `CICIDS_CLIENTS_PER_RARE` | 3 | how many clients hold each rare class. Sets the cross-test ceiling |
| `CICIDS_CLASSES_PER_CLIENT` | 3 | labelskew only |
| `CICIDS_DIRICHLET_ALPHA` | 0.5 | dirichlet only. Lower is more skewed |
| `CICIDS_CROSS_TEST` | 0 | 1 evaluates clients on classes absent from their training set |
| `CICIDS_CLIP_PCT` | 0 | percentile clip on features |
| `CICIDS_MIN_CLIENT_ROWS` | 200 | drop clients below this |
| `CICIDS_TEAM_SEED` | 0 | shuffles the team assignment order only |

### NSL-KDD, `utils/nslkdd.py`

`NSLKDD_DIR` (`data/nsl_kdd`), `NSLKDD_PARTITION` (domain),
`NSLKDD_DIRICHLET_ALPHA` (0.5), `NSLKDD_MIN_CLIENT_ROWS` (200),
`NSLKDD_TEAM_SEED` (0). Split is the canonical KDDTrain+/KDDTest+, not resampled.

### TON-IoT, `utils/toniot.py`

`TONIOT_DIR`, `TONIOT_PARTITION` (domain, `domain`|`victimip`|`dirichlet`),
`TONIOT_MIN_ENTROPY` (0.2, DP-FL victim-IP filter), `TONIOT_ALPHA` (0.5),
`TONIOT_MIN_CLIENT_ROWS` (200), `TONIOT_CROSS_TEST` (0).

### Global

`PERMFL_SEED` set by `--seed`, read by every loader.
`TONIOT_FEATS` (38) and `HIDDEN_LAYERS` (1) read at `main.py:16-17`.

## 7. Glossary: internal attributes whose names mislead

| Attribute | Actually is |
|---|---|
| `self.local_epochs` | `--local_iters`, minibatch steps not epochs |
| `self.team_iters` | `--num_team_iters` |
| `self.group` | `--num_teams`, an integer count |
| `self.users` | list of lists, indexed by team |
| `self.tau[grp]` | team sample share, only used when `weighted_agg=1` |
| `self.theta_bar` | a model object reused as an accumulator, zeroed in `aggregate_parameters` |
| `self.old_global_model` | `x` at the start of the outer round, the `x_old` in eq 3 and 5 |
| `self.w_bar` | mean of team models, eq 4 |
| `self.model` | the global model `x` |

## 8. Traps

1. `--exp_start N --times N` runs nothing, exit 0. Section 1.
2. `--K` is not the paper's K. The paper's K is `--num_team_iters`.
3. `--model_name dnn` has no branch for `Emnist10`. Use `mclr`.
4. `gamma > 2*lamda` is a convergence condition. Violating it does not raise.
5. Benign thinning must apply to train only. Thinning test moved the CICIDS
   accuracy floor from 0.8171 to 0.7008.
6. Our loaders materialise every client's features as Python lists
   (`cicids.py:369`, `toniot.py:149`, `nslkdd.py:123`) because upstream
   `read_user_data` expects a dict of lists. Roughly 1 GB per process on
   CICIDS. Cap parallelism at 4 on 36 GiB.
7. zsh does not word-split unquoted parameter expansions. `$FLAGS` reaches
   argparse as one token. Inline the flags.
8. `timeout` is not on macOS by default. Runs prefixed with it exit 127.
9. Teams are reset every outer round: `set_team_parameter` (`serverPerMFL.py:357`)
   copies `old_global_model` into every team model. Team state does not persist
   across `t`, so `--num_team_iters` is the only place team-level learning
   accumulates.
