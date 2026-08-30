# Parameter glossary

Every flag, environment variable and internal attribute. Execution order is in [code-flow.md](code-flow.md).

## CLI flags

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

## Environment variables

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

## Internal attributes whose names mislead

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

