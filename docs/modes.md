# Modes

Every switch the repository exposes, what each value does, and which combinations
work. Parameter defaults are in [parameters.md](parameters.md).

## 1. Algorithms, `--algorithm`

Eight values. Only `PerMFL` was modified by this project; the rest are upstream.

| Value | Tiers | Server file | State |
|---|---|---|---|
| `PerMFL` | device, team, global | `servers/serverPerMFL.py` | the subject. Macro F1, recall, FPR added |
| `pFedMe_original` | device, global | `servers/serverpFedMeoriginal.py` | macro F1 added, otherwise upstream |
| `FedAvg` | global | `servers/serveravg.py` | upstream, accuracy only |
| `hierarchical-FedAvg` | team, global | `servers/hierarchicalserveravg.py` | upstream, accuracy only |
| `PerAvg` | device, global | `servers/serverperavg.py` | upstream, accuracy only |
| `ditto` | device, global | `servers/ditto_server.py` | upstream, accuracy only |
| `AL2GD` | device, team, global | `servers/serverL2GD.py` | upstream, accuracy only |
| `pFedBayes` | device, global | `servers/serverpFedbayes.py` | upstream, accuracy only |
| `Hier-Local-QSGD` | team, global | `servers/server_hqsgd.py` | **unreachable, see below** |

`Hier-Local-QSGD` cannot be selected. argparse accepts the string
`Hier-Local-QSGD` (`main.py:305`) but the dispatch tests
`algorithm == "Hier_local_qsgd"` (`main.py:143`). No branch matches, `server` is
never assigned, and `server.train()` raises at `main.py:291`.

Only `PerMFL` and `pFedMe_original` report macro F1. The others report accuracy,
which on an 81.7 per cent benign corpus is dominated by the true-negative cell.

## 2. Datasets, `--dataset`

| Value | Source | Classes | Features | Loader | Added by us |
|---|---|---|---|---|---|
| `Mnist` | torchvision | 10 | 784 | `model_utils.py` | no |
| `FMnist` | torchvision | 10 | 784 | `model_utils.py` | no |
| `Emnist` | torchvision, `byclass` | 62 | 784 | `model_utils.py` | no |
| `Emnist10` | torchvision, `digits` | 10 | 784 | `model_utils.py` | **yes** |
| `Cifar10` | torchvision | 10 | 3072 | `model_utils.py` | no |
| `Cifar100` | torchvision | 100 | 3072 | `model_utils.py` | no |
| `Synthetic` | generated | 10 | 60 | `model_utils.py` | no |
| `Movielens` | listed in argparse | - | - | **no loader exists** | no |
| `Cicids` | CICIDS2017 `TrafficLabelling` | 9 | 79 | `utils/cicids.py` | **yes** |
| `Nslkdd` | KDDTrain+ / KDDTest+ | 5 | 122 | `utils/nslkdd.py` | **yes** |
| `Toniot` | TON-IoT `train_test_network.csv` | 10 | 38 | `utils/toniot.py` | **yes** |

`Movielens` is accepted by argparse and has no loader. It fails at load time.

## 3. Models, `--model_name`

Support is a sparse matrix. `cnn` and `dnn` have no fallback branch, so an
unsupported pair leaves `model` undefined and raises `NameError` later, not at
the point of the mistake.

| Dataset | `mclr` | `cnn` | `dnn` | `pbnn` |
|---|---|---|---|---|
| `Mnist` | yes | yes | yes | yes |
| `FMnist` | yes | yes | **no** | fallback |
| `Emnist` | yes | yes | **no** | fallback |
| `Emnist10` | yes | yes | **NameError** | fallback |
| `Cifar10` | silent wrong model | yes | **no** | fallback |
| `Cifar100` | yes | yes | **no** | fallback |
| `Synthetic` | silent wrong model | **no** | yes | fallback |
| `Cicids` | yes | **no** | yes | fallback |
| `Nslkdd` | yes | **no** | yes | fallback |
| `Toniot` | yes | **no** | yes | fallback |

`mclr` has an `else` branch that silently builds `Mclr_Logistic(60, 10)` for any
unlisted dataset. For `Synthetic` that happens to be correct. For `Cifar10` it is
a 60-input model fed 3072 features, which fails at the first forward pass.

`VGG11`, `VGG13`, `VGG16`, `VGG19` are in the argparse choices and have a
construction branch, but no dataset branch feeds them, so they are unreachable.

`--model_name dnn` builds `DNN(input, 100, output, HIDDEN_LAYERS)`. The
`HIDDEN_LAYERS` environment variable defaults to 1, which is the shipped
behaviour. The paper's Section 4 and D.3 describe two hidden layers, so
`HIDDEN_LAYERS=2` matches the paper's text and `1` matches the released code.

## 4. Team assignment, `--group_division`

| Value | Behaviour | Where |
|---|---|---|
| `0` | sequential. Clients 0..k-1 to team 0, k..2k-1 to team 1 | loader |
| `1` | random shuffle, seeded by `*_TEAM_SEED` | loader |
| `2` | one team holding every client | `model_utils.py` only |
| `3` | **derived.** Loader is called with `1`, then teams are reformed from client updates | `serverPerMFL.py:142` |

The IDS loaders implement `0` and `else`. Value `2` falls through to the random
branch there, so on `Cicids`, `Nslkdd` and `Toniot` you get a single team by
setting `--num_teams 1`, not by `--group_division 2`.

Value `3` is the only one that engages `FLAlgorithms/clustering/team_former.py`.
Everything in section 5 is inert without it.

## 5. Team re-formation, active only under `--group_division 3`

| Flag | Values | Effect |
|---|---|---|
| `--team_signal` | `residual` | cluster on `theta - w`, PerMFL's own proximal residual |
| | `grad` | cluster on the raw update, as SCMoE-PFL does |
| `--recluster_every` | `1` | reform teams every global round |
| | `0` | form once at `--recluster_from`, then hold fixed |
| `--recluster_from` | int | first global round at which reclustering is considered |
| `--eps_hi` | float | CFMD-i gate. Recluster only if max pairwise client difference exceeds this |
| `--eps_lo` | float | CFMD-i gate. And only if mean pairwise difference stays below this |
| `--pca_dim` | int | MCTC. PCA components taken before cosine similarity |

Formation is MCTC: L2-normalise each client signal, PCA to `--pca_dim`, cosine
similarity, seed centres from the least-similar pair, then hard equal-size
assignment. `team_former.agreement` reports adjusted Rand index against the
loader's own grouping.

## 6. Coupling, the contribution

| Flag | Values | Effect |
|---|---|---|
| `--lamda_team` | unset | equation 3 weights member average by `--lamda`. Upstream behaviour |
| | float | equation 3 weights member average by this instead, decoupled from equation 1 |
| `--weighted_agg` | `0` | equation 4 averages teams uniformly, as Algorithm 1 specifies |
| | `1` | equation 4 weights teams by sample count, as FedAvg does |

Member influence into the team model is `lamda_team / gamma`. Upstream this is
`lamda / gamma`, and the convergence condition `gamma > 2 lamda` caps it below
0.5. Decoupling removes that cap. Stability still requires
`eta * (lamda_team + gamma) < 1`.

## 7. Partitioning, per loader, set by environment variable

### CICIDS2017, `CICIDS_PARTITION`

| Value | Basis | Literature |
|---|---|---|
| `domain` | one client group per capture day | CFMD-i Scenario 1 |
| `domainmix` | as `domain`, but `CICIDS_DOMAIN_MIX` of each client's rows come from its own day and the rest from any day | our variant, graded from domain to IID |
| `labelskew` | `CICIDS_CLASSES_PER_CLIENT` classes per client | SOH-FL Label1/3/6 ladder |
| `dirichlet` | class proportions drawn from Dir(`CICIDS_DIRICHLET_ALPHA`) | standard |

### NSL-KDD, `NSLKDD_PARTITION`

`domain` groups clients by attack family, DoS, Probe, R2L, U2R, with normal
spread across all. `dirichlet` as above. The train and test split is the
canonical KDDTrain+ / KDDTest+ and is never resampled.

### TON-IoT, `TONIOT_PARTITION`

`domain` by attack type. `victimip` groups by destination host, filtered to hosts
whose label entropy exceeds `TONIOT_MIN_ENTROPY`, following DP-FL. `dirichlet`
as above.

## 8. Evaluation modes

| Switch | Values | Effect |
|---|---|---|
| `CICIDS_SPLIT` | `temporal` | split each class on its own timeline. No leakage across the boundary |
| | `random` | stratified random split |
| `CICIDS_CROSS_TEST`, `TONIOT_CROSS_TEST` | `0` | clients evaluated on their own class set |
| | `1` | clients evaluated on classes absent from their training set, the zero-day condition, SOH-FL Table II precedent |
| `CICIDS_BENIGN_FRACTION` | float | benign thinning, applied to the training split only |

Cross-test imposes a structural ceiling. With `CICIDS_CLIENTS_PER_RARE` clients
holding each attack class out of `--tot_users`, macro F1 cannot exceed the mean
over classes of `2(h/N) / (1 + h/N)`.

## 9. Output routing

`--analysis` does not change the algorithm. It is one path component in
`results/<algorithm>/<dataset>/<model>/<analysis>/<p_teams>/`. Two runs that
differ only in a flag not present in the filename overwrite each other. The
filename carries exp_no, lambda, gamma, beta, model and dataset, and nothing
else. See trap 9 in [code-flow.md](code-flow.md).
