# PerMFL

Personalized multi-tier federated learning. Fork of `sourasb05/PerMFL_1`
(arXiv:2407.14251), extended with an EMNIST-10 digits loader, a CICIDS2017
loader, and update-derived team formation.

## Architecture

Three tiers, each holding a complete model:

| tier | symbol | quantity | reported as |
|---|---|---|---|
| device | theta | one per client | PM |
| team | w | `--num_teams` | not reported |
| global | x | one | GM |

Each global round runs `--num_team_iters` team rounds; each team round runs
`--local_iters` local minibatch steps per device. Local iterations are
gradient steps, not epochs.

## Requirements

Python 3.12. Versions below are contemporary with the July 2024 submission;
`FLAlgorithms/optimizers/fedoptimizer.py` uses an `add_` overload removed in
later torch releases.

```
python3 -m venv .venv
```

```
.venv/bin/python -m pip install torch==2.3.1 torchvision==0.18.1 numpy==1.26.4 pandas==2.2.2 h5py==3.11.0 tqdm==4.66.4 certifi
```

Dataset downloads require a certificate bundle on macOS:

```
export SSL_CERT_FILE="$(.venv/bin/python -m certifi)"
```

## Datasets

| `--dataset` | source | classes | notes |
|---|---|---|---|
| `Synthetic` | generated | 10 | no download |
| `Mnist`, `FMnist`, `Cifar10`, `Cifar100` | torchvision | 10 / 10 / 10 / 100 | |
| `Emnist` | torchvision, `byclass` | 62 | upstream path |
| `Emnist10` | torchvision, `digits` | 10 | added |
| `Cicids` | local CSV | 8 | added; see below |

CICIDS2017 requires the `TrafficLabelling` variant, which carries the
`Timestamp` and `Flow ID` columns. `MachineLearningCVE` carries neither and
cannot support a temporal split. Set the directory before running:

```
export CICIDS_DIR=/path/to/TrafficLabelling
```

Optional: `CICIDS_BENIGN_RATIO` (default `3.0`), `CICIDS_MAX_PER_CLIENT`
(default `8000`).

## Running

Client count must divide team count. Team size must be below the class count
on `Emnist10`, since label sets are assigned on a cycle of period 10 and teams
of ten or more consecutive clients become statistically identical.

PerMFL on EMNIST-10:

```
OMP_NUM_THREADS=1 .venv/bin/python main.py --algorithm PerMFL --dataset Emnist10 --model_name mclr --lamda 0.5 --gamma 1.5 --beta 0.6 --alpha 0.01 --eta 0.03 --num_global_iters 100 --num_team_iters 10 --local_iters 20 --tot_users 20 --num_teams 4 --numusers 5 --p_teams 4 --num_labels 2 --group_division 0 --exp_start 0 --times 1
```

PerMFL on CICIDS2017 with derived teams:

```
OMP_NUM_THREADS=1 .venv/bin/python main.py --algorithm PerMFL --dataset Cicids --model_name mclr --lamda 0.5 --gamma 1.5 --beta 0.6 --alpha 0.01 --eta 0.03 --num_global_iters 100 --num_team_iters 10 --local_iters 20 --tot_users 20 --num_teams 5 --numusers 4 --p_teams 5 --num_labels 8 --group_division 3 --exp_start 0 --times 1
```

`OMP_NUM_THREADS=1` is advised for `mclr` and `dnn`, where matrix products are
too small for threading to pay; measured 22 to 39 percent faster with identical
results. Omit it for `cnn`.

## Team assignment

| `--group_division` | behaviour |
|---|---|
| `0` | sequential blocks; on `Cicids` this reproduces the day-domains |
| `1` | random; reseeds on wall-clock time, so teams differ per run |
| `2` | upstream; raises `TypeError` |
| `3` | derived from client updates during training |

Mode `3` starts from the random partition, since teams are assigned before any
client update exists, and reforms teams in the server's round loop. Related
flags: `--team_signal` (`residual`, default, or `grad`), `--recluster_from`,
`--eps_hi`, `--eps_lo`, `--pca_dim`.

## Algorithms

`PerMFL`, `FedAvg`, `hierarchical-FedAvg`, `PerAvg`, `pFedMe_original`,
`AL2GD`, `ditto`, `pFedBayes`.

`Hier-Local-QSGD` is accepted by the parser but the dispatch in `main.py`
tests `Hier_local_qsgd`, so selecting it raises `NameError`.

## Output

Results are written to
`results/<algorithm>/<dataset>/<model_name>/<analysis>/<p_teams>/<exp_no>_*.h5`.

```
.venv/bin/python -c "
import h5py, glob
for f in sorted(glob.glob('results/**/*.h5', recursive=True)):
    with h5py.File(f) as h:
        print(f)
        for k in ('per_test_accuracy', 'global_test_accuracy'):
            if k in h and len(h[k]):
                a = h[k][:]
                print('   %-22s final %6.2f%%  best %6.2f%%' % (k, a[-1]*100, a.max()*100))"
```

`per_test_accuracy` is PM, `global_test_accuracy` is GM. Both are pooled
across clients.

The filename encodes `exp_no`, lambda, gamma, beta, model and dataset, but not
`T`, client count or `local_iters`. Runs differing only in those fields
overwrite each other; assign a distinct `--exp_start` to each.

`--exp_start N` requires `--times` greater than `N`. The loop condition is
`while i < times` seeded from `exp_start`, so `--exp_start 1 --times 1`
performs no work and exits zero.

`--times N` produces N identical runs. `torch.manual_seed(0)` is fixed in
`main.py`, loader seeds are fixed, and `select_users` seeds on the round
index; under `--group_division 0` nothing varies between repeats.

`average_result` is commented out in `main.py` and its call passes two
arguments to a three-argument definition.

## Modifications from upstream

| file | change |
|---|---|
| `FLAlgorithms/users/userPerMFL.py` | keyword arguments to `User.__init__`; `--lamda` previously bound to `self.eta` and `--local_iters` to `self.lamda` |
| `utils/model_utils.py` | `random` seeded in `read_synthetic_data`; `read_EMnist10_data` added; `Emnist10` and `Cicids` dispatch; `Emnist10` reshape in `read_user_data` |
| `utils/cicids.py` | CICIDS2017 loader |
| `FLAlgorithms/clustering/` | derived team formation |
| `FLAlgorithms/servers/serverPerMFL.py` | reclustering hook in the round loop |
| `FLAlgorithms/trainmodel/models.py` | `cnn_emnist` accepts `output_dim`, default 62 |
| `main.py` | `Emnist10` and `Cicids` model branches; clustering flags |

## Licence

MIT, inherited from upstream.
