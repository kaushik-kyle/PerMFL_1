# Code flow

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

## 5. Traps

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
9. `save_results` writes neither `lamda_team` nor `weighted_agg` to the h5 or
   the filename (`serverPerMFL.py:452,477`). A shipped run and a decoupled run
   at the same `--lamda` are identical in every recorded field and, at the same
   `--exp_start`, overwrite each other. Provenance rests entirely on the
   exp_no convention: in the decoupling series, even is shipped, odd is
   decoupled at `--lamda_team 1.5`.
10. `--algorithm Hier-Local-QSGD` passes argparse and matches no dispatch
   branch. The strings differ (`main.py:305` against `main.py:143`), so `server`
   is unbound when `server.train()` runs.
11. `--dataset Movielens` is an argparse choice with no loader.
12. Teams are reset every outer round: `set_team_parameter` (`serverPerMFL.py:357`)
   copies `old_global_model` into every team model. Team state does not persist
   across `t`, so `--num_team_iters` is the only place team-level learning
   accumulates.


Parameter definitions are in [parameters.md](parameters.md). Every mode switch is in [modes.md](modes.md).
