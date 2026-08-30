# Defect catalogue

Sixteen defects in the released implementation. Originally recorded in the run
book on `wip/paper-config`; every location has been re-verified against
`dev_cicids` and the line numbers below are current, not the ones first written.

Status is one of: **fixed** by this project, **worked around**, or **open**,
meaning still present and not addressed because it does not affect the results
reported.

| # | Location | Defect | Status |
|---|---|---|---|
| 1 | `FLAlgorithms/users/userPerMFL.py:12` | positional argument shift in `super().__init__`. `--lamda` lands in `self.eta` and `--local_iters` in `self.lamda`, so `pFedMeOptimizer` receives lamda 20 instead of 0.5 | **fixed** |
| 2 | `utils/model_utils.py` | Synthetic split unseeded, differs every process | **fixed** |
| 3 | `utils/model_utils.py:356` | `current_time = time.time()` then `random.seed(current_time)`. Teams differ every run and the clock seed leaks into the `random.sample` team selection in `serverPerMFL.py` | open |
| 4 | `utils/model_utils.py:199,207` | `user_list = random.shuffle(user_list)` assigns `None`, so the next line raises `TypeError`. `--group_division 2` has never run on these loaders | open |
| 5 | `utils/model_utils.py:554` | `--dataset Emnist` uses `split='byclass'`, 62 classes, which is the paper's FEMNIST. Appendix D.2.3 specifies digits | **worked around** by adding `Emnist10` |
| 6 | `utils/model_utils.py` | the EMNIST pre-split shuffle is commented out, so the first 25 per cent taken as test is close to single-class | **worked around** |
| 7 | `utils/model_utils.py:685` | EMNIST `group_division == 0` hardcodes `i == 10 or i == 36`, giving an 11/26/3 split at 40 users and leaving the fourth team empty | **worked around** |
| 8 | `utils/model_utils.py:1434` | `read_user_data` tests `dataset == "EMNIST"` but the CLI value is `Emnist`, so EMNIST skips the NCHW reshape and the CNN path cannot run | **worked around** |
| 9 | `main.py:71` | no `dnn` branch for `Emnist` or `Emnist10`. `--model_name dnn` leaves `model` unbound and raises `NameError` later | open, documented in [modes.md](modes.md) |
| 10 | `main.py:14,299` | `average_result` import and call are both commented out, and the call is arity-broken | open |
| 11 | repo-wide | no `weight_decay` anywhere, though Appendix D.3 describes MLR "with l2 regularization" | open |
| 12 | `main.py` | `--optimizer` is parsed and passed to every server, then never read | open |
| 13 | `main.py:35` | `while i < times` with `i` seeded from `--exp_start`, so `--exp_start N --times M` with `N >= M` runs nothing and exits 0 with no warning | open, documented |
| 14 | `FLAlgorithms/servers/serverPerMFL.py:454` | the output filename is keyed on exp_no, lambda, gamma, beta, model and dataset but **not** on T, so rerunning a config at a different horizon silently overwrites the earlier result | open |
| 15 | `utils/model_utils.py`, all loaders | client count must divide team count. `--tot_users 10 --num_teams 4` gives 2 clients per team and the increment overruns the four-slot team list at user 8, raising `IndexError` | open |
| 16 | `main.py:305` against `main.py:143` | argparse accepts `Hier-Local-QSGD`, the dispatch tests `Hier_local_qsgd`. h-QSGD is unreachable and `server.train()` raises. This is the paper's headline multi-tier baseline in Table 1 | open |

## Additions found since

| # | Location | Defect | Status |
|---|---|---|---|
| 17 | `main.py:313` | `--dataset Movielens` is an argparse choice with no loader anywhere in `utils/` | open |
| 18 | `main.py:38-56` | `--model_name mclr` has an `else` branch building `Mclr_Logistic(60, 10)` for any unlisted dataset. On `Cifar10` this silently constructs a 60-input model for 3072 features | open |
| 19 | `main.py` argparse | `VGG11`, `VGG13`, `VGG16`, `VGG19` are valid choices with a construction branch, but no dataset branch feeds them | open |
| 20 | `main.py` argparse | `--selected_users` is defined and never passed to `main()`. `--numusers` is the live flag | open |
| 21 | `FLAlgorithms/servers/serverPerMFL.py:454,477` | neither `lamda_team` nor `weighted_agg` is written to the h5 or the filename. A PerMFL and a Split-λ run at the same lambda are indistinguishable in every recorded field | open |

| 22 | `main.py` defaults, `clustering/team_former.py:53` | `--eps_hi` defaults to 0.0 and `--eps_lo` to infinity, so the CFMD-i trigger `mx > eps_hi and mean < eps_lo` reduces to `mx > 0 and mean < inf`, which is true whenever any two clients differ. The adaptive gate is unconditionally open and reclustering fires every round. Ours, not upstream | open |

## Why 5 to 9 were worked around rather than patched

Defects 5, 6, 7 and 8 all sit on the `Emnist` path and interact. Patching them in
place would have changed the meaning of `--dataset Emnist`, which is the 62-class
FEMNIST the paper also uses. `Emnist10` was added as a separate dataset instead,
templated on `read_Mnist_data`, which is correct on all four. The original path
is left untouched so any comparison against the paper's FEMNIST column remains
possible.

## Defects that shaped the results

Defect 1 is the reason the reproduction is meaningful. The run that lands on the
published 96.49 carries the argument shift, which is evidence the defect was
present when the paper's numbers were produced.

Defect 13 cost a full queue of runs that exited 0 having done nothing.

Defect 14 means any two runs differing only in horizon collide. All horizon
comparisons in [results.md](results.md) use distinct `--exp_start` values for
this reason.

Defect 16 is why no h-QSGD baseline appears anywhere in [results.md](results.md).
It is the paper's main multi-tier comparator and it cannot be run without a
one-line edit.
