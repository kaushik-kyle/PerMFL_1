# Code flow

High level. No code is reproduced here; every step names the file and line.
Concepts and terms are in [concepts.md](concepts.md), parameters in
[parameters.md](parameters.md).

## 1. Branches

| Branch | What it holds |
|---|---|
| `main` | upstream, byte-identical to `origin/main`. Never modified |
| `dev` | the two minimum fixes only. A sibling of `dev_cicids`, not an ancestor |
| `dev_cicids` | **the working branch.** Everything in section 4 |
| `wip/*` | local only, superseded, kept for provenance |

`git diff main dev_cicids -- '*.py'` is the complete change: 1,442 insertions,
24 deletions, 14 files.

## 2. What runs, in order

```
main.py                        parse flags, seed, build model, dispatch
   |
   +-- utils/<dataset>.py      load and partition data into clients and teams
   |
   +-- serverPerMFL.py         construct server, teams, client objects
          |
          +-- train()          the three nested loops below
                 |
                 +-- save_results()   write results/<...>.h5
```

| Step | File | Line |
|---|---|---|
| Flags and dispatch | `main.py` | 300+ |
| Dataset loaders | `utils/cicids.py`, `nslkdd.py`, `toniot.py`, `model_utils.py` | — |
| Server construction | `FLAlgorithms/servers/serverPerMFL.py` | 18 |
| Training loop | `FLAlgorithms/servers/serverPerMFL.py` | 823 |
| Results written | `FLAlgorithms/servers/serverPerMFL.py` | 447 |

## 3. The training loop

Three nested loops. Sizes are `--num_global_iters`, `--num_team_iters`,
`--local_iters`.

```
for each global round T:                       serverPerMFL.py:825
    save current global model as x_old         :353
    pick teams to participate                  :828
    for each team:
        reset team model to the global model   :357
        for each team round K:                 :841
            pick clients from this team        :849
            for each client:
                for each local step L:         userPerMFL.py:39
                    one gradient step toward
                    lower loss and toward
                    the team model             fedoptimizer.py:89
            average the clients' models        :328
            update the team model              :407   <-- the change is here
    average the team models                    :424
    update the global model                    :441
    evaluate both tiers                        :669
    optionally re-form teams                   :257
```

Two facts that follow from the structure:

- Team models are reset to the global model at the start of every global round
  (`:357`), so team learning does not accumulate across global rounds. Only
  `--num_team_iters` gives it room.
- Communication per global round is `2 × K × clients × teams_sampled`. The
  number of teams cancels; only K and the client count move it.

## 4. The three update rules

Each tier moves toward the tier above it. Direction of pull, not the algebra.

| Tier | Moves toward | Controlled by | File and line |
|---|---|---|---|
| Client `θ` | lower loss, and its team model `w` | `α` learning rate, `λ` pull | `fedoptimizer.py:89` |
| Team `w` | the global model `x`, and its members' average | `η` step, `γ` pull to global, **`λ_team` pull to members** | `serverPerMFL.py:407` |
| Global `x` | the average of team models | `β` step, `γ` | `serverPerMFL.py:441` |

## 5. PerMFL against Split-λ

The only difference is which number multiplies the members' average in the team
update.

```
PerMFL                                Split-λ
------                                -------
client update uses  λ                 client update uses  λ
team update uses    λ                 team update uses    λ_team
                    ^                                     ^
              same number                          set independently
```

| | Client coefficient | Team coefficient | Flag |
|---|---|---|---|
| PerMFL | λ | λ | none, defaults to λ |
| Split-λ | λ | λ_team | `--lamda_team` |

Set at [serverPerMFL.py:95](../FLAlgorithms/servers/serverPerMFL.py#L95), used at
[:413](../FLAlgorithms/servers/serverPerMFL.py#L413).

Why it matters is in [concepts.md](concepts.md) section 4.

## 6. Reading order

| # | File | Look for |
|---|---|---|
| 1 | `serverPerMFL.py:823` | the loop nesting in section 3 |
| 2 | `serverPerMFL.py:407, 424, 441` | the team, average and global updates |
| 3 | `userPerMFL.py:37` | the client loop |
| 4 | `optimizers/fedoptimizer.py:71` | the proximal step |
| 5 | `serverPerMFL.py:669` | which metrics reach which array |
| 6 | `FLAlgorithms/metrics.py` | macro F1, recall, FPR. 73 lines |
| 7 | `utils/cicids.py:174` | partition, split and thinning order |
| 8 | `FLAlgorithms/clustering/team_former.py` | team re-formation |

Skip `utils/model_utils.py` except `read_user_data`. Skip every `servers/*` file
other than `serverPerMFL.py`.

## 7. What this project changed

| File | Change |
|---|---|
| `users/userPerMFL.py:12` | keyword arguments in the parent call. Positional binding sent `--lamda` to the wrong attribute |
| `main.py:20` | `--seed` parameterised; the seed had been hardcoded |
| `serverPerMFL.py:95, 413` | `λ_team` separated from `λ` |
| `serverPerMFL.py:424` | optional sample-proportional team averaging |
| `serverPerMFL.py:186, 207` | pooled macro F1, per-client report |
| `serverPerMFL.py:257` | team re-formation hook |
| `serverPerMFL.py:447` | confusion matrices and run configuration persisted |
| `FLAlgorithms/metrics.py` | new. Macro F1, recall, FPR |
| `FLAlgorithms/clustering/team_former.py` | new. Team formation and drift trigger |
| `utils/cicids.py`, `nslkdd.py`, `toniot.py` | new. Three intrusion dataset loaders |
| `users/userbase.py`, `userpFedMebase.py` | vectorised confusion matrix |
| `servers/serverpFedMeoriginalbase.py` | macro F1 for pFedMe |
| `trainmodel/models.py` | optional second hidden layer; EMNIST output size |
| `utils/model_utils.py` | dispatch for the four added datasets |

Of the seven baseline algorithms only pFedMe reports macro F1. The rest are
untouched and report accuracy only.

## 8. Traps

| # | Trap |
|---|---|
| 1 | `--exp_start N --times N` runs nothing and exits 0. The loop is `while i < times` seeded from `exp_start` |
| 2 | `--K` is not the paper's K. The paper's K is `--num_team_iters` |
| 3 | `--model_name dnn` has no branch for `Emnist10`; `model` is left unbound |
| 4 | `γ > 2λ` is a convergence condition. Violating it raises nothing |
| 5 | Benign thinning must apply to the training split only |
| 6 | Loaders materialise features as Python lists, roughly 1 GB per process on CICIDS. Cap parallelism at 4 |
| 7 | zsh does not word-split unquoted variables; flags in a shell variable reach argparse as one token |
| 8 | `timeout` is not present on macOS by default |
| 9 | Output filenames omit T, K and L, so runs differing only in horizon overwrite each other |
| 10 | `--algorithm Hier-Local-QSGD` passes argparse and matches no dispatch branch |
| 11 | `--dataset Movielens` is a valid choice with no loader |
| 12 | Team models reset to the global model every global round |

Full defect list with status in [defects.md](defects.md).
