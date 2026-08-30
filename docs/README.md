# Documentation

| File | Contents |
|---|---|
| [code-flow.md](code-flow.md) | branch layout, execution path, the six update equations with file:line, read order, what this project changed, twelve traps |
| [parameters.md](parameters.md) | every CLI flag split into live, dead and ours; every environment variable per loader; internal attribute names that mislead |
| [modes.md](modes.md) | every switch and its values: algorithms, datasets, the model support matrix, team assignment, re-formation, coupling, partitioning, evaluation, output routing |
| [results.md](results.md) | every measurement in `results/`, 297 runs, including the regression and the coverage gaps |
| [defects.md](defects.md) | twenty-one defects in the released implementation, each with a verified current file:line and a fixed / worked-around / open status |
| [run-book.md](run-book.md) | archived working log from `wip/paper-config`: environment setup, the certificate workaround, the run matrix, compute expectations |
| [run-queue.md](run-queue.md) | archived experiment queue: the local-steps sweep with wall-clock, the team-size constraint analysis, four resolved blockers |
| [backlog.md](backlog.md) | runs not yet done, in four tiers, with what each one buys and a rough cost |
| [structure-mapping.md](structure-mapping.md) | the draft's eight chapters against the handbook's six, with the mapping and a recommendation |
| [comparison.md](comparison.md) | the paper's Tables 1, 2 and 3 verbatim, the paper's stated setup, and our run inventory beside them |

## Orientation

The subject is PerMFL (arXiv:2407.14251), a three-tier personalised federated
learning method: device parameters `theta`, team parameters `w`, global
parameters `x`, coupled by squared Euclidean penalties.

The contribution is `--lamda_team`. Upstream, a single `--lamda` serves as both
the device-to-team proximal coefficient in the device update and the team's
weight on its members' average in the team update. The convergence condition
`gamma > 2 lamda` therefore caps member influence below 50 per cent, and at the
published defaults it is 3.3 per cent, which leaves the team tier inert.
Separating the two roles is the change under test.

Start with [code-flow.md](code-flow.md) section 2 for the equations, then
[results.md](results.md) sections 2, 4 and 6b. Section 6b is the strongest
result: the comparison at the paper's own stated configuration.

## Reproducing a run

Commands are in the repository `README.md`. Two things that will otherwise cost
time: `--exp_start N --times N` runs nothing and exits 0, and zsh does not
word-split unquoted parameter expansions, so flags held in a shell variable
reach argparse as one token. Both are in [code-flow.md](code-flow.md) traps.
