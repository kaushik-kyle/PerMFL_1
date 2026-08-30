# Run book, archived

Imported from the `wip/paper-config` branch. This is the working log kept while
the reproduction and the EMNIST-10 loader were built. It is **historical**. Where
it disagrees with the current documents, the current documents win.

Superseded sections and where the current version lives:

| In this file | Current location |
|---|---|
| Section 5b, 5c, results | [results.md](results.md) |
| Section 6, known defects | [defects.md](defects.md), with verified line numbers |
| Section 2, branches | [code-flow.md](code-flow.md) section 0 |
| Section 3, paper targets | [comparison.md](comparison.md) |

Still current and not repeated elsewhere: section 1 (environment and the
certificate workaround for dataset download), section 3's run matrix and the two
launch traps, and section 7's compute expectations.

---

# PerMFL run book

How to reproduce every run in this project by hand, what each one is for, and
what the paper reports for comparison.

Paper: PerMFL, arXiv:2407.14251. Upstream code: `sourasb05/PerMFL_1` at `85f46c6`.

All commands assume you are already in the project root.

---

## 1. Environment, one time only

The paper specifies no package versions. Appendix D.1 gives hardware only, a
DGX-A100, and the README lists six package names with no versions. The pins
below are the mid-2024 stack contemporary with the July 2024 submission.

Pinning back matters. `FLAlgorithms/optimizers/fedoptimizer.py:24` uses
`p.data.add_(-beta, d_p)`, a positional-alpha overload that current torch
rejects outright. It is reached only by Per-FedAvg, but that baseline dies on
torch 2.12.

```bash
python3 -m venv .venv
```

```bash
.venv/bin/python -m pip install --upgrade pip
```

```bash
.venv/bin/python -m pip install torch==2.3.1 torchvision==0.18.1 numpy==1.26.4 pandas==2.2.2 h5py==3.11.0 tqdm==4.66.4 certifi
```

Verify:

```bash
.venv/bin/python -c "import torch,torchvision,numpy,pandas,h5py,tqdm;print(torch.__version__,torchvision.__version__,numpy.__version__)"
```

Expect `2.3.1 0.18.1 1.26.4`. CUDA will be false and MPS true. The code never
uses MPS, so every run is CPU.

### Certificates, needed for any dataset download

The python.org Python 3.12 build ships without root certificates, so
torchvision's download fails with `CERTIFICATE_VERIFY_FAILED`. Export the
certifi bundle before any run that downloads data. Synthetic needs no download.

```bash
export SSL_CERT_FILE="$(.venv/bin/python -m certifi)"
```

Confirm the EMNIST host is reachable before starting a long run:

```bash
.venv/bin/python -c "import urllib.request as u;print(u.urlopen(u.Request('https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip',method='HEAD',headers={'User-Agent':'Mozilla/5.0'}),timeout=30).status)"
```

Expect `200`. The archive is 562 MB and downloads once into `./data/`.

---

## 2. Branches

| branch | contents | role |
|---|---|---|
| `main` | pristine upstream | untouched reference |
| `dev/emnist10-data` | EMNIST-10 loader, additive only | **baseline arm**, keeps the lamda bug |
| `dev/paper-config` | above + both fixes | **fixed arm** |

The loader lives on its own branch off `main` so the baseline arm can run
EMNIST-10 while still carrying the original defects. The only behavioural
difference between the two arms is the two fixes.

```bash
git log --oneline --graph --all
```

### What the two fixes are

**Fix 1, `FLAlgorithms/users/userPerMFL.py:12`.** `UserPerMFL` passed five
positional args into `userbase.User`'s six slots after `beta`, shifting
everything by one. `--lamda` landed in `self.eta` and `--local_iters` landed in
`self.lamda`, so `pFedMeOptimizer` received `lamda=20` rather than the CLI
value, and `self.local_epochs` was 0. `self.eta`, `self.gamma` and
`self.local_epochs` are assigned but never read, so lamda is the only
behavioural change.

**Fix 2, `utils/model_utils.py:927`.** `read_synthetic_data` seeded numpy but
not Python's `random` before the per-client 75/25 shuffle, so the split differed
on every process. EMNIST and MNIST already seed `random` up front, so this fix
only affects Synthetic.

Verify both:

```bash
PYTHONPATH="$PWD" .venv/bin/python -c "
import torch
from FLAlgorithms.trainmodel.models import DNN
from FLAlgorithms.users.userPerMFL import UserPerMFL
d=[(torch.randn(60),torch.tensor(0)) for _ in range(200)]
u=UserPerMFL(torch.device('cpu'),'f_00000',d,d,DNN(60,20,10),'dnn',124,0.01,0.99,5.0,20,'Synthetic')
print('optimizer lamda =',u.optimizer.param_groups[0]['lamda'],'(20 = unfixed, 5.0 = fixed)')
print('local_epochs    =',u.local_epochs,'(0 = unfixed, 20 = fixed)')"
```

---

## 3. Run matrix

### R0, literal default, sanity only

Every flag at its shipped default. This is `hierarchical-FedAvg` on `Synthetic`
with `dnn`, **not** PerMFL.

```bash
.venv/bin/python main.py
```

Runtime 5 min. Result 85.18% final global test accuracy over 100 rounds. The
paper never reports hierarchical-FedAvg on Synthetic, so this is a smoke test,
not a comparison. Nearest neighbour is h-SGD(GM) Synthetic DNN at 87.42 (+/-5.67).

### R1, PerMFL at code defaults, to record the divergence

```bash
.venv/bin/python main.py --algorithm PerMFL --dataset Synthetic --model_name dnn
```

Expect the global model to blow up. At defaults `beta=0.99` and `gamma=5.0`, so
the global update coefficient is `1 - beta*gamma = 1 - 4.95 = -3.95`. Theorem 2
requires `beta <= 1/(4*gamma) = 0.05`. Keep the log, this is evidence, not a
failed run.

### R2, baseline arm, EMNIST-10, MCLR

```bash
git checkout dev/emnist10-data
```

```bash
.venv/bin/python main.py --algorithm PerMFL --dataset Emnist10 --model_name mclr --lamda 0.5 --gamma 1.5 --beta 0.6 --alpha 0.01 --eta 0.03 --num_global_iters 100 --num_team_iters 10 --local_iters 20 --tot_users 40 --num_teams 4 --numusers 10 --p_teams 4 --num_labels 2 --group_division 0 --exp_start 0 --times 1
```

### R3, fixed arm, EMNIST-10, MCLR

Identical flags, other branch.

```bash
git checkout dev/paper-config
```

```bash
.venv/bin/python main.py --algorithm PerMFL --dataset Emnist10 --model_name mclr --lamda 0.5 --gamma 1.5 --beta 0.6 --alpha 0.01 --eta 0.03 --num_global_iters 100 --num_team_iters 10 --local_iters 20 --tot_users 40 --num_teams 4 --numusers 10 --p_teams 4 --num_labels 2 --group_division 0 --exp_start 1 --times 2
```

R2 against R3 isolates the lamda fix. Everything else is held constant, which is
why `--group_division 0` is used rather than the paper's random assignment.

### R4 and R5, the CNN half, expensive

Same flag sets with `--model_name cnn`. `cnn_emnist(output_dim=10)` is 2.36M
parameters against MCLR's 7,850, and a run is 800,000 local steps on CPU.
Budget 2 to 4 hours each. Start these overnight.

### Two traps when launching these by hand

**`--exp_start` gates the loop.** `main.py:25` is `while i < times` with
`i = --exp_start`, so `--exp_start 1 --times 1` runs nothing at all and exits
0 silently. To get a single run tagged `exp_no=1`, use `--exp_start 1 --times 2`.
The tag matters because the output filename is keyed on `exp_no`, so two arms
sharing a tag overwrite each other in `results/`.

**Do not put the flags in a shell variable.** zsh does not word-split unquoted
parameter expansions the way bash does, so `main.py $FLAGS` arrives as a single
argument and argparse rejects every flag at once. Every command here is inline
for that reason. In zsh use `${=FLAGS}` if you must.

### Where the hyperparameters come from

**Table 1, the headline performance table, states no hyperparameters at all.**
The paper gives a complete setting in exactly one place, the Table 2
team-formation ablation, `lambda=0.5, gamma=1.5, beta=0.6, alpha=0.01,
eta=0.03, T=400, K=10, L=20`. R2 to R5 use that setting at `T=100` for turnaround.
For a Table-2-faithful run add `--num_global_iters 400`, which is 4x the time.

Client counts need no decision, the paper and the code defaults already agree:
40 devices, 4 teams, 10 devices per team, 2 classes per device, full
participation of both teams and devices.

### Bounds check on those settings

Theorem 2 requires `beta <= 1/(4*gamma)`, `gamma > 2*lambda`, and stability
needs `1-beta*gamma` in [0,1].

| config | gamma | lambda | beta | `beta <= 1/(4g)` | `g > 2l` | `1-bg` |
|---|---|---|---|---|---|---|
| Table 2 setting | 1.5 | 0.5 | 0.6 | needs <= 0.167, **3.6x over** | yes | 0.10, stable |
| code defaults | 5.0 | 5.0 | 0.99 | needs <= 0.05, **20x over** | **no** | **-3.95, diverges** |

The paper's own settings sit outside its stated bound but remain stable. The
shipped defaults are neither.

---

## 4. Verifying a run

Progress, while it is running:

```bash
grep -c "Global test accurancy" nohup.out
```

Final numbers from the h5:

```bash
.venv/bin/python -c "
import h5py,glob
for f in sorted(glob.glob('results/**/*.h5',recursive=True)):
    with h5py.File(f) as h:
        ks=[k for k in ('global_test_accuracy','per_test_accuracy') if k in h and len(h[k])]
        print(f.split('/results/')[-1])
        for k in ks: print('   %-22s final %6.2f%%  best %6.2f%%'%(k,h[k][-1]*100,max(h[k])*100))"
```

`global_test_accuracy` is PerMFL(GM). `per_test_accuracy` is PerMFL(PM).
Both are pooled, `sum(correct)/sum(samples)` across clients, which is the
honest headline. PerMFL(PM) is measured over `participated_devices`, equal to
all 40 under the full participation used here.

### Repeats do not vary

`--times N` runs N times but **produces N identical results**. `torch.manual_seed(0)`
is hardcoded at `main.py:15`, the loaders seed `random.seed(5)` and
`np.random.seed(9)`, and `select_users` seeds on the round index. With
`--group_division 0` nothing is left to vary.

Check it yourself:

```bash
.venv/bin/python main.py --algorithm PerMFL --dataset Synthetic --model_name mclr --lamda 0.5 --gamma 1.5 --beta 0.6 --num_global_iters 3 --times 2 --group_division 0
```

Two identical accuracy traces means real seed variation needs
`torch.manual_seed` to be parameterised, which is a further change not yet made.
This matters because the paper reports mean and std over 10 runs and almost
every PerMFL entry in Table 1 carries a std of exactly `(+/-0.0)`.

Do not use `average_result`. It is commented out at `main.py:271` and also
broken, the call passes 2 args to a 3-arg def at `utils/result_utils.py:227`.

---

## 5. Paper targets, EMNIST-10, Table 1

| algorithm | MCLR | CNN |
|---|---|---|
| PerMFL(PM) | 96.49 | 98.79 |
| PerMFL(GM) | 91.68 | 93.12 |
| h-SGD(GM) | 92.33 | 96.03 |
| FedAvg(GM) | 91.60 | 92.73 |
| Per-FedAvg(PM) | 97.57 | 97.37 |
| pFedMe(PM) | 91.23 | 97.18 |
| DemLearn(PM) | 97.24 | 98.74 |

Caveat carried from above, Table 1's hyperparameters are unstated, so R2 to R5
are a best-faith reconstruction from the Table 2 setting rather than a
like-for-like reproduction.

---

## 5b. Results so far

EMNIST-10, MCLR, PerMFL, `lambda=0.5 gamma=1.5 beta=0.6 alpha=0.01 eta=0.03`,
T=100, K=10, L=20, 40 devices, 4 teams, `--group_division 0`, CPU.

| round | PM base | PM fixed | delta | GM base | GM fixed | delta |
|---|---|---|---|---|---|---|
| 0 | 93.83 | 97.39 | +3.55 | 8.97 | 9.54 | +0.57 |
| 25 | 95.36 | 97.64 | +2.28 | 34.44 | 63.87 | **+29.43** |
| 50 | 95.92 | 97.70 | +1.78 | 62.29 | 79.50 | +17.21 |
| 99 | **96.35** | **97.85** | +1.50 | **77.72** | **85.29** | +7.58 |

Paper Table 1 EMNIST-10 MCLR: PM 96.49, GM 91.68.

### T=400, the paper's Table 2 horizon

| round | PM base | PM fixed | delta | GM base | GM fixed | delta |
|---|---|---|---|---|---|---|
| 99 | 96.35 | 97.85 | +1.50 | 77.72 | 85.29 | +7.58 |
| 199 | 96.18 | 97.92 | +1.74 | 84.31 | 88.43 | +4.12 |
| 299 | 96.09 | 97.86 | +1.77 | 86.54 | 89.63 | +3.09 |
| 399 | **95.99** | **97.79** | +1.80 | **87.84** | **90.45** | +2.61 |

PM peak: base 96.45 at round 130, fixed 98.05 at round 191.
Runtime: baseline 1542 s, fixed 1623 s.

**The baseline arm's PM peak of 96.45 sits 0.04 points from the paper's 96.49.**
The baseline carries the lamda bug, which is what the released code does, so
this is the reproduction.

Two things to carry forward.

PM peaks and then declines in both arms, base 96.45 at r130 down to 95.99 at
r399, fixed 98.05 at r191 down to 97.79. The personal models overfit. The paper
reports one number per cell with no stated T, so whether it is a final or a
best value changes the comparison.

GM was still climbing at r399 in both arms, base at +0.0103 points per round
and fixed at +0.0051. The baseline would need roughly another 370 rounds to
reach 91.68 at that rate. Do not read the remaining GM gap as a defect.



Baseline runtime 381 s, fixed arm 335 s.

Reading. The baseline arm carries the lamda bug, which is what the published
code does, and it lands within 0.14 points of the paper's PM. That is the
reproduction. The fix raises PM by 1.50 and GM by 7.58, and roughly doubles
early GM convergence.

Caveat, GM had not converged at T=100 in either arm, both were still climbing
at round 99. The paper's only fully-specified setting uses T=400. Do not report
the GM gap against 91.68 from these runs.

Caveat, PM is not a 10-class number. Each device holds 2 of 10 classes and its
personal model is scored only on its own test slice, so pooling forty 2-class
problems does not recover a 10-class problem. PM 97.85 and GM 85.29 do not
measure the same task. Macro F1 with an explicit `labels=list(range(10))`
would separate them.

## 5c. Performance report

EMNIST-10, MCLR, PerMFL, fixed arm, T=100, `--group_division 0`, M3 Max
(10 performance + 4 efficiency cores, 36 GiB), CPU only.

| clients | teams | threads | wallclock | user | sys | sys/real |
|---|---|---|---|---|---|---|
| 40 | 4x10 | 14 (default) | 335 s | 614 s | 3682 s | 11.0x |
| 40 | 4x10 | 1 | **262 s** | 184 s | 76 s | 0.3x |
| 20 | 4x5 | 14 (default) | 239 s | 344 s | 2090 s | 8.7x |
| 20 | 4x5 | 1 | **145 s** | 104 s | 41 s | 0.3x |

Accuracy is unchanged by thread count, bit-identical in both pairs.

| run | PM final | PM peak | GM final |
|---|---|---|---|
| 40 clients | 97.85 | 97.89 | 85.29 |
| 20 clients | **98.34** | **98.38** | **85.79** |

Reading.

`OMP_NUM_THREADS=1` is worth 22% at 40 clients and 39% at 20, and it changes no
result. Torch's default of 14 spawns a thread per core for a 124x784 by 784x10
matmul, and they spend their time synchronising, which is what the 8 to 11x
sys/real ratio is. Setting it to 1 also leaves 13 cores free, so there is no
speed-against-headroom trade to make here.

Threading is only counterproductive because these matmuls are tiny. For the CNN,
where the 18432x128 layer is large enough to pay, test 10 against 14 rather than
assuming either. Ten keeps the parallel region on performance cores, avoiding
efficiency-core stragglers that the whole fork-join blocks on.

Halving clients does not halve time, 239 s against 335 s at the same thread
count. The step count does halve, 400k against 800k, but `read_EMnist10_data`
carries `if NUM_USERS <= 20: num_samples *= 2` from the MNIST loader, so each
client holds roughly twice the data and evaluation costs more per client.
Accuracy rises slightly for the same reason.

**Fastest useful config: 20 clients, 4 teams of 5, `OMP_NUM_THREADS=1`, 145 s
per 100-round run.** That is 2.3x the shipped 40-client default.

Twenty rather than ten because ten forces `--num_teams 2`, and comparing derived
team assignment against random needs more than two teams to vary.

```bash
OMP_NUM_THREADS=1 .venv/bin/python main.py --algorithm PerMFL --dataset Emnist10 --model_name mclr --lamda 0.5 --gamma 1.5 --beta 0.6 --alpha 0.01 --eta 0.03 --num_global_iters 100 --num_team_iters 10 --local_iters 20 --tot_users 20 --num_teams 4 --numusers 5 --p_teams 4 --num_labels 2 --group_division 0 --exp_start 0 --times 1
```

---

---

## 6. Known defects, for reference

Confirmed by reading and by direct execution.

| # | location | defect |
|---|---|---|
| 1 | `users/userPerMFL.py:12` | positional arg shift, `--lamda` never reaches the device tier. **fixed on `dev/paper-config`** |
| 2 | `model_utils.py:927` | Synthetic split unseeded, differs every process. **fixed on `dev/paper-config`** |
| 3 | `model_utils.py` x5 | `group_division==1` calls `random.seed(time.time())`, so teams differ every run and the clock seed leaks into `random.sample` team selection in `serverPerMFL.py:667` |
| 4 | `model_utils.py` x5 | `group_division==2` does `user_list = random.shuffle(user_list)`, which returns `None`, so the next line raises `TypeError`. That mode has never run |
| 5 | `model_utils.py:556` | `--dataset Emnist` uses `split='byclass'`, 62 classes, which is the paper's FEMNIST, not EMNIST-10. Appendix D.2.3 says digits |
| 6 | `model_utils.py:664` | EMNIST pre-split shuffle is commented out, so the first 25% taken as test is close to single-class |
| 7 | `model_utils.py:682` | EMNIST `group_division==0` hardcodes `i == 10 or i == 36`, giving 11/26/3 at 40 users and leaving team 4 empty |
| 8 | `model_utils.py:1263` | `read_user_data` tests `dataset == "EMNIST"` but the CLI value is `Emnist`, so EMNIST skips the NCHW reshape and the CNN path cannot run |
| 9 | `main.py:51` | no `dnn` branch for Emnist, `--dataset Emnist --model_name dnn` raises `NameError` |
| 10 | `main.py:271` | `average_result` commented out and arity-broken |
| 11 | repo-wide | no `weight_decay` anywhere, though Appendix D.3 describes MLR "with l2 regularization" |
| 12 | `main.py:305` | `--optimizer` is parsed and passed to every server, then never read |
| 16 | `main.py:115` vs `:315` | argparse accepts `Hier-Local-QSGD`, dispatch tests `Hier_local_qsgd`, so h-QSGD is unreachable and `server.train()` raises `NameError`. h-SGD is the paper's headline multi-tier baseline in Table 1 |
| 15 | `model_utils.py` all loaders | client count must divide team count. `--tot_users 10 --num_teams 4` gives `cl_per_grp = 2` and the increment overruns the 4-slot team list at user 8, raising `IndexError` |
| 14 | `serverPerMFL.py:305` | output filename is keyed on exp_no, lambda, gamma, beta, model and dataset but **not** on T, so rerunning a config at a different horizon silently overwrites the earlier result |
| 13 | `main.py:25` | `while i < times` with `i = --exp_start`, so `--exp_start N --times M` where `N >= M` runs nothing and exits 0 with no warning |

Defects 5 to 9 are why `Emnist10` was added as a separate dataset rather than
patching `Emnist`. The new loader is templated on `read_Mnist_data`, which is
correct on all of them.

---

## 7. CICIDS2017, compute expectations

Training compute goes **down**, not up. The step count is fixed by the schedule
and is dataset-independent, `T x teams x K x devices x L = 800,000`. Only
per-step cost changes, and it scales with model size.

| config | params | relative to Synthetic |
|---|---|---|
| CICIDS mclr `Mclr(78,12)` | 948 | 0.7x |
| Synthetic dnn `DNN(60,20,10)` | 1,430 | 1.0x |
| EMNIST-10 mclr `Mclr(784,10)` | 7,850 | 5.5x |
| CICIDS dnn `DNN(78,100,12)` | 9,112 | 6.4x |
| EMNIST-10 cnn `cnn_emnist(10)` | 2,362,874 | 1652x |

78 features against 784 pixels means CICIDS MCLR is the cheapest thing here.
Expect roughly R0's 5 minutes, call it 6 to 10 for PerMFL, which is slower than
hierarchical-FedAvg because `userPerMFL.train` deep-copies the team model on
every one of its 40,000 calls.

The real cost is not compute.

- **Preprocessing**, one time. Eight CSVs, about 2.83M flows, 1.1 GB. Read,
  clean `inf` and NaN in `Flow Bytes/s` and `Flow Packets/s`, dedupe, sort by
  timestamp, standardise on train only. Minutes, and cacheable to `.npy`.
- **Memory**. Every loader here materialises Python lists via `.tolist()`.
  2.83M rows x 78 floats that way is 15 to 20 GB. Subsample to roughly
  100k-250k total, which is EMNIST-10 scale anyway at 40 clients, or bypass
  `.tolist()` with a numpy path.
- **Temporal split.** Not optional. A random split leaks, 88.1% of test rows
  have a near-twin in train. The old project's entire headline reversed sign
  under a temporal split.
- **Macro F1.** Accuracy is uninformative at 73% BENIGN. Adding F1 means
  changing the server `evaluate` methods, and it must pass an explicit
  `labels=list(range(num_classes))` or personalised methods are flattered by
  being graded only on the classes each client holds.

So the honest summary is that CICIDS is cheaper to train and more expensive to
build. Nothing about it needs a GPU pod.
