# Setup

Everything runs on CPU on the Mac. There is no GPU path.

## Environment

The paper specifies no package versions. Appendix D.1 gives hardware only. The
pins below are the mid-2024 stack contemporary with the July 2024 submission.

```bash
python3 -m venv .venv
```

```bash
.venv/bin/python -m pip install torch==2.3.1 torchvision==0.18.1 numpy==1.26.4 pandas==2.2.2 h5py==3.11.0 tqdm==4.66.4 certifi
```

Pinning back is necessary, not cosmetic. `FLAlgorithms/optimizers/fedoptimizer.py:24`
uses a positional-alpha overload of `Tensor.add_` that current torch rejects. It
is reached only by Per-FedAvg, but that baseline fails outright on torch 2.12.

Verify:

```bash
.venv/bin/python -c "import torch,torchvision,numpy;print(torch.__version__,torchvision.__version__,numpy.__version__)"
```

Expect `2.3.1 0.18.1 1.26.4`. CUDA is false, MPS is true, and the code uses
neither, so every run is CPU.

## Certificates

The python.org build ships without root certificates, so torchvision downloads
fail with `CERTIFICATE_VERIFY_FAILED`. Export the certifi bundle before any run
that downloads data:

```bash
export SSL_CERT_FILE="$(.venv/bin/python -m certifi)"
```

## Figures

Figures use a separate environment so the pinned reproduction venv stays
untouched.

```bash
python3 -m venv .venv-figs && .venv-figs/bin/pip install matplotlib h5py numpy
```

```bash
.venv-figs/bin/python tools/make_figures.py
```

```bash
.venv-figs/bin/python tools/make_diagrams.py
```

## Running batches

`tools/run.sh` executes nothing without an explicit batch name.

```bash
tools/run.sh list
```

```bash
tools/run.sh B1 --dry
```

Every run writes `logs/<batch>/<exp>_<label>.log` with the full command, git SHA
and dirty count, host, OS, Python version, environment variables, UTC start and
end, elapsed seconds and exit code, plus a row in `logs/<batch>/manifest.tsv`.

## Two traps that cost the most time

- `--exp_start N --times N` runs nothing and exits 0. The loop is
  `while i < times` seeded from `exp_start`, so use `--times N+1`.
- zsh does not word-split unquoted parameter expansions, so flags held in a
  shell variable reach argparse as a single token. Inline them.

Full list in [code-flow.md](code-flow.md) section 8.
