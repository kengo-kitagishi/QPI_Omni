# environment/ - the analysis environment (Windows, NVIDIA GPU)

One pinned conda environment, `omnipose`, exported from the analysis PC on 2026-09-14:
Python 3.10.18, PyTorch 2.5.1 + CUDA 11.8 (conda `pytorch` / `nvidia` channels), omnipose 1.0.6,
numpy 1.25.2, scipy 1.10.1, scikit-image 0.25.2, pandas 2.3.2, tifffile 2023.4.12.
Omnipose masks depend on the torch / omnipose build, so every machine that segments a dataset
must use these exact pins (and the GPU: CPU inference gives different masks).

| file | role |
|---|---|
| `omnipose_win64_explicit.txt` | conda packages as exact URLs (`conda create --file`), win-64 only |
| `omnipose_pip_requirements.txt` | pip packages, pinned; installed into the env after the conda step |
| `omnipose_win64_full.yml` | full `conda env export` for reference (adapt from this on Linux / macOS) |
| `bootstrap_windows.ps1` | installs Miniconda per-user if no conda is found, creates the env, runs the check |
| `check_env.py` | smoke test: pinned imports, CUDA device, checkpoint hash, model load, one inference |

## Fresh Windows machine (no admin rights needed)

```powershell
git clone https://github.com/kengo-kitagishi/QPI_Omni.git
cd QPI_Omni
powershell -ExecutionPolicy Bypass -File environment\bootstrap_windows.ps1
```

What it does: finds an existing Anaconda / Miniconda, otherwise downloads the Miniconda installer
and installs it per-user into `%USERPROFILE%\miniconda3` with `/AddToPath=0 /RegisterPython=0`
(the CPython 3.11 that the Micro-Manager BeanShell scripts call stays untouched); then
`conda create -n omnipose --file environment\omnipose_win64_explicit.txt`,
`pip install -r environment\omnipose_pip_requirements.txt`, and `check_env.py` against the newest
checkpoint in `models/`. Re-running is safe: an existing conda or env is reused. Downloads are
about 3 GB (torch + CUDA runtime), so allow 15-30 min on the first run.

The analysis Python is then `%USERPROFILE%\miniconda3\envs\omnipose\python.exe`
(or `...\anaconda3\envs\omnipose\python.exe` where Anaconda already exists). Every script under
`scripts/` is run with that interpreter; `run_dataset_pipeline.py` passes it on to its subprocesses.

## Checking an existing machine

```powershell
<env>\python.exe environment\check_env.py --model models\omni_model_d20_2026_09_07_12_41_31.782047
```

Prints each pinned module with `ok` / `VERSION MISMATCH`, the CUDA device, the checkpoint hash
against `models/MODELS.json`, and the time of one inference. Exit code 0 means the machine can
segment and track.

## Updating the pins

Only when a package has to change (new torch, new omnipose). On the machine where the new
environment was validated:

```powershell
conda list -n omnipose --explicit > environment\omnipose_win64_explicit.txt
conda env export -n omnipose | Select-String -NotMatch "^prefix:" > environment\omnipose_win64_full.yml
```

then copy the `- pip:` block of the export into `omnipose_pip_requirements.txt` (one `name==version`
per line), re-run `check_env.py`, and note the change in `docs/PROTOCOL_TIMELAPSE.md`. Masks made
with different pins are not comparable with the 260517 master; re-segment the whole dataset or keep
the old environment for it.
