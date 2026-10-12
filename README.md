# ICD-10 Classifier

Fine-tunes BioClinical ModernBERT to predict ICD-10 diagnosis codes (multi-label) from MIMIC-IV
discharge summaries. See `claude.md` for the project rules, especially the data-handling rules:
MIMIC data, derived files and checkpoints never leave `data/` and `outputs/`, and never get committed.

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128  # RTX 50-series needs cu128
.venv\Scripts\python -m pip install -e ".[dev]"
$env:MIMIC_DIR = "D:\path\to\mimic-iv"   # folder containing note\ and hosp\
```

## Workflow

```powershell
$py = ".venv\Scripts\python"
& $py scripts/build_dataset.py  top50_base   # notes + ICD-10 codes -> data/processed
& $py scripts/compute_splits.py top50_base   # patient-level splits -> data/splits
& $py -m icd10.train top50_base
& $py -m icd10.evaluate outputs/top50-base/<timestamp>
tensorboard --logdir outputs
```

`train.py` builds the dataset, splits and tokenized cache if they don't exist yet, so the first
two scripts are optional. To use the Edin et al. (2023) split, set `data.split_file` to their
`mimiciv_icd10_split.feather`.

## Changing hyperparameters

All training settings live in `src/icd10/config.py`. `BASE` holds every setting with comments,
grouped into paths, data, model, train and loss. Each experiment (`TOP50_BASE`, `FULL_BASE`,
`TOP50_LARGE`, `SMOKE`) lists only what it changes from `BASE`. Edit the values there and run the
experiment by name. To add an experiment, define it with `deep_merge(BASE, {...})` and add it to
`EXPERIMENTS`.

## Tests and smoke run (synthetic data only)

```powershell
& $py -m pytest
& $py -m icd10.train smoke       # tiny random model, about 10 seconds
& $py -m icd10.train synthetic   # real model and BASE settings on synthetic data: checks memory and speed
```
