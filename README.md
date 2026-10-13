# ICD-10 Classifier

Fine-tunes [BioClinical ModernBERT](https://huggingface.co/thomas-sounack/BioClinical-ModernBERT-base)
to predict ICD-10 diagnosis codes from MIMIC-IV discharge summaries. Each hospital admission has
many codes, so this is multi-label classification: one sigmoid output per code, trained with
binary cross-entropy.

- **Input:** the full discharge summary for an admission, with whitespace normalized and nothing else changed.
- **Output:** a probability for each code in the label vocabulary.
- **Label settings:** `top50` uses the 50 most frequent codes, for fast iteration and comparison with the
  literature. `full` uses every code seen in the training split, which is several thousand codes with a long tail.
- **Training:** QLoRA by default. The encoder is loaded in 4-bit and frozen, and LoRA adapters plus the
  classification head are trained. This fits long notes (4096 tokens) on a single consumer GPU.
  
📊 **[Project presentation](https://docs.google.com/presentation/d/1968Msu17MojJsgJjZordTt2Z4Cgu9Yd0L1mWVu8n0lo/edit?usp=sharing)**: slides explaining the project and experiments in depth.

## Data handling

MIMIC-IV is credentialed data under the PhysioNet Data Use Agreement. Its responsible-use policy
doesn't allow sending the data to third-party online services, and that includes AI assistants.

- **No raw text or patient rows on screen or online.** Don't print raw note text or patient-level rows,
  and never paste them into an online tool.
- **Nothing derived from MIMIC gets committed.** That covers processed data, splits, tokenized caches,
  logits, checkpoints and models. All of it lives under `data/` and `outputs/`, which are gitignored.
- **The raw MIMIC files are never modified.** Every step writes new files.
- **The example prediction contains real note text.** After training on MIMIC, `train.py` writes one test
  note with its true and predicted codes to `outputs/.../example_prediction.txt`, and doesn't print it.
  The file contains real note text, so treat it like the raw data. The note is printed to the terminal
  only for runs on the synthetic fixture.
- **Use the synthetic fixture for tests and examples.** It is in `tests/fixtures/synthetic_mimic/`,
  is entirely made up, and is safe to print and share.

## Setup (Windows, PowerShell)

Requires Python 3.12+ and an NVIDIA GPU with CUDA for QLoRA runs. The `smoke` run and the tests also
work on CPU.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128  # RTX 50-series needs cu128
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pip install peft bitsandbytes  # needed for QLoRA
```

Then open `src/icd10/config.py` and set `paths.mimic_dir` in `BASE` to your MIMIC-IV folder, the one
that contains `note\` and `hosp\`. The code reads three files from it:

| File | Used for |
|---|---|
| `note/discharge.csv.gz` | Note text (`subject_id`, `hadm_id`, `note_seq`, `text`) |
| `hosp/diagnoses_icd.csv.gz` | Labels (`hadm_id`, `seq_num`, `icd_code`, `icd_version`) |
| `hosp/d_icd_diagnoses.csv.gz` | Code descriptions (for analysis only) |

Paths in the config can be absolute or relative to the repo root.

## Quick start

```powershell
$py = ".venv\Scripts\python"

# 1. Check that everything works on synthetic data
& $py -m pytest
& $py -m icd10.train smoke         # tiny random model on CPU, about 10 seconds
& $py -m icd10.train synthetic     # the real model and settings: checks GPU memory and speed

# 2. Train on MIMIC
& $py -m icd10.train top50_base

# 3. Watch training (in a second terminal)
.venv\Scripts\tensorboard --logdir outputs   # then open http://localhost:6006
```

Run the `synthetic` experiment before a long MIMIC run. It uses the same model, sequence length and
memory settings, so an out-of-memory error shows up in minutes rather than hours. Its metrics are
meaningless.

## Experiments

An experiment is a named set of settings in `src/icd10/config.py`. Run one with
`python -m icd10.train <name>`.

| Name | Model | Labels | Effective batch | Notes |
|---|---|---|---|---|
| `top50_base` | base | top 50 | 8 × 4 = 32 | Start here |
| `full_base` | base | all train codes | 8 × 4 = 32 | 1 epoch; reports head/medium/tail buckets |
| `top50_large` | large | top 50 | 2 × 8 = 16 | lr 1e-4 |
| `synthetic` | base | top 50 | 8 × 4 = 32 | BASE settings on synthetic data |
| `smoke` | tiny random | top 50 | 16 × 2 = 32 | CPU, no QLoRA, 512 tokens |

All MIMIC experiments use a max length of 4096 tokens, QLoRA (rank 16), lr 2e-4 unless noted, and
2 epochs unless noted.

### Changing settings

All settings are in `src/icd10/config.py`. `BASE` holds every setting with a comment explaining it,
grouped into `paths`, `data`, `model`, `qlora` and `train`. Each experiment lists only what it
changes from `BASE`. To change a run, edit the values there; there are no command-line overrides.
To add an experiment:

```python
TOP50_8K = deep_merge(BASE, {"run_name": "top50-8k", "data": {"max_length": 8192}})

EXPERIMENTS = {
    ...
    "top50_8k": TOP50_8K,
}
```

Settings you're most likely to change:

| Setting | What it does |
|---|---|
| `data.max_length` | Tokens kept per note (model max 8192). Longer notes are truncated. |
| `data.label_setting` | `top50` or `full` |
| `model.checkpoint` | `thomas-sounack/BioClinical-ModernBERT-base` or `-large` |
| `qlora.enabled` | `False` for full fine-tuning without quantization (needs much more memory) |
| `train.batch_size`, `train.grad_accum` | Lower `batch_size` and raise `grad_accum` to save memory with the same effective batch |
| `train.precision` | `auto` picks bf16 if the GPU supports it, otherwise fp16 (fp32 on CPU) |

## How it works

`train.py` runs the whole pipeline. It builds and caches each step's output the first time and
reuses it afterwards.

1. **Load and join** (`data/dataset.py`). Reads discharge notes and diagnoses and keeps only ICD-10
   codes (`icd_version == 10`). If an admission has several notes, they are joined in `note_seq` order.
   Notes are joined to codes on `hadm_id`, and admissions with no note or no ICD-10 codes are dropped.
   Whitespace is normalized and de-identification placeholders (`___`) are kept. Codes stay in MIMIC's
   dot-less format (`I5023`, not `I50.23`). Saved to `data/processed/mimiciv_icd10.parquet`.
2. **Split** (`data/splits.py`). Patients (`subject_id`), not admissions, are randomly assigned to
   train/val/test (80/10/10) with the config seed, so no patient appears in two splits. The ID lists are
   saved to `data/splits/subject_seed<seed>.csv`.
3. **Label vocabulary** (`data/labels.py`). Codes are counted on the train split only and ordered by
   frequency. `top50` keeps the first 50, `full` keeps them all. Codes outside the vocabulary are
   dropped, and admissions left with no codes are removed (standard for top-50 benchmarks).
4. **Tokenize** (`data/prepare.py`). Every note is tokenized once and truncated to `max_length`. The cache
   lives in `data/tokenized/<text hash>/<tokenizer>_len<max_length>/`, so changing the tokenizer or the
   length creates a new cache, and one cache serves both label settings.
5. **Train** (`train.py`). The Hugging Face `Trainer` runs with `AutoModelForSequenceClassification`
   (`problem_type="multi_label_classification"`), SDPA attention, gradient checkpointing and
   length-grouped batches. Validation runs after each epoch, and the epoch with the best val micro F1
   (at the best threshold) is kept.
6. **Predict and evaluate** (`evaluate.py`, `metrics.py`). Raw logits for val and test are saved, the
   decision threshold is tuned on val, and test metrics are computed with that threshold.

### Caches to know about

The cached files are reused as long as they exist, and some setting changes don't invalidate them:

- **The processed parquet** doesn't depend on `mimic_dir`. If you point `mimic_dir` somewhere else, or
  change the loading code in `dataset.py`, delete `data/processed/`. The `synthetic` and `smoke`
  experiments use `data/smoke/` instead, so they never mix with real data.
- **The split file** is keyed only by seed. If you change `val_fraction` or `test_fraction`, delete
  `data/splits/` or change the seed. Changing the seed creates a new split.
- **The tokenized cache** is rebuilt whenever the text, the tokenizer or `max_length` changes.

## Outputs

Each run creates `outputs/<run_name>/<timestamp>/`:

| File | Contents |
|---|---|
| `config.yaml` | Full settings used for the run |
| `run_metadata.json` | Seed, precision, torch/transformers/peft/CUDA versions, GPU, start time |
| `label_vocab.json` | Label setting, codes (column order) and their train counts |
| `{val,test}_logits.npy` | Raw logits, float32 (admissions × labels) |
| `{val,test}_labels.npy` | Gold multi-hot labels, uint8, same shape |
| `{val,test}_hadm_ids.npy` | Admission ID for each row |
| `metrics.json` | Results (see below) |
| `example_prediction.txt` | One test note with true and predicted codes. **Contains real note text on MIMIC runs.** |
| `tensorboard/` | Training curves, final test metrics, config and metadata |
| `checkpoints/` | Per-epoch Trainer checkpoints (latest 2) |
| `model/` | Final model (the best epoch) |

## Evaluation

Metrics are computed from the saved logits, so you can re-evaluate a run (for example after changing
`metrics.py`) without running the model again:

```powershell
& $py -m icd10.evaluate outputs\top50-base\<timestamp>
```

`metrics.json` contains:

- **`global_threshold`:** the single threshold (searched 0.01–0.99) that maximizes micro F1 on **val**.
  The test set is never used to choose it.
- **`val`, `test`:** metrics at that threshold:
  - micro and macro F1
  - micro and macro AUC-ROC, with macro computed over labels that have positives in the split
  - precision@5, @8 and @15
- **`test_per_label_thresholds`:** test metrics with one threshold per label, each tuned on val.
- **`test_buckets`** (`full` only): metrics split by how often each code appears in train: tail
  (fewer than 10 admissions), medium (10–99) and head (100+).

## TensorBoard

```powershell
.venv\Scripts\tensorboard --logdir outputs
```

Open http://localhost:6006. The **Scalars** tab shows training loss, learning rate, val metrics per
epoch and the final `test/*` metrics. The **Text** tab shows each run's config and metadata. To
look at one experiment only, use a narrower log dir, e.g. `--logdir outputs\top50-base`. No note text
is logged to TensorBoard.

## Tests and synthetic data

```powershell
& $py -m pytest
```

The tests run on synthetic data only and cover data loading, label encoding, split leakage (no shared
`subject_id` across splits), config, and metric correctness.

The synthetic fixture mirrors MIMIC's layout, but it uses plain `.csv` files so they are easy to read.
The loader accepts either format. To regenerate it:

```powershell
& $py scripts\make_synthetic_mimic.py --seed 0 --n-patients 200
```

## Project layout

```
src/icd10/
  config.py          all settings: BASE + experiments
  train.py           training entrypoint: python -m icd10.train <experiment>
  evaluate.py        metrics from a run's saved logits: python -m icd10.evaluate <run_dir>
  metrics.py         F1, AUC, precision@k, threshold tuning, frequency buckets
  data/
    dataset.py       load notes + ICD-10 codes, join on hadm_id
    splits.py        patient-level train/val/test splits
    labels.py        label vocabulary from train, multi-hot encoding
    prepare.py       config -> tokenized per-split datasets; multi-label collator
scripts/
  make_synthetic_mimic.py   generates tests/fixtures/synthetic_mimic
tests/               pytest suite (synthetic data only)
data/                gitignored: processed data, splits, tokenized caches
outputs/             gitignored: one directory per run
```
