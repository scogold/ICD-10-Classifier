"""Fine-tune a sequence classifier for multi-label ICD-10 coding.

Usage:
    python -m icd10.train top50_base

Experiments and their settings are defined in icd10/config.py.

Writes everything for the run to outputs/<run_name>/<timestamp>/ (see evaluate.py for the
file format) and evaluates test with a threshold tuned on val.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import peft
import torch
import transformers
import yaml
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

from icd10.config import EXPERIMENTS, REPO_ROOT, get_config, resolve_path
from icd10.data.dataset import resolve_mimic_dir
from icd10.data.labels import multi_hot
from icd10.data.prepare import MultiLabelCollator, prepare_data
from icd10.evaluate import evaluate_run
from icd10.metrics import compute_metrics, sigmoid, tune_global_threshold
from icd10.losses import LossFn, make_loss_fn

SYNTHETIC_MIMIC_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic_mimic"


class MultiLabelTrainer(Trainer):
    """Trainer with a configurable multi-label loss instead of the model's built-in BCE."""

    # Our loss is a per-batch mean, so the Trainer should divide it by the accumulation steps.
    loss_is_scaled_for_ga = False

    def __init__(self, *args, loss_fn: LossFn, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits.float(), labels.float())
        return (loss, outputs) if return_outputs else loss


TORCH_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def build_model(config: dict[str, Any], vocab: list[str], precision: str) -> torch.nn.Module:
    """Pretrained encoder plus a new linear head with one sigmoid output per ICD-10 code.

    With QLoRA the encoder is loaded in 4-bit and frozen; only LoRA adapters and the
    full-precision head layers are trained.
    """
    model_kwargs = {
        "num_labels": len(vocab),
        "problem_type": "multi_label_classification",
        "attn_implementation": "sdpa",  # flash_attention_2 is hard to install on Windows
        "id2label": dict(enumerate(vocab)),
        "label2id": {code: i for i, code in enumerate(vocab)},
    }
    checkpoint = config["model"]["checkpoint"]
    qlora = config["qlora"]
    if not qlora["enabled"]:
        return AutoModelForSequenceClassification.from_pretrained(checkpoint, **model_kwargs)
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA needs a CUDA GPU (bitsandbytes 4-bit); set qlora.enabled to False")

    compute_dtype = TORCH_DTYPES[precision]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora["quant_type"],
        bnb_4bit_use_double_quant=qlora["double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
        # 4-bit weights can't be trained, so the layers we train fully must stay unquantized.
        llm_int8_skip_modules=qlora["full_modules"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, quantization_config=quantization, dtype=compute_dtype, **model_kwargs
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config["train"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=qlora["r"],
        lora_alpha=qlora["alpha"],
        lora_dropout=qlora["dropout"],
        target_modules=qlora["target_modules"],
        modules_to_save=qlora["full_modules"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def resolve_precision(precision: str) -> str:
    """'auto' means bf16 if the GPU supports it, else fp16 on GPU, else fp32."""
    if precision != "auto":
        return precision
    if not torch.cuda.is_available():
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def build_training_args(config: dict[str, Any], run_dir: Path, precision: str) -> TrainingArguments:
    train = config["train"]
    return TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        seed=config["seed"],
        data_seed=config["seed"],
        learning_rate=train["lr"],
        per_device_train_batch_size=train["batch_size"],
        per_device_eval_batch_size=train["eval_batch_size"],
        gradient_accumulation_steps=train["grad_accum"],
        num_train_epochs=train["epochs"],
        warmup_steps=train["warmup_steps"],
        weight_decay=train["weight_decay"],
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        gradient_checkpointing=train["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        train_sampling_strategy="group_by_length" if train["group_by_length"] else "random",
        length_column_name="length",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=train["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model=train["metric_for_best_model"],
        logging_steps=train["logging_steps"],
        remove_unused_columns=False,  # the collator needs label_ids, which the model doesn't accept
        report_to=["tensorboard"],
    )


def eval_metrics(prediction: EvalPrediction) -> dict[str, float]:
    """Validation metrics during training: fixed 0.5 threshold plus the best global threshold."""
    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    probs = sigmoid(logits)
    labels = prediction.label_ids.astype(np.uint8)
    results = compute_metrics(labels, probs, threshold=0.5)
    tuned = compute_metrics(labels, probs, threshold=tune_global_threshold(labels, probs), ks=())
    results["micro_f1_tuned"] = tuned["micro_f1"]
    return results


def save_predictions(trainer: Trainer, dataset, num_labels: int, run_dir: Path, split: str) -> None:
    output = trainer.predict(dataset)
    logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    np.save(run_dir / f"{split}_logits.npy", logits.astype(np.float32))
    np.save(run_dir / f"{split}_labels.npy", multi_hot(dataset["label_ids"], num_labels))
    np.save(run_dir / f"{split}_hadm_ids.npy", np.asarray(dataset["hadm_id"], dtype=np.int64))


def format_example_prediction(
    tokenizer, dataset, vocab: list[str], run_dir: Path, threshold: float, index: int = 0
) -> str:
    """One test note (as the model saw it, after truncation) with its true and predicted codes."""
    probs = sigmoid(np.load(run_dir / "test_logits.npy")[index])
    example = dataset[index]
    note = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
    true_codes = [vocab[i] for i in example["label_ids"]]
    predicted = [(vocab[i], probs[i]) for i in np.argsort(-probs) if probs[i] >= threshold]
    lines = [
        "=== Note ===",
        note,
        "",
        f"=== True codes ({len(true_codes)}) ===",
        ", ".join(true_codes),
        "",
        f"=== Predicted codes at threshold {threshold:.2f} ({len(predicted)}) ===",
        ", ".join(f"{code} ({p:.2f}){' *' if code in true_codes else ''}" for code, p in predicted),
        "(* = correct)",
    ]
    return "\n".join(lines)


def show_example_prediction(config: dict[str, Any], example: str, run_dir: Path) -> None:
    """Print the example only for synthetic data: real MIMIC note text must never reach the terminal."""
    if resolve_mimic_dir(config["paths"]["mimic_dir"]) == SYNTHETIC_MIMIC_DIR:
        print(example)
        return
    path = run_dir / "example_prediction.txt"
    path.write_text(example, encoding="utf-8")
    print(f"Example prediction (contains MIMIC note text, do not share) saved to {path}")


def run_metadata(config: dict[str, Any], precision: str) -> dict[str, Any]:
    return {
        "seed": config["seed"],
        "precision": precision,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "started": datetime.now().isoformat(timespec="seconds"),
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def train(config: dict[str, Any]) -> Path:
    set_seed(config["seed"])  # seeds random, numpy and torch
    run_dir = resolve_path(config["paths"]["output_dir"]) / config["run_name"] / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    precision = resolve_precision(config["train"]["precision"])
    metadata = run_metadata(config, precision)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(run_dir / "run_metadata.json", metadata)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["checkpoint"])
    data = prepare_data(config, tokenizer)
    num_labels = len(data.vocab)
    train_counts = [data.train_counts[code] for code in data.vocab]
    write_json(
        run_dir / "label_vocab.json",
        {"label_setting": config["data"]["label_setting"], "codes": data.vocab, "train_counts": train_counts},
    )
    print(f"Labels: {num_labels}; admissions per split: { {s: len(d) for s, d in data.datasets.items()} }")

    loss_fn = make_loss_fn(config["loss"], np.asarray(train_counts), len(data.datasets["train"]))
    model = build_model(config, data.vocab, precision)

    os.environ["TENSORBOARD_LOGGING_DIR"] = str(run_dir / "tensorboard")
    trainer = MultiLabelTrainer(
        model=model,
        args=build_training_args(config, run_dir, precision),
        train_dataset=data.datasets["train"],
        eval_dataset=data.datasets["val"],
        data_collator=MultiLabelCollator(tokenizer, num_labels),
        processing_class=tokenizer,
        compute_metrics=eval_metrics,
        loss_fn=loss_fn,
    )
    #trainer.train(resume_from_checkpoint="outputs/top50-large/20261010-111505/checkpoints/checkpoint-11530")
    trainer.train()
    trainer.save_model(str(run_dir / "model"))

    for split in ("val", "test"):
        save_predictions(trainer, data.datasets[split], num_labels, run_dir, split)
    results = evaluate_run(run_dir)

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    writer.add_text("config", f"```yaml\n{yaml.safe_dump(config, sort_keys=False)}\n```")
    writer.add_text("run_metadata", f"```json\n{json.dumps(metadata, indent=2)}\n```")
    for name, value in results["test"].items():
        writer.add_scalar(f"test/{name}", value)
    writer.add_scalar("test/global_threshold", results["global_threshold"])
    writer.close()

    example = format_example_prediction(
        tokenizer, data.datasets["test"], data.vocab, run_dir, results["global_threshold"]
    )
    show_example_prediction(config, example, run_dir)
    print(json.dumps({"global_threshold": results["global_threshold"], "test": results["test"]}, indent=2))
    print(f"Run saved to {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=EXPERIMENTS, help="Experiment defined in icd10/config.py")
    args = parser.parse_args(argv)
    train(get_config(args.experiment))


if __name__ == "__main__":
    main()
