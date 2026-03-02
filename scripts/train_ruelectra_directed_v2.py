#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune RU-ELECTRA-small for binary "directed speech" classification:
label=1 -> addressed (should trigger assistant while in attention mode)
label=0 -> not addressed

Supports:
A) Pre-split CSVs:
   --train data_v2_train.csv --val data_v2_val.csv --test data_v2_test.csv

B) Single CSV + GROUP split (no leakage by template group):
   --data data_v2.csv   (must include column: group_id)
   Splits are done by group_id into train/val/test.

Columns expected:
  - text (str)
  - label (int: 0/1)
  - optional group_id (str) for group-split

Dependencies:
  pip install -U torch transformers datasets accelerate numpy scikit-learn

Run examples:
  python train_ruelectra_directed.py --train data_v2_train.csv --val data_v2_val.csv --test data_v2_test.csv --out ./directed-ruElectra-small
  python train_ruelectra_directed.py --data data_v2.csv --out ./directed-ruElectra-small

Notes:
- "Some weights ... newly initialized" is normal: classification head is created for this downstream task.
- Transformers versions differ: newer versions use eval_strategy instead of evaluation_strategy.
  This script filters TrainingArguments kwargs by the actual installed signature.
"""

import argparse
import json
import os
import random
import inspect
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)

def compute_metrics(eval_pred):
    """Metrics for Trainer (computed on eval split each epoch)."""
    logits, labels = eval_pred
    probs = softmax_np(logits)
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        auc = float("nan")

    return {
        "accuracy": float(acc),
        "precision_pos": float(precision),
        "recall_pos": float(recall),
        "f1_pos": float(f1),
        "roc_auc": float(auc),
    }

def load_single_csv(path: str):
    return load_dataset("csv", data_files={"data": path})["data"]

def load_split_csvs(train_path: str, val_path: str, test_path: str):
    dsd = load_dataset("csv", data_files={"train": train_path, "val": val_path, "test": test_path})
    return dsd["train"], dsd["val"], dsd["test"]

def group_split_dataset(ds, group_col: str, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    """Split HF Dataset by group_id without leakage across splits."""
    if group_col not in ds.column_names:
        raise ValueError(f"Column '{group_col}' not found. Available: {ds.column_names}")

    gids = ds[group_col]
    groups = list(set(gids))
    rng = random.Random(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_g = set(groups[:n_train])
    val_g = set(groups[n_train:n_train + n_val])
    test_g = set(groups[n_train + n_val:])

    train_idx, val_idx, test_idx = [], [], []
    for i, gid in enumerate(gids):
        if gid in train_g:
            train_idx.append(i)
        elif gid in val_g:
            val_idx.append(i)
        else:
            test_idx.append(i)

    train_ds = ds.select(train_idx)
    val_ds = ds.select(val_idx)
    test_ds = ds.select(test_idx)

    # sanity check leakage
    assert train_g.isdisjoint(val_g) and train_g.isdisjoint(test_g) and val_g.isdisjoint(test_g)

    return train_ds, val_ds, test_ds, (len(train_g), len(val_g), len(test_g))

def main():
    ap = argparse.ArgumentParser()

    # Data inputs:
    ap.add_argument("--data", default=None,
                    help="Single CSV (text,label[,group_id]). If --train/--val/--test not provided, used as source for splitting.")
    ap.add_argument("--train", default=None, help="Train CSV (text,label[,group_id])")
    ap.add_argument("--val", default=None, help="Validation CSV (text,label[,group_id])")
    ap.add_argument("--test", default=None, help="Test CSV (text,label[,group_id])")

    # Group split controls (used when --data is used, not split CSVs)
    ap.add_argument("--group_split", action="store_true", default=True,
                    help="Use group-split by group_id when --data is provided (default: True).")
    ap.add_argument("--no_group_split", action="store_false", dest="group_split",
                    help="Disable group split and use random split when --data is provided.")
    ap.add_argument("--group_col", default="group_id", help="Group column name for group-split. Default: group_id")

    # Split ratios (used when --data is used)
    ap.add_argument("--train_ratio", type=float, default=0.80)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.05)

    # Model / training args:
    ap.add_argument("--model", default="ai-forever/ruElectra-small",
                    help="HF model id. Alternative: sberbank-ai/ruElectra-small")
    ap.add_argument("--out", default="./directed-ruElectra-small", help="Output dir to save model+tokenizer")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", help="Use fp16 (GPU only).")
    ap.add_argument("--bf16", action="store_true", help="Use bf16 (Ampere+).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Validate split ratios
    s = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {s}")

    # 1) Load data
    if args.train and args.val and args.test:
        train_ds, eval_ds, test_ds = load_split_csvs(args.train, args.val, args.test)
        split_info = {"mode": "explicit", "train": len(train_ds), "val": len(eval_ds), "test": len(test_ds)}
    else:
        if not args.data:
            raise ValueError("Provide either --train/--val/--test OR --data.")
        ds = load_single_csv(args.data)

        # Optional: shuffle before random split only; for group-split we shuffle groups instead
        if args.group_split:
            train_ds, eval_ds, test_ds, gcounts = group_split_dataset(
                ds, group_col=args.group_col,
                train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
                seed=args.seed
            )
            split_info = {
                "mode": "group",
                "group_col": args.group_col,
                "groups": {"train": gcounts[0], "val": gcounts[1], "test": gcounts[2]},
                "rows": {"train": len(train_ds), "val": len(eval_ds), "test": len(test_ds)},
            }
        else:
            ds = ds.shuffle(seed=args.seed)
            tmp = ds.train_test_split(test_size=args.test_ratio, seed=args.seed)
            trainval = tmp["train"]
            test_ds = tmp["test"]
            trainval = trainval.train_test_split(test_size=args.val_ratio / (args.train_ratio + args.val_ratio), seed=args.seed)
            train_ds = trainval["train"]
            eval_ds = trainval["test"]
            split_info = {"mode": "random", "train": len(train_ds), "val": len(eval_ds), "test": len(test_ds)}

    print("DATA SPLIT:", json.dumps(split_info, ensure_ascii=False, indent=2))

    # 2) Tokenizer + tokenization
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tok, batched=True)
    eval_ds = eval_ds.map(tok, batched=True)
    test_ds = test_ds.map(tok, batched=True)

    # 3) Rename label -> labels, keep minimal columns
    def prepare(ds_):
        if "label" not in ds_.column_names:
            raise ValueError(f"CSV must have 'label' column. Found: {ds_.column_names}")
        ds_ = ds_.rename_column("label", "labels")
        cols = ["input_ids", "attention_mask", "labels"]
        if "token_type_ids" in ds_.column_names:
            cols.append("token_type_ids")
        return ds_.with_format("torch", columns=cols)

    train_ds = prepare(train_ds)
    eval_ds = prepare(eval_ds)
    test_ds = prepare(test_ds)

    # 4) Model
    id2label = {0: "not_directed", 1: "directed"}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5) TrainingArguments compatibility across transformers versions
    use_fp16 = args.fp16 and torch.cuda.is_available()
    use_bf16 = args.bf16 and torch.cuda.is_available()

    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())

    # Steps/epoch for older versions (if using eval_steps/save_steps)
    steps_per_epoch = max(1, int(np.ceil(len(train_ds) / (args.batch * max(1, args.grad_accum)))))

    ta_kwargs = dict(
        output_dir=args.out,
        overwrite_output_dir=True,
        seed=args.seed,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        # Newer transformers:
        eval_strategy="epoch",
        evaluation_strategy="epoch",  # older name (filtered if unsupported)
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_pos",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",

        # Mixed precision (GPU only):
        fp16=use_fp16,
        bf16=use_bf16,

        # Older transformers fallback:
        evaluate_during_training=True,
        eval_steps=steps_per_epoch,
        save_steps=steps_per_epoch,
    )

    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in allowed}
    targs = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # 6) Test: nice report
    pred = trainer.predict(test_ds)
    logits = pred.predictions
    labels = pred.label_ids
    probs = softmax_np(logits)
    preds = np.argmax(logits, axis=-1)

    report_text = classification_report(
        labels, preds,
        target_names=[id2label[0], id2label[1]],
        digits=4,
        zero_division=0
    )
    cm = confusion_matrix(labels, preds).tolist()

    try:
        auc = float(roc_auc_score(labels, probs[:, 1]))
    except Exception:
        auc = float("nan")

    acc = float(accuracy_score(labels, preds))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1], zero_division=0
    )

    test_summary = {
        "data_split": split_info,
        "accuracy": acc,
        "roc_auc": auc,
        "per_class": {
            id2label[0]: {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
            id2label[1]: {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
        },
        "confusion_matrix": cm,
        "classification_report": report_text,
    }

    print("\n=== TEST REPORT ===")
    print(report_text)
    print("Confusion matrix [ [tn, fp], [fn, tp] ]:")
    print(cm)
    print(f"ROC-AUC: {auc:.4f}")
    print("===================\n")

    report_path = os.path.join(args.out, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(test_summary, f, ensure_ascii=False, indent=2)

    # 7) Save best model
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved model+tokenizer to: {args.out}")
    print(f"Saved test report to: {report_path}")

if __name__ == "__main__":
    main()
