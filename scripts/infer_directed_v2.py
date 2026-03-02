#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference + error analysis for directed-speech classifier.

Features:
1) Interactive mode (type text -> p(directed)).
2) Evaluation mode on labeled CSV and showing Top-K:
   - False Positives (label=0, pred=1) sorted by highest p(directed)
   - False Negatives (label=1, pred=0) sorted by lowest p(directed)
   i.e., "most confidently wrong" examples.

CSV format (minimum):
  text,label
Optional:
  group_id (ignored here)

Run examples:

Interactive:
  python infer_directed_v2.py --model ./directed-ruElectra-small

Evaluate + show top mistakes:
  python infer_directed_v2.py --model ./directed-ruElectra-small --eval data_v2_test.csv --topk 20 --threshold 0.7

Dependencies:
  pip install -U torch transformers numpy
"""

import argparse
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def softmax_torch(logits: torch.Tensor) -> torch.Tensor:
    logits = logits - logits.max(dim=-1, keepdim=True).values
    exp = torch.exp(logits)
    return exp / exp.sum(dim=-1, keepdim=True)


@dataclass
class PredRow:
    text: str
    label: Optional[int]  # None in interactive mode
    pred: int
    p_dir: float
    p_not: float


def load_labeled_csv(path: str, text_col: str = "text", label_col: str = "label") -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if text_col not in r.fieldnames or label_col not in r.fieldnames:
            raise ValueError(f"CSV must contain columns '{text_col}' and '{label_col}'. Found: {r.fieldnames}")
        for row in r:
            text = (row.get(text_col) or "").strip()
            if not text:
                continue
            try:
                label = int(row[label_col])
            except Exception:
                continue
            rows.append((text, label))
    return rows


def batched_predict(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 64,
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """
    Returns probs array of shape (N, 2) for [not_directed, directed].
    """
    model.eval()
    probs_all = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            probs = softmax_torch(out.logits).detach().cpu().numpy()
        probs_all.append(probs)
    return np.vstack(probs_all) if probs_all else np.zeros((0, 2), dtype=np.float32)


def decide_label(probs: np.ndarray, threshold: float = 0.7, use_threshold: bool = True) -> np.ndarray:
    p_dir = probs[:, 1]
    if use_threshold:
        return (p_dir >= threshold).astype(int)
    return probs.argmax(axis=1).astype(int)


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> List[List[int]]:
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return [[tn, fp], [fn, tp]]


def print_top_errors(rows: List[PredRow], topk: int = 20):
    fps = [r for r in rows if r.label == 0 and r.pred == 1]
    fns = [r for r in rows if r.label == 1 and r.pred == 0]

    fps.sort(key=lambda r: r.p_dir, reverse=True)     # most confident directed, but actually not
    fns.sort(key=lambda r: r.p_dir)                   # least directed, but actually directed

    print("\n=== TOP FALSE POSITIVES (label=0, pred=1) ===")
    if not fps:
        print("None 🎉")
    for r in fps[:topk]:
        print(f"p_dir={r.p_dir:.3f} | text: {r.text}")

    print("\n=== TOP FALSE NEGATIVES (label=1, pred=0) ===")
    if not fns:
        print("None 🎉")
    for r in fns[:topk]:
        print(f"p_dir={r.p_dir:.3f} | text: {r.text}")


def eval_mode(args, model, tokenizer, device: str):
    rows = load_labeled_csv(args.eval, text_col=args.text_col, label_col=args.label_col)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    texts = [t for t, _ in rows]
    y_true = np.array([y for _, y in rows], dtype=np.int64)

    probs = batched_predict(
        model, tokenizer, texts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )
    y_pred = decide_label(probs, threshold=args.threshold, use_threshold=args.use_threshold)

    acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    cm = confusion_matrix_binary(y_true, y_pred)

    # Precision/Recall/F1 for positive class
    tp = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"\nEVAL: {args.eval}")
    print(f"Rows: {len(rows)} | threshold={args.threshold:.2f} | use_threshold={args.use_threshold}")
    print(f"Accuracy: {acc:.4f} | Precision_pos: {precision:.4f} | Recall_pos: {recall:.4f} | F1_pos: {f1:.4f}")
    print("Confusion matrix [ [tn, fp], [fn, tp] ]:")
    print(cm)

    pred_rows: List[PredRow] = []
    for (text, label), pr in zip(rows, probs):
        p_not, p_dir = float(pr[0]), float(pr[1])
        pred = int(p_dir >= args.threshold) if args.use_threshold else int(np.argmax(pr))
        pred_rows.append(PredRow(text=text, label=label, pred=pred, p_dir=p_dir, p_not=p_not))

    print_top_errors(pred_rows, topk=args.topk)


def interactive_mode(args, model, tokenizer, device: str):
    model.eval()
    print("Type text (Ctrl+C to exit):")
    while True:
        try:
            text = input("> ").strip()
        except KeyboardInterrupt:
            print("\nbye")
            break

        if not text:
            continue

        probs = batched_predict(model, tokenizer, [text], max_length=args.max_length, batch_size=1, device=device)[0]
        p_not, p_dir = float(probs[0]), float(probs[1])
        pred = int(p_dir >= args.threshold) if args.use_threshold else int(np.argmax(probs))
        print(f"p(directed)={p_dir:.3f} -> {'DIRECTED' if pred == 1 else 'NOT'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path or HF id to fine-tuned model")
    ap.add_argument("--threshold", type=float, default=0.70, help="p(directed) threshold")
    ap.add_argument("--use_threshold", action="store_true", default=True,
                    help="Use threshold on p(directed) for pred label (default True).")
    ap.add_argument("--argmax", action="store_false", dest="use_threshold",
                    help="Use argmax logits for pred label instead of threshold.")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    ap.add_argument("--eval", default=None, help="Path to labeled CSV to evaluate + show top errors")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of eval rows (0 = no limit)")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    args = ap.parse_args()

    # device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)

    if args.eval:
        eval_mode(args, model, tokenizer, device)

    # Always allow interactive at the end (unless user explicitly wants eval-only)
    if not args.eval or True:
        interactive_mode(args, model, tokenizer, device)


if __name__ == "__main__":
    main()
