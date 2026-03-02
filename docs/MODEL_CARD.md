# Model Card - Directed Speech Classifier (RU)

## What
Binary "directed speech" classifier over text (typically ASR output):
- 1: addressed to the assistant
- 0: not addressed to the assistant

Model: fine-tuned `ruElectra-small` (Sequence Classification, 2 labels).

## Intended use
Used as an attention gate before NLU/LLM:
ASR -> directed-speech -> (if 1) NLU/LLM; otherwise ignore.

## Training data
Synthetic dataset from `scripts/generate_data_v3.py` with ASR-like noise and split by `group_id`.

## Metrics
See `test_report.json` (if attached) and `docs/REPRODUCIBILITY.md`.

## Decision threshold
At inference you can use:
- `p(directed) >= threshold` (default `threshold=0.7`)
- or logits `argmax` (see `--argmax` in inference)

Threshold should be selected for the target product based on FP/FN trade-off.

## Limitations & risks
- On real speech/ASR, distribution shifts are possible (accents, domains, slang).
- FP errors (false triggers) may wake the assistant unnecessarily.
- FN errors (misses) may suppress valid assistant requests.
Threshold tuning on real data and continuous error monitoring are recommended.

## License
Repository code is licensed under `LICENSE`.
The base pretrained model has its own license/terms on Hugging Face.
