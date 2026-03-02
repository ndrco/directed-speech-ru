# Model Card — Directed Speech Classifier (RU)

## What
Бинарный классификатор “directed speech” по тексту (обычно это выход ASR):
- 1: адресовано ассистенту
- 0: не адресовано ассистенту

Модель: fine-tuned `ruElectra-small` (Sequence Classification, 2 labels).

## Intended use
Используется как “attention gate” перед NLU/LLM:
ASR → directed-speech → (если 1) NLU/LLM; иначе игнор.

## Training data
Синтетический датасет из `scripts/generate_data_v3.py` с ASR-подобным шумом и split по `group_id`.

## Metrics
См. `test_report.json` (если приложен) и `docs/REPRODUCIBILITY.md`.

## Decision threshold
В инференсе можно использовать:
- `p(directed) >= threshold` (по умолчанию `threshold=0.7`)
- или `argmax` по логитам (см. `--argmax` в inference)

Порог выбирается под продукт (трейд-офф FP/FN).

## Limitations & risks
- На реальной речи/ASR возможны смещения (акценты, домены, сленг).
- Ошибки FP (ложные срабатывания) могут “будить” ассистента.
- Ошибки FN (пропуски) могут “глушить” ассистента.
Рекомендуется тюнить threshold на реальных данных и мониторить ошибки.

## License
Код в репозитории — по `LICENSE`. Предобученная базовая модель и её лицензия/условия — отдельно (Hugging Face).
