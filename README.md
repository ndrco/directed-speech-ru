![CI](https://github.com/ТВОЙ_НИК/directed-speech-ru/actions/workflows/ci.yml/badge.svg)

# Directed Speech Classifier (RU)

Бинарный классификатор, определяющий, **адресована ли реплика ассистенту** (“directed speech”) по тексту (обычно это вывод ASR).

Пайплайн:
1) генерация синтетических данных с ASR-подобным шумом  
2) fine-tune `ruElectra-small`  
3) инференс + eval + анализ ошибок (top FP/FN)

**Пример метрик (из `test_report.json`):**
- Accuracy: 0.9959
- ROC-AUC: 0.9999
- Confusion matrix: [[237, 2], [0, 248]]
- Split (group): train 8447 / val 1523 / test 487 (groups: {'train': 321, 'val': 60, 'test': 21})

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## Quickstart

### 1) Generate dataset
```bash
python scripts/generate_data_v3.py
```
Скрипт создаст: `data_v3.csv`, `data_v3_train.csv`, `data_v3_val.csv`, `data_v3_test.csv`.

### 2) Train
```bash
python scripts/train_ruelectra_directed_v2.py \
  --train data_v3_train.csv --val data_v3_val.csv --test data_v3_test.csv \
  --out ./directed-ruElectra-small
```

### 3) Inference (interactive)
```bash
python scripts/infer_directed_v2.py --model ./directed-ruElectra-small
```
Введите текст → получите `p(directed)` и класс.

### 4) Evaluate + Top errors
```bash
python scripts/infer_directed_v2.py --model ./directed-ruElectra-small \
  --eval data_v3_test.csv --topk 20 --threshold 0.7
```

## Data format

CSV минимально:
- `text` — строка
- `label` — 0/1

Опционально:
- `group_id` — используется для group split в тренировочном скрипте.

## Threshold vs argmax

По умолчанию предсказание делается по порогу `p(directed) >= threshold`.
Чтобы использовать `argmax` по логитам, передайте `--argmax` в `infer_directed_v2.py`.

## Repository structure

- `scripts/` — генерация данных / train / infer
- `docs/` — dataset/model/reproducibility
- `data/` — датасет; сгенерирован при помощи scripts/generate_data_v3.py и дополнен реальными приммерами в конце.

## License

См. файл `LICENSE`.
