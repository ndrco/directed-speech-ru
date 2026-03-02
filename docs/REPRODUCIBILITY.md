# Reproducibility

Цель: воспроизводимо получить модель directed-speech classifier (RU) и метрики.

## Environment
- Python: 3.10+
- Зависимости: см. `requirements.txt`
- Для GPU: CUDA-совместимый torch (по вашей системе)

## Steps

### 1) Generate synthetic dataset
```bash
python scripts/generate_data_v3.py
```
Выход: `data_v3.csv` и сплиты `data_v3_*`.

### 2) Train
```bash
python scripts/train_ruelectra_directed_v2.py \
  --train data_v3_train.csv --val data_v3_val.csv --test data_v3_test.csv \
  --out ./directed-ruElectra-small
```

### 3) Evaluate / error analysis
```bash
python scripts/infer_directed_v2.py --model ./directed-ruElectra-small \
  --eval data_v3_test.csv --topk 20 --threshold 0.7
```

## Notes
- Для сравнения метрик используйте одинаковый seed и одинаковые версии библиотек.

