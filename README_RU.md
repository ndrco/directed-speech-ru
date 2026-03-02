# Directed Speech Classifier (RU)

Бинарный классификатор, определяющий, **адресована ли реплика AI-помощнику** ("directed speech") по тексту (обычно это вывод ASR).

## Для чего нужен этот проект

Этот проект предназначен для дообучения `ruElectra-small`:
https://huggingface.co/ai-forever/ruElectra-small

Задача: бинарная классификация
- `1` -> фраза направлена AI-помощнику
- `0` -> фраза не направлена AI-помощнику

Целевое применение в продакшене: как этап фильтрации перед downstream-пайплайном ассистента.

Типичный поток:
1. Система распознавания речи (например, ASR на базе Whisper) генерирует текстовый поток.
2. Этот классификатор оценивает каждую фразу: `p(directed)`.
3. Только фразы, предсказанные как directed, передаются дальше в дообученный стек assistant/NLU/LLM.
4. Ненаправленные фразы отфильтровываются.

Почему это полезно:
- уменьшает случайные активации из фоновой речи или диалогов между людьми
- снижает лишнюю нагрузку на downstream-компоненты ассистента
- повышает устойчивость сценариев always-on или attention-mode

Пайплайн в этом репозитории:
1. генерация синтетического датасета с ASR-подобным шумом
2. дообучение `ruElectra-small`
3. инференс + оценка + анализ top FP/FN ошибок

**Пример метрик (из `test_report.json`):**
- Accuracy: 0.9959
- ROC-AUC: 0.9999
- Confusion matrix: [[237, 2], [0, 248]]
- Split (group): train 8447 / val 1523 / test 487 (groups: {'train': 321, 'val': 60, 'test': 21})

## Установка

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## Быстрый старт

### 1) Сгенерировать датасет
```bash
python scripts/generate_data_v3.py
```
Скрипт создаст: `data_v3.csv`, `data_v3_train.csv`, `data_v3_val.csv`, `data_v3_test.csv`.

### 2) Обучить модель
```bash
python scripts/train_ruelectra_directed_v2.py \
  --train data_v3_train.csv --val data_v3_val.csv --test data_v3_test.csv \
  --out ./directed-ruElectra-small
```

### 3) Инференс (интерактивно)
```bash
python scripts/infer_directed_v2.py --model ./directed-ruElectra-small
```
Введите текст, чтобы получить `p(directed)` и предсказанный класс.

### 4) Оценка + top ошибок
```bash
python scripts/infer_directed_v2.py --model ./directed-ruElectra-small \
  --eval data_v3_test.csv --topk 20 --threshold 0.7
```

## Формат данных

Минимальный CSV:
- `text` - строка
- `label` - 0/1

Опционально:
- `group_id` - используется для group split в тренировочном скрипте.

## Threshold vs argmax

По умолчанию предсказание делается по правилу `p(directed) >= threshold`.
Чтобы использовать `argmax` по логитам, передайте `--argmax` в `infer_directed_v2.py`.

## Структура репозитория

- `scripts/` - генерация данных / train / infer
- `docs/` - заметки по датасету/модели/воспроизводимости
- `data/` - датасет, сгенерированный `scripts/generate_data_v3.py` и дополненный реальными примерами в конце

## Лицензия

См. `LICENSE`.
