#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset generator v3 for "directed speech" (addressed to assistant) WITHOUT assistant name tokens.

New in v3:
- Post-noise validator for POSITIVE samples:
  prevents degenerate positives like filler-only fragments ("em..." / "uh...") that are too content-poor
  after ASR corruption. If a generated positive fails validation, it is discarded and regenerated.

Outputs:
  data_v3.csv           (full, shuffled)
  data_v3_train.csv     (group-split train)
  data_v3_val.csv       (group-split validation)
  data_v3_test.csv      (group-split test)

Columns:
  text,label,group_id

Run:
  python generate_data_v3.py
"""

import random, csv, re, pathlib

SEED = 42
TOTAL = 10_000
POS_RATIO = 0.5

OUT_DIR = pathlib.Path(".")  # current dir

# Exclude assistant name tokens entirely
BANNED = {"ассистент", "помощник", "assistant"}

people = ["Саша", "Маша", "Игорь", "Оля", "Дима", "Катя", "Лена", "Петя", "Ваня", "Надя", "Сергей", "Аня"]
roles  = ["мам", "мама", "пап", "папа", "брат", "сестра", "дед", "бабушка", "коллега", "шеф"]

fillers = ["эм", "мм", "ну", "короче", "так", "слушай", "ээ", "м", "хм"]

imperatives = [
    "включи", "выключи", "прибавь", "убавь", "поставь", "запусти", "останови",
    "найди", "покажи", "скажи", "подскажи", "объясни", "поясни", "напомни",
    "сделай", "создай", "добавь", "удали", "отмени", "переведи", "сократи",
    "перефразируй", "исправь", "проверь", "сравни", "продолжай", "пауза", "стоп"
]

q_starts = ["как", "почему", "зачем", "когда", "где", "что", "сколько", "какая", "какой", "какие"]
modals  = ["можешь", "умеешь", "можно ли", "нужно ли", "стоит ли", "получится ли"]

# Extra "control" words (very common in attention mode; allow 1-token positives)
control_prefixes = ["стоп", "отм", "пауз", "продол", "короч", "повтор", "тише", "хват", "отмен"]

topics_action = [
    "таймер на 10 минут", "будильник на 7 30", "напоминание через час",
    "музыку", "радио", "плейлист для работы", "громкость на 30 процентов",
    "погоду на завтра", "погоду в амстердаме", "температуру на улице",
    "курс евро к доллару", "новости за сегодня", "расписание на завтра",
    "список покупок", "заметку про идеи", "задачу в to do",
    "перевод на английский", "короткое резюме текста", "ошибки в коде",
    "пример на python", "регулярку для email", "объяснение vad", "настройку aec",
]

topics_question = [
    "какая погода завтра", "какая погода сейчас", "что за песня играет",
    "сколько времени", "сколько будет 17 умножить на 23", "какой сегодня день недели",
    "где ближайшая аптека", "как настроить git", "как работает docker",
    "почему код падает", "как исправить ошибку import", "как сделать запрос http",
    "что такое rag", "что такое vad", "как работает echo cancellation",
    "зачем нужен tokenizer", "как ускорить инференс", "что лучше threads или async",
]

people_commands = [
    "включи свет", "выключи свет", "поставь чайник", "закрой дверь", "открой окно",
    "прибавь громкость", "убавь громкость", "поставь таймер", "закажи пиццу",
    "объясни домашку", "помоги с кодом", "посмотри что там", "напомни мне завтра",
    "проверь почту", "сбрось ссылку", "скинь файл", "перезвони позже"
]

chatter_statements = [
    "я сегодня очень устал", "у меня завтра много дел", "вчера было холодно",
    "мы встречаемся в семь вечера", "я забыл ключи дома", "в холодильнике почти ничего нет",
    "я люблю эту песню", "у меня болит голова", "надо купить молоко и хлеб",
    "в офисе сегодня шумно", "я давно не был в отпуске", "мне нравится этот фильм",
    "кажется начался дождь", "я опаздываю на встречу", "в метро опять задержка",
    "мне пришло сообщение", "у нас завтра созвон", "я думаю взять выходной",
    "я читал статью про нейросети", "сегодня был сложный день",
    "мне нужно закончить задачу", "я не уверен что это хорошая идея",
    "у меня закончился кофе", "я хочу спать", "мне надо позвонить другу"
]

def contains_banned(text: str) -> bool:
    t = text.lower()
    return any(b in t for b in BANNED)

alphabet_cyr = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
alphabet_lat = "abcdefghijklmnopqrstuvwxyz"

def drop_punct(s: str) -> str:
    return re.sub(r"[,\.\!\?\:\;\(\)\[\]\"“”'«»]", "", s)

def random_char_noise(s: str, p=0.12) -> str:
    if random.random() > p or len(s) < 8:
        return s
    chars = list(s)
    n_ops = 1 if len(chars) < 20 else 2
    for _ in range(n_ops):
        idx = random.randrange(0, len(chars))
        op = random.choice(["del", "sub"])
        if op == "del" and len(chars) > 1:
            chars.pop(idx)
        elif op == "sub":
            ch = chars[idx]
            if ch.isalpha():
                if 'а' <= ch.lower() <= 'я' or ch.lower() == 'ё':
                    repl = random.choice(alphabet_cyr)
                else:
                    repl = random.choice(alphabet_lat)
                chars[idx] = repl if ch.islower() else repl.upper()
    return "".join(chars)

def maybe_partial(s: str, p=0.10) -> str:
    if random.random() > p:
        return s
    words = s.split()
    if len(words) < 3:
        return s
    if random.random() < 0.5:
        k = random.randint(1, min(3, len(words)-1))
        return " ".join(words[:-k]) + "…"
    else:
        k = random.randint(1, min(2, len(words)-2))
        return "… " + " ".join(words[k:])

def maybe_add_fillers(s: str, p=0.6) -> str:
    if random.random() > p:
        return s
    n = 1 if random.random() < 0.7 else 2
    pref = [random.choice(fillers) for _ in range(n)]
    joiner = ", " if random.random() < 0.5 else " "
    return joiner.join(pref) + (", " if random.random() < 0.5 else " ") + s

def maybe_remove_commas(s: str, p=0.35) -> str:
    return s.replace(",", "") if random.random() < p else s

def maybe_lower(s: str, p=0.5) -> str:
    return s.lower() if random.random() < p else s

def asr_noise(s: str) -> str:
    s = maybe_add_fillers(s)
    s = maybe_remove_commas(s)
    if random.random() < 0.55:
        s = drop_punct(s)
    s = random_char_noise(s)
    s = maybe_partial(s)
    s = maybe_lower(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:180]

def _tokenize_for_validation(text: str):
    t = text.lower()
    t = t.replace("…", " ")
    t = re.sub(r"[^\w\s\-]+", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    toks = [x for x in t.split(" ") if x]
    return toks

def validate_pos_post_noise(text: str) -> bool:
    """
    POS validator after ASR noise:
    - must not be only fillers
    - must contain at least one "anchor" token:
      * imperative-ish prefix OR question prefix OR modal prefix OR control prefix
    - allows 1-token positives only if token matches a control prefix (for example, short "stop/cancel" forms)
    """
    toks = _tokenize_for_validation(text)
    if not toks:
        return False

    toks_nf = [t for t in toks if t not in fillers]
    toks_nf = [t for t in toks_nf if len(t) >= 2]
    if not toks_nf:
        return False

    imp_pref = {w[:3] for w in imperatives if len(w) >= 3}
    q_pref = {w[:2] for w in q_starts if len(w) >= 2}
    modal_pref = {w[:3] for w in modals if len(w) >= 3}
    ctrl_pref = set(control_prefixes)

    def is_anchor(tok: str) -> bool:
        if any(tok.startswith(p) for p in ctrl_pref):
            return True
        if len(tok) >= 3 and tok[:3] in imp_pref:
            return True
        if len(tok) >= 2 and tok[:2] in q_pref:
            return True
        if len(tok) >= 3 and tok[:3] in modal_pref:
            return True
        return False

    anchors = [t for t in toks_nf if is_anchor(t)]

    if len(toks_nf) == 1:
        return bool(anchors) and any(toks_nf[0].startswith(p) for p in ctrl_pref)

    return len(anchors) >= 1

def gen_pos(base: str) -> str:
    if base == "pos_cmd_media":
        cmd = random.choice(["включи", "выключи", "прибавь", "убавь", "поставь"])
        obj = random.choice(["музыку", "радио", "плейлист", "громкость", "трек"])
        tail = random.choice(["", "на 30 процентов", "погромче", "потише", "для работы"])
        s = f"{cmd} {obj} {tail}".strip()
    elif base == "pos_cmd_timer":
        cmd = random.choice(["поставь", "отмени", "запусти", "останови"])
        obj = random.choice(["таймер", "будильник", "напоминание"])
        spec = random.choice(["на 10 минут", "на 5 минут", "на 1 час", "на завтра утром", "в 7 30"])
        s = f"{cmd} {obj} {spec}"
    elif base == "pos_cmd_text":
        cmd = random.choice(["переведи", "сократи", "перефразируй", "исправь", "проверь"])
        obj = random.choice(["этот текст", "сообщение", "абзац", "письмо", "описание"])
        style = random.choice(["", "по дружески", "официально", "коротко", "простыми словами"])
        s = f"{cmd} {obj} {style}".strip()
    elif base == "pos_q_weather":
        s = random.choice([
            "какая погода завтра", "какая погода сейчас", "будет ли дождь завтра",
            "какая температура на улице", "какой прогноз на выходные"
        ])
    elif base == "pos_q_info":
        s = random.choice([
            "сколько сейчас времени", "какой сегодня день недели", "сколько будет 17 умножить на 23",
            "какой курс евро к доллару", "что нового в новостях"
        ])
    elif base == "pos_help_tech":
        s = random.choice([
            "помоги разобраться с ошибкой в python", "объясни как работает vad",
            "как настроить aec", "почему код падает", "как ускорить инференс модели"
        ])
    elif base == "pos_dialog_control":
        s = random.choice([
            "стоп", "отмена", "пауза", "продолжай", "сделай короче", "давай подробнее", "повтори ещё раз"
        ])
    else:
        if random.random() < 0.6:
            s = f"{random.choice(imperatives)} {random.choice(topics_action)}"
        else:
            s = f"{random.choice(q_starts)} {random.choice(topics_question)}" if random.random() < 0.5 else f"{random.choice(modals)} {random.choice(topics_action)}"

    if random.random() < 0.35 and s not in {"стоп", "отмена", "пауза"}:
        s = s + " пожалуйста"
    if random.random() < 0.5:
        s += random.choice(["?", ".", "!"])

    s_noisy = asr_noise(s)

    # Post-noise validator for positives (discard degenerate cases)
    if not validate_pos_post_noise(s_noisy):
        return ""

    return s_noisy

def gen_neg(base: str) -> str:
    if base == "neg_cmd_to_person":
        who = random.choice(people + roles)
        cmd = random.choice(people_commands)
        s = f"{who}, {cmd}"
        if random.random() < 0.25:
            s += " пожалуйста"
    elif base == "neg_statement_random":
        s = random.choice(chatter_statements)
        if random.random() < 0.20:
            s += " " + random.choice(["в общем", "если честно", "наверное", "кажется"])
        if random.random() < 0.5:
            s += "."
    elif base == "neg_smalltalk_to_person":
        who = random.choice(people + roles)
        s = f"{who}, {random.choice(['как дела', 'ты где', 'что нового', 'ты скоро'])}"
        if random.random() < 0.7:
            s += "?"
    else:
        s = random.choice(chatter_statements) if random.random() < 0.75 else random.choice(["угу", "ага", "да", "нет", "ладно", "понятно", "ясно", "окей"])
        if random.random() < 0.35:
            s += "."
    return asr_noise(s)

pos_bases = ["pos_cmd_media", "pos_cmd_timer", "pos_cmd_text", "pos_q_weather", "pos_q_info", "pos_help_tech", "pos_dialog_control", "pos_mix"]
neg_bases = ["neg_cmd_to_person", "neg_statement_random", "neg_smalltalk_to_person", "neg_misc"]

def safe_text(text: str) -> str:
    if not text:
        return ""
    if contains_banned(text):
        return ""
    text = text.strip()
    if len(text) < 2:
        return ""
    return text[:180]

def mk_group_ids(base_groups, k_per_base: int):
    gids = []
    for bg in base_groups:
        for i in range(k_per_base):
            gids.append(f"{bg}__{i:02d}")
    random.shuffle(gids)
    return gids

def group_split(rows, train_ratio=0.80, val_ratio=0.15, test_ratio=0.05, seed=42):
    groups = list({gid for _, _, gid in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_g = set(groups[:n_train])
    val_g = set(groups[n_train:n_train+n_val])
    test_g = set(groups[n_train+n_val:])

    train, val, test = [], [], []
    for t, y, gid in rows:
        if gid in train_g:
            train.append((t, y, gid))
        elif gid in val_g:
            val.append((t, y, gid))
        else:
            test.append((t, y, gid))
    return train, val, test

def write_csv(path, data):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "group_id"])
        w.writerows(data)

def main():
    random.seed(SEED)
    n_pos = int(TOTAL * POS_RATIO)
    n_neg = TOTAL - n_pos

    pos_gids = mk_group_ids(pos_bases, 30)  # 240 groups
    neg_gids = mk_group_ids(neg_bases, 40)  # 160 groups

    def pick_gid(label: int) -> str:
        return random.choice(pos_gids if label == 1 else neg_gids)

    rows = []
    seen = set()
    counts = {0: 0, 1: 0}
    target = {0: n_neg, 1: n_pos}
    attempts = 0
    max_attempts = TOTAL * 120  # higher because we may discard invalid positives

    while (counts[0] < target[0] or counts[1] < target[1]) and attempts < max_attempts:
        attempts += 1
        label = 1 if (counts[1] < target[1] and (counts[0] >= target[0] or random.random() < 0.5)) else 0

        gid = pick_gid(label)
        base = gid.split("__")[0]
        text = safe_text(gen_pos(base) if label == 1 else gen_neg(base))
        if not text:
            continue

        key = (text.lower(), label)
        if key in seen:
            continue
        seen.add(key)
        rows.append((text, label, gid))
        counts[label] += 1

    if len(rows) < TOTAL:
        raise RuntimeError(f"Not enough unique samples: {len(rows)} / {TOTAL}")

    random.shuffle(rows)

    write_csv(OUT_DIR / "data_v3.csv", rows)
    train, val, test = group_split(rows)
    write_csv(OUT_DIR / "data_v3_train.csv", train)
    write_csv(OUT_DIR / "data_v3_val.csv", val)
    write_csv(OUT_DIR / "data_v3_test.csv", test)

    print("Done.")
    print("Wrote: data_v3.csv, data_v3_train.csv, data_v3_val.csv, data_v3_test.csv")

if __name__ == "__main__":
    main()
