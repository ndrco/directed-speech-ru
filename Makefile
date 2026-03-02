.PHONY: venv install data train infer eval

venv:
	python -m venv .venv

install:
	. .venv/bin/activate && pip install -r requirements.txt

data:
	python scripts/generate_data_v3.py

train:
	python scripts/train_ruelectra_directed_v2.py --train data_v3_train.csv --val data_v3_val.csv --test data_v3_test.csv --out ./directed-ruElectra-small

infer:
	python scripts/infer_directed_v2.py --model ./directed-ruElectra-small

eval:
	python scripts/infer_directed_v2.py --model ./directed-ruElectra-small --eval data_v3_test.csv --topk 20 --threshold 0.7
