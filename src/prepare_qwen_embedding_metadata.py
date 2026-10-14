#!/usr/bin/env python3
import csv,json
from pathlib import Path
import numpy as np
out=Path('experiments/qwen15_embeddings');rows=[json.loads(line) for line in open('data/train.jsonl',encoding='utf8') if line.strip()];labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};np.save(out/'ids.npy',np.asarray([r['id'] for r in rows]));np.save(out/'labels.npy',np.asarray([labels[r['id']] for r in rows]));np.save(out/'split.npy',np.asarray([split[r['id']] for r in rows]));print(len(rows))
