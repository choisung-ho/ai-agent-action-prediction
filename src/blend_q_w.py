#!/usr/bin/env python3
import csv,json,numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from train_modeling_b import ALL_CLASSES
labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};ids=[json.loads(l)['id'] for l in open('data/train.jsonl',encoding='utf8') if l.strip()];ids=[i for i in ids if split[i]=='val'];y=np.asarray([labels[i] for i in ids]);q=np.load('experiments/modeling_q_calibration/oof_probabilities.npy');w=np.load('experiments/modeling_w_xgb_tune/oof_probabilities.npy')
def sc(p):return f1_score(y,np.asarray(ALL_CLASSES)[np.argmax(p,1)],labels=ALL_CLASSES,average='macro',zero_division=0)
best=(0,None,None)
for a in np.arange(0,1.05,.05):
 p=a*q+(1-a)*w;s=sc(p);print(a,s)
 if s>best[0]:best=(s,a,p)
print('BEST',best[0],best[1]);Path('experiments/blend_q_w').mkdir(exist_ok=True);np.save('experiments/blend_q_w/oof.npy',best[2]);Path('experiments/blend_q_w/summary.txt').write_text(f'{best[0]},{best[1]}\n')
