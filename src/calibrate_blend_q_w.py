#!/usr/bin/env python3
import csv,json,numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from train_modeling_b import ALL_CLASSES
labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};ids=[json.loads(l)['id'] for l in open('data/train.jsonl',encoding='utf8') if l.strip()];ids=[i for i in ids if split[i]=='val'];y=np.asarray([labels[i] for i in ids]);p=np.load('experiments/blend_q_w/oof.npy')
def score(w):return f1_score(y,np.asarray(ALL_CLASSES)[np.argmax(p*np.exp(w),1)],labels=ALL_CLASSES,average='macro',zero_division=0)
w=np.zeros(14);best=score(w);grid=np.log(np.array([.5,.6,.7,.8,.9,1,1.1,1.25,1.45,1.7,2.0]))
for r in range(5):
 ch=False
 for j in range(14):
  old=w[j];pick=old;local=best
  for d in grid:
   w[j]=old+d;v=score(w)
   if v>local:local=v;pick=w[j]
  w[j]=pick
  if local>best:best=local;ch=True
 print(r,best,flush=True)
 if not ch:break
print('BEST',best,np.exp(w),flush=True);Path('experiments/blend_q_w/calibration.json').write_text(json.dumps({'score':best,'weights':dict(zip(ALL_CLASSES,np.exp(w).tolist()))},indent=2))
