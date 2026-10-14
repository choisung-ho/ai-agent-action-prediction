#!/usr/bin/env python3
"""OOF probability calibration for Q's macro-F1 decision rule."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from train_modeling_b import ALL_CLASSES

x=np.load('experiments/modeling_q/prepared_inputs.npz',allow_pickle=False)['x']; labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))}; split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};ids=[json.loads(l)['id'] for l in open('data/train.jsonl',encoding='utf8') if l.strip()];ids=[i for i in ids if split[i]=='val'];y=np.asarray([labels[i] for i in ids]);groups=np.asarray([i.split('-step_',1)[0] for i in ids]);proba=np.zeros((len(y),len(ALL_CLASSES)),dtype=np.float32)
for n,(tr,va) in enumerate(GroupKFold(5).split(x,y,groups),1):
 m=RandomForestClassifier(n_estimators=300,min_samples_leaf=10,max_features=.7,class_weight='balanced_subsample',n_jobs=8,random_state=42).fit(x[tr],y[tr]);p=m.predict_proba(x[va]);col={c:i for i,c in enumerate(m.classes_)}
 for j,c in enumerate(ALL_CLASSES):
  if c in col: proba[va,j]=p[:,col[c]]
 print('fold',n,flush=True)
def score(logw): return f1_score(y,np.asarray(ALL_CLASSES)[np.argmax(proba*np.exp(logw),axis=1)],labels=ALL_CLASSES,average='macro',zero_division=0)
logw=np.zeros(len(ALL_CLASSES));best=score(logw);grid=np.log(np.array([.55,.65,.75,.85,.93,1,1.08,1.18,1.32,1.5,1.75]))
for rnd in range(5):
 changed=False
 for j in range(len(ALL_CLASSES)):
  old=logw[j]; pick=old; local=best
  for delta in grid:
   logw[j]=old+delta;v=score(logw)
   if v>local:local=v;pick=logw[j]
  logw[j]=pick
  if local>best+1e-9:best=local;changed=True
 print('round',rnd,'score',best,flush=True)
 if not changed:break
weights=np.exp(logw);out=Path('experiments/modeling_q_calibration');out.mkdir(exist_ok=True);np.save(out/'oof_probabilities.npy',proba);(out/'calibration.json').write_text(json.dumps({'macro_f1':best,'weights':{c:float(w) for c,w in zip(ALL_CLASSES,weights)}},indent=2)+'\n');print('BEST',best,dict(zip(ALL_CLASSES,weights)),flush=True)
