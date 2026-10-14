#!/usr/bin/env python3
"""Evaluate source-specific Q selectors on the frozen validation split."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,classification_report
from sklearn.model_selection import GroupKFold
from train_modeling_b import ALL_CLASSES

x=np.load('experiments/modeling_q/prepared_inputs.npz',allow_pickle=False)['x'];labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};ids=[json.loads(l)['id'] for l in open('data/train.jsonl',encoding='utf8') if l.strip()];ids=[i for i in ids if split[i]=='val'];y=np.asarray([labels[i] for i in ids]);src=np.asarray(['au' if i.startswith('sess_au_') else 'sim' for i in ids]);pred=np.empty(len(y),dtype=object)
for kind in ('sim','au'):
 idx=np.flatnonzero(src==kind); xx=x[idx]; yy=y[idx]; groups=np.asarray([ids[i].split('-step_',1)[0] for i in idx]);print(kind,xx.shape,flush=True)
 for n,(tr,va) in enumerate(GroupKFold(5).split(xx,yy,groups),1):
  m=RandomForestClassifier(n_estimators=300,min_samples_leaf=10,max_features=.7,class_weight='balanced_subsample',n_jobs=8,random_state=42).fit(xx[tr],yy[tr]);pred[idx[va]]=m.predict(xx[va]);print(kind,n,flush=True)
print('T_source_q',f1_score(y,pred,labels=ALL_CLASSES,average='macro',zero_division=0),flush=True);Path('experiments/modeling_t_source_q').mkdir(exist_ok=True);Path('experiments/modeling_t_source_q/report.txt').write_text(classification_report(y,pred,labels=ALL_CLASSES,digits=4,zero_division=0))
