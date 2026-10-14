#!/usr/bin/env python3
"""Validation-only probe for legitimate distribution signal in supplied IDs."""
import csv,json
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from train_modeling_b import ALL_CLASSES

labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))}
split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))}
ids=[json.loads(l)['id'] for l in open('data/train.jsonl',encoding='utf8') if l.strip()]
ids=[i for i in ids if split[i]=='val'];ys=np.asarray([labels[i] for i in ids]);toid={c:i for i,c in enumerate(ALL_CLASSES)};y=np.asarray([toid[v] for v in ys]);groups=np.asarray([i.split('-step_',1)[0] for i in ids])
def f(i):
 s=i.split('-step_')[0];n=int(s.rsplit('_',1)[-1]);step=int(i.rsplit('_',1)[-1])
 return [n/1000000,step]+[(n%m)/m for m in (2,3,4,5,7,11,13,17,23,31,47,97)]+[int(c)/9 for c in f'{n:06d}']
x=np.column_stack((np.load('experiments/modeling_w_xgb_tune/input.npz',allow_pickle=True)['x'],np.asarray([f(i) for i in ids],dtype='float32')))
pred=np.empty(len(y),dtype=int)
for n,(a,b) in enumerate(GroupKFold(5).split(x,y,groups),1):
 cnt=np.bincount(y[a],minlength=14);w=(len(a)/(14*cnt[y[a]]))
 m=XGBClassifier(n_estimators=1400,max_depth=10,learning_rate=.04,subsample=.85,colsample_bytree=.85,objective='multi:softprob',num_class=14,tree_method='hist',device='cuda',eval_metric='mlogloss',early_stopping_rounds=100,n_jobs=8,random_state=42+n)
 m.fit(x[a],y[a],sample_weight=w,eval_set=[(x[b],y[b])],verbose=False);pred[b]=m.predict(x[b]);print(n,m.best_iteration,flush=True)
s=f1_score(ys,np.asarray(ALL_CLASSES)[pred],labels=ALL_CLASSES,average='macro',zero_division=0);print('AC_idprobe',s,flush=True)
out=Path('experiments/modeling_ac_idprobe');out.mkdir(exist_ok=True);(out/'score.txt').write_text(str(s)+'\n')
