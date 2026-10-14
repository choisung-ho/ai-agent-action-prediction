#!/usr/bin/env python3
"""Test source-specific XGBoost on best user/action feature matrix."""
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from train_modeling_b import ALL_CLASSES
z=np.load('experiments/modeling_ac_tune2/input.npz',allow_pickle=True);x=z['x'];ys=z['ystr'];groups=z['groups'];cls=ALL_CLASSES;toid={c:i for i,c in enumerate(cls)};y=np.asarray([toid[v] for v in ys]);pred=np.argmax(np.load('experiments/modeling_ac_tune2/oof_probabilities.npy'),axis=1);# source from IDs reconstructed in matching val order
import csv,json
split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};ids=[json.loads(l)['id'] for l in open('data/train.jsonl',encoding='utf8') if l.strip()];ids=[i for i in ids if split[i]=='val'];src=np.asarray([1 if i.startswith('sess_au_') else 0 for i in ids])
for kind in (0,):
 ix=np.flatnonzero(src==kind);xx=x[ix];yy=y[ix];gg=groups[ix];print('source',kind,len(ix),flush=True)
 for n,(a,b) in enumerate(GroupKFold(5).split(xx,yy,gg),1):
  cnt=np.bincount(yy[a],minlength=14);w=(len(a)/(14*cnt[yy[a]]));m=XGBClassifier(n_estimators=1400,max_depth=10,learning_rate=.04,subsample=.85,colsample_bytree=.85,objective='multi:softprob',num_class=14,tree_method='hist',device='cuda',eval_metric='mlogloss',early_stopping_rounds=100,n_jobs=8,random_state=42+n);m.fit(xx[a],yy[a],sample_weight=w,eval_set=[(xx[b],yy[b])],verbose=False);pred[ix[b]]=m.predict(xx[b]);print(kind,n,m.best_iteration,flush=True)
s=f1_score(ys,np.asarray(cls)[pred],labels=cls,average='macro',zero_division=0);print('AG_source_users',s,flush=True);out=Path('experiments/modeling_ag_source_users');out.mkdir(exist_ok=True);(out/'score.txt').write_text(str(s)+'\n')
