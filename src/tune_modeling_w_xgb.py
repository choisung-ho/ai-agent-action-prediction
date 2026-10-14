#!/usr/bin/env python3
"""Hyperparameter sweep for source-aware Q XGBoost selector."""
import csv,json
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from data import load_jsonl,load_labels
from train_modeling_b import ALL_CLASSES
from train_modeling_e import build_modeling_e_features
from train_modeling_r_source_stack import source_of,fit_source_models,source_e_features
out=Path('experiments/modeling_w_xgb_tune');out.mkdir(exist_ok=True);cache=out/'input.npz'
if cache.exists():
 z=np.load(cache,allow_pickle=True);x=z['x'];ystr=z['ystr'];groups=z['groups']
else:
 samples=load_jsonl('data/train.jsonl');labels=load_labels('data/train_labels.csv');split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};ids,texts=build_modeling_e_features(samples);tr=np.asarray([i for i,v in enumerate(ids) if split[v]=='train']);va=np.asarray([i for i,v in enumerate(ids) if split[v]=='val']);ystr=np.asarray([labels[ids[i]] for i in va]);groups=np.asarray([ids[i].split('-step_',1)[0] for i in va]);src=np.asarray([source_of(v) for v in ids]);models=fit_source_models([texts[i] for i in tr],np.asarray([labels[ids[i]] for i in tr]),src[tr]);extra,_,_=source_e_features(models,[texts[i] for i in va],src[va]);x=np.column_stack((np.load('experiments/modeling_q/prepared_inputs.npz',allow_pickle=False)['x'],extra)).astype('float32');np.savez_compressed(cache,x=x,ystr=ystr,groups=groups)
toid={c:i for i,c in enumerate(ALL_CLASSES)};y=np.asarray([toid[v] for v in ystr]);folds=list(GroupKFold(5).split(x,y,groups));configs=[('d6_p05',6,.5,.05),('d10_p05',10,.5,.05),('d8_p10',8,1.,.05),('d10_p10',10,1.,.04)]
rows=[]
for name,depth,power,lr in configs:
 pred=np.empty(len(y),dtype=int)
 for n,(tr,va) in enumerate(folds,1):
  cnt=np.bincount(y[tr],minlength=14);w=(len(tr)/(14*cnt[y[tr]]))**power;m=XGBClassifier(n_estimators=1400,max_depth=depth,learning_rate=lr,subsample=.85,colsample_bytree=.85,min_child_weight=1,objective='multi:softprob',num_class=14,tree_method='hist',device='cuda',eval_metric='mlogloss',early_stopping_rounds=100,n_jobs=8,random_state=42+n);m.fit(x[tr],y[tr],sample_weight=w,eval_set=[(x[va],y[va])],verbose=False);pred[va]=m.predict(x[va])
 score=f1_score(ystr,np.asarray(ALL_CLASSES)[pred],labels=ALL_CLASSES,average='macro',zero_division=0);rows.append((name,score));print(name,score,flush=True)
(out/'results.csv').write_text('name,macro_f1\n'+'\n'.join(f'{a},{b}' for a,b in rows)+'\n')
