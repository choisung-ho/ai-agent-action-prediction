#!/usr/bin/env python3
"""GPU XGBoost over Q state/margins plus source-routed E margins."""
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
samples=load_jsonl('data/train.jsonl');labels=load_labels('data/train_labels.csv');split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};ids,texts=build_modeling_e_features(samples);tr=np.asarray([i for i,z in enumerate(ids) if split[z]=='train']);va=np.asarray([i for i,z in enumerate(ids) if split[z]=='val']);ystr=np.asarray([labels[ids[i]] for i in va]);toid={c:i for i,c in enumerate(ALL_CLASSES)};y=np.asarray([toid[z] for z in ystr]);groups=np.asarray([ids[i].split('-step_',1)[0] for i in va]);src=np.asarray([source_of(z) for z in ids]);models=fit_source_models([texts[i] for i in tr],np.asarray([labels[ids[i]] for i in tr]),src[tr]);extra,_,_=source_e_features(models,[texts[i] for i in va],src[va]);x=np.column_stack((np.load('experiments/modeling_q/prepared_inputs.npz',allow_pickle=False)['x'],extra)).astype('float32');print(x.shape,flush=True);pred=np.empty(len(y),dtype=int)
for n,(a,b) in enumerate(GroupKFold(5).split(x,y,groups),1):
 cnt=np.bincount(y[a],minlength=14);w=np.sqrt(len(a)/(14*cnt[y[a]]));m=XGBClassifier(n_estimators=1200,max_depth=8,learning_rate=.05,subsample=.85,colsample_bytree=.8,objective='multi:softprob',num_class=14,tree_method='hist',device='cuda',eval_metric='mlogloss',early_stopping_rounds=100,n_jobs=8,random_state=42+n);m.fit(x[a],y[a],sample_weight=w,eval_set=[(x[b],y[b])],verbose=False);pred[b]=m.predict(x[b]);print(n,m.best_iteration,f1_score(ystr[b],np.asarray(ALL_CLASSES)[pred[b]],labels=ALL_CLASSES,average='macro',zero_division=0),flush=True)
s=f1_score(ystr,np.asarray(ALL_CLASSES)[pred],labels=ALL_CLASSES,average='macro',zero_division=0);print('W_xgb_q_source',s,flush=True);out=Path('experiments/modeling_w_xgb_q_source');out.mkdir(exist_ok=True);(out/'score.txt').write_text(str(s)+'\n')
