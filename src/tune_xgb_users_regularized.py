#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from train_modeling_b import ALL_CLASSES
z=np.load('experiments/modeling_ac_tune2/input.npz',allow_pickle=True);x=z['x'];ys=z['ystr'];groups=z['groups'];toid={c:i for i,c in enumerate(ALL_CLASSES)};y=np.asarray([toid[v] for v in ys]);folds=list(GroupKFold(5).split(x,y,groups))
cfgs=[('reg_a',dict(max_depth=12,colsample_bytree=.60,subsample=.90,min_child_weight=3,reg_lambda=5,gamma=.1,learning_rate=.035)),('reg_b',dict(max_depth=8,colsample_bytree=1.,subsample=.90,min_child_weight=3,reg_lambda=5,gamma=.1,learning_rate=.04)),('reg_c',dict(max_depth=14,colsample_bytree=.70,subsample=.85,min_child_weight=5,reg_lambda=8,gamma=.2,learning_rate=.03)),('lossguide',dict(max_depth=0,grow_policy='lossguide',max_leaves=256,colsample_bytree=.75,subsample=.9,min_child_weight=2,reg_lambda=3,gamma=.05,learning_rate=.04))]
rows=[]
for name,kw in cfgs:
 pred=np.empty(len(y),dtype=int)
 for n,(a,b) in enumerate(folds,1):
  cnt=np.bincount(y[a],minlength=14);w=(len(a)/(14*cnt[y[a]]));m=XGBClassifier(n_estimators=1800,objective='multi:softprob',num_class=14,tree_method='hist',device='cuda',eval_metric='mlogloss',early_stopping_rounds=120,n_jobs=8,random_state=42+n,**kw);m.fit(x[a],y[a],sample_weight=w,eval_set=[(x[b],y[b])],verbose=False);pred[b]=m.predict(x[b])
 s=f1_score(ys,np.asarray(ALL_CLASSES)[pred],labels=ALL_CLASSES,average='macro',zero_division=0);rows.append((name,s));print(name,s,flush=True)
out=Path('experiments/tune_xgb_users_regularized');out.mkdir(exist_ok=True);(out/'results.csv').write_text('name,score\n'+'\n'.join(f'{a},{b}' for a,b in rows)+'\n')
