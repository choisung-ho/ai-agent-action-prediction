#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
z=np.load('experiments/modeling_ac_tune2/input.npz',allow_pickle=True);x=z['x'];ys=z['ystr'];groups=z['groups'];classes=['read_file','grep_search','list_directory','glob_pattern','edit_file','write_file','apply_patch','run_bash','run_tests','lint_or_typecheck','ask_user','plan_task','web_search','respond_only'];toid={c:i for i,c in enumerate(classes)};y=np.asarray([toid[v] for v in ys]);p=np.zeros((len(y),14),dtype='float32')
for n,(a,b) in enumerate(GroupKFold(5).split(x,y,groups),1):
 cnt=np.bincount(y[a],minlength=14);w=(len(a)/(14*cnt[y[a]]));m=XGBClassifier(n_estimators=1400,max_depth=10,learning_rate=.04,subsample=.85,colsample_bytree=.85,objective='multi:softprob',num_class=14,tree_method='hist',device='cuda',eval_metric='mlogloss',early_stopping_rounds=100,n_jobs=8,random_state=42+n);m.fit(x[a],y[a],sample_weight=w,eval_set=[(x[b],y[b])],verbose=False);p[b]=m.predict_proba(x[b]);print(n,m.best_iteration,flush=True)
out=Path('experiments/modeling_ac_tune2');np.save(out/'oof_probabilities.npy',p)
