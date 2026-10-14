#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from train_modeling_b import ALL_CLASSES
z=np.load('experiments/modeling_ac_tune2/input.npz',allow_pickle=True);x=z['x'];ys=z['ystr'];groups=z['groups'];p=np.empty(len(ys),dtype=object)
for n,(a,b) in enumerate(GroupKFold(5).split(x,ys,groups),1):
 m=CatBoostClassifier(iterations=1200,depth=10,learning_rate=.06,loss_function='MultiClass',eval_metric='TotalF1:average=Macro',auto_class_weights='Balanced',l2_leaf_reg=7,random_seed=42+n,task_type='GPU',devices='0',allow_writing_files=False,verbose=False);m.fit(x[a],ys[a],eval_set=(x[b],ys[b]),early_stopping_rounds=120,use_best_model=True);p[b]=m.predict(x[b]).reshape(-1);print(n,m.get_best_iteration(),f1_score(ys[b],p[b],labels=ALL_CLASSES,average='macro',zero_division=0),flush=True)
s=f1_score(ys,p,labels=ALL_CLASSES,average='macro',zero_division=0);print('AH_cat_users',s,flush=True);out=Path('experiments/modeling_ah_cat_users');out.mkdir(exist_ok=True);(out/'score.txt').write_text(str(s)+'\n')
