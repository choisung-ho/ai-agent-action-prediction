#!/usr/bin/env python3
import csv,json,numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from train_modeling_b import ALL_CLASSES
labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};ids=[json.loads(l)['id'] for l in open('data/train.jsonl',encoding='utf8') if l.strip()];ids=[i for i in ids if split[i]=='val'];ys=np.asarray([labels[i] for i in ids]);toid={c:i for i,c in enumerate(ALL_CLASSES)};y=np.asarray([toid[v] for v in ys]);groups=np.asarray([i.split('-step_',1)[0] for i in ids]);q=np.load('experiments/modeling_q_calibration/oof_probabilities.npy');w=np.load('experiments/modeling_w_xgb_tune/oof_probabilities.npy');x=np.column_stack((q,w)).astype('float32');pred=np.empty(len(y),dtype=int)
for n,(tr,va) in enumerate(GroupKFold(5).split(x,y,groups),1):
 cnt=np.bincount(y[tr],minlength=14);weight=(len(tr)/(14*cnt[y[tr]]));m=XGBClassifier(n_estimators=600,max_depth=4,learning_rate=.04,subsample=.9,colsample_bytree=.9,objective='multi:softprob',num_class=14,tree_method='hist',device='cuda',eval_metric='mlogloss',early_stopping_rounds=80,n_jobs=8,random_state=42+n);m.fit(x[tr],y[tr],sample_weight=weight,eval_set=[(x[va],y[va])],verbose=False);pred[va]=m.predict(x[va]);print(n,m.best_iteration,flush=True)
s=f1_score(ys,np.asarray(ALL_CLASSES)[pred],labels=ALL_CLASSES,average='macro',zero_division=0);print('META',s,flush=True);Path('experiments/meta_blend_q_w').mkdir(exist_ok=True);Path('experiments/meta_blend_q_w/score.txt').write_text(str(s)+'\n')
