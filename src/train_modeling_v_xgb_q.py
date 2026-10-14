#!/usr/bin/env python3
"""GPU XGBoost selector over Q's 450 inference-safe features."""
import csv,json,collections
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score,classification_report
from sklearn.model_selection import GroupKFold
from train_modeling_b import ALL_CLASSES
x=np.load('experiments/modeling_q/prepared_inputs.npz',allow_pickle=False)['x'];labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};ids=[json.loads(l)['id'] for l in open('data/train.jsonl',encoding='utf8') if l.strip()];ids=[i for i in ids if split[i]=='val'];ystr=np.asarray([labels[i] for i in ids]);toid={c:i for i,c in enumerate(ALL_CLASSES)};y=np.asarray([toid[c] for c in ystr]);groups=np.asarray([i.split('-step_',1)[0] for i in ids]);pred=np.empty(len(y),dtype=int)
for n,(tr,va) in enumerate(GroupKFold(5).split(x,y,groups),1):
 counts=np.bincount(y[tr],minlength=len(ALL_CLASSES));w=np.sqrt(len(tr)/(len(ALL_CLASSES)*counts[y[tr]]))
 m=XGBClassifier(n_estimators=1200,max_depth=8,learning_rate=.05,subsample=.85,colsample_bytree=.8,objective='multi:softprob',num_class=len(ALL_CLASSES),tree_method='hist',device='cuda',eval_metric='mlogloss',early_stopping_rounds=100,n_jobs=8,random_state=42+n);m.fit(x[tr],y[tr],sample_weight=w,eval_set=[(x[va],y[va])],verbose=False);pred[va]=m.predict(x[va]);print('fold',n,m.best_iteration,f1_score(ystr[va],np.asarray(ALL_CLASSES)[pred[va]],labels=ALL_CLASSES,average='macro',zero_division=0),flush=True)
p=np.asarray(ALL_CLASSES)[pred];s=f1_score(ystr,p,labels=ALL_CLASSES,average='macro',zero_division=0);print('V_xgb_q',s,flush=True);out=Path('experiments/modeling_v_xgb_q');out.mkdir(exist_ok=True);(out/'report.txt').write_text(classification_report(ystr,p,labels=ALL_CLASSES,digits=4,zero_division=0));(out/'score.txt').write_text(str(s)+'\n')
