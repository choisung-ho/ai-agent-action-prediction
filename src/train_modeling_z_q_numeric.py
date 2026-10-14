#!/usr/bin/env python3
"""Q + continuous metadata / text-shape features for XGBoost."""
import csv,json,math
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from train_modeling_b import ALL_CLASSES
labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};samples=[json.loads(l) for l in open('data/train.jsonl',encoding='utf8') if l.strip()];rows=[s for s in samples if split[s['id']]=='val'];ystr=np.asarray([labels[s['id']] for s in rows]);toid={c:i for i,c in enumerate(ALL_CLASSES)};y=np.asarray([toid[v] for v in ystr]);groups=np.asarray([s['id'].split('-step_',1)[0] for s in rows])
langs=['py','js','ts','java','go','cpp','c','html','css','json','yaml','md','sql','sh']
def feat(s):
 m=s.get('session_meta') if isinstance(s.get('session_meta'),dict) else {};w=m.get('workspace') if isinstance(m.get('workspace'),dict) else {};p=str(s.get('current_prompt') or '');h=s.get('history') if isinstance(s.get('history'),list) else [];mix=w.get('language_mix') if isinstance(w.get('language_mix'),dict) else {};files=w.get('open_files') if isinstance(w.get('open_files'),list) else []
 base=[float(m.get('turn_index') or 0),float(m.get('budget_tokens_remaining') or 0)/200000,float(m.get('elapsed_session_sec') or 0)/1500,float(w.get('loc') or 0)/100000,len(h),len(files),len(p),len(p.split()),p.count('?'),p.count('!'),sum(c.isdigit() for c in p),sum('가'<=c<='힣' for c in p)/max(1,len(p)),sum(c.isascii() and c.isalpha() for c in p)/max(1,len(p))]
 return base+[float(mix.get(k,0)) for k in langs]
num=np.asarray([feat(s) for s in rows],dtype='float32');x=np.column_stack((np.load('experiments/modeling_w_xgb_tune/input.npz',allow_pickle=True)['x'],num));print(x.shape,flush=True);pred=np.empty(len(y),dtype=int)
for n,(tr,va) in enumerate(GroupKFold(5).split(x,y,groups),1):
 cnt=np.bincount(y[tr],minlength=14);ww=(len(tr)/(14*cnt[y[tr]]));m=XGBClassifier(n_estimators=1400,max_depth=10,learning_rate=.04,subsample=.85,colsample_bytree=.85,objective='multi:softprob',num_class=14,tree_method='hist',device='cuda',eval_metric='mlogloss',early_stopping_rounds=100,n_jobs=8,random_state=42+n);m.fit(x[tr],y[tr],sample_weight=ww,eval_set=[(x[va],y[va])],verbose=False);pred[va]=m.predict(x[va]);print(n,m.best_iteration,flush=True)
s=f1_score(ystr,np.asarray(ALL_CLASSES)[pred],labels=ALL_CLASSES,average='macro',zero_division=0);print('Z_numeric',s,flush=True);out=Path('experiments/modeling_z_q_numeric');out.mkdir(exist_ok=True);(out/'score.txt').write_text(str(s)+'\n')
