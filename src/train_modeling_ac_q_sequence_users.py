#!/usr/bin/env python3
"""Sequence-state selector enriched with recent user-message token features."""
import csv,json,re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from train_modeling_b import ALL_CLASSES
labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};samples=[json.loads(l) for l in open('data/train.jsonl',encoding='utf8') if l.strip()];rows=[s for s in samples if split[s['id']]=='val'];ystr=np.asarray([labels[s['id']] for s in rows]);toid={c:i for i,c in enumerate(ALL_CLASSES)};y=np.asarray([toid[v] for v in ystr]);groups=np.asarray([s['id'].split('-step_',1)[0] for s in rows]);rx=re.compile(r"[\w가-힣]+",re.UNICODE)
def d(s):
 h=s.get('history') if isinstance(s.get('history'),list) else [];a=[str(x.get('name') or x.get('action') or 'none').lower() for x in h if isinstance(x,dict) and x.get('role')=='assistant_action'];f={}
 for p,v in enumerate(reversed(a[-20:]),1):f[f'pos{p}={v}']=1
 for n in (2,3,4):
  for i in range(max(0,len(a)-20),len(a)-n+1):f[f'ng{n}='+'>'.join(a[i:i+n])]=1
 for v in a:f[f'count={v}']=f.get(f'count={v}',0)+1
 users=[str(x.get('content') or '').lower() for x in h if isinstance(x,dict) and x.get('role')=='user']
 for pos,msg in enumerate(reversed(users[-3:]),1):
  t=rx.findall(msg)[:60]
  for w in t:
   if len(w)>1:f[f'u{pos}:{w}']=1
  for i in range(len(t)-1):f[f'ub{pos}:{t[i]}_{t[i+1]}']=1
 return f
seq=FeatureHasher(n_features=1536,input_type='dict',alternate_sign=False).transform([d(s) for s in rows]).toarray().astype('float32');x=np.column_stack((np.load('experiments/modeling_w_xgb_tune/input.npz',allow_pickle=True)['x'],seq));print(x.shape,flush=True);pred=np.empty(len(y),dtype=int)
for n,(a,b) in enumerate(GroupKFold(5).split(x,y,groups),1):
 cnt=np.bincount(y[a],minlength=14);w=(len(a)/(14*cnt[y[a]]));m=XGBClassifier(n_estimators=1400,max_depth=10,learning_rate=.04,subsample=.85,colsample_bytree=.85,objective='multi:softprob',num_class=14,tree_method='hist',device='cuda',eval_metric='mlogloss',early_stopping_rounds=100,n_jobs=8,random_state=42+n);m.fit(x[a],y[a],sample_weight=w,eval_set=[(x[b],y[b])],verbose=False);pred[b]=m.predict(x[b]);print(n,m.best_iteration,flush=True)
s=f1_score(ystr,np.asarray(ALL_CLASSES)[pred],labels=ALL_CLASSES,average='macro',zero_division=0);print('AC_sequence_users',s,flush=True);out=Path('experiments/modeling_ac_q_sequence_users');out.mkdir(exist_ok=True);(out/'score.txt').write_text(str(s)+'\n')
