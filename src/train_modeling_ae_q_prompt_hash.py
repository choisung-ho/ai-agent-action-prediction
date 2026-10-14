#!/usr/bin/env python3
"""Q/source selector with hashed current + recent user prompt lexical features."""
import csv,json,re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from train_modeling_b import ALL_CLASSES
labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};samples=[json.loads(l) for l in open('data/train.jsonl',encoding='utf8') if l.strip()];rows=[s for s in samples if split[s['id']]=='val'];ystr=np.asarray([labels[s['id']] for s in rows]);toid={c:i for i,c in enumerate(ALL_CLASSES)};y=np.asarray([toid[v] for v in ystr]);groups=np.asarray([s['id'].split('-step_',1)[0] for s in rows]);rx=re.compile(r'[\w가-힣]+',re.UNICODE)
def add(v,prefix,msg):
 t=rx.findall(str(msg or '').lower())[:100]
 for w in t:
  if len(w)>1:v[f'{prefix}w:{w}']=1
 for i in range(len(t)-1):v[f'{prefix}b:{t[i]}_{t[i+1]}']=1
def f(s):
 v={};add(v,'c',s.get('current_prompt'));h=s.get('history') if isinstance(s.get('history'),list) else [];acts=[str(q.get('name') or q.get('action') or 'none').lower() for q in h if isinstance(q,dict) and q.get('role')=='assistant_action']
 for p,k in enumerate(reversed(acts[-20:]),1):v[f'pos{p}={k}']=1
 for n in (2,3,4):
  for i in range(max(0,len(acts)-20),len(acts)-n+1):v[f'ng{n}='+'>'.join(acts[i:i+n])]=1
 users=[q.get('content') for q in h if isinstance(q,dict) and q.get('role')=='user']
 for p,msg in enumerate(reversed(users[-3:]),1):add(v,f'u{p}',msg)
 return v
seq=FeatureHasher(n_features=4096,input_type='dict',alternate_sign=False).transform([f(s) for s in rows]).toarray().astype('float32');x=np.column_stack((np.load('experiments/modeling_w_xgb_tune/input.npz',allow_pickle=True)['x'],seq));print(x.shape,flush=True);pred=np.empty(len(y),dtype=int)
for n,(a,b) in enumerate(GroupKFold(5).split(x,y,groups),1):
 cnt=np.bincount(y[a],minlength=14);w=(len(a)/(14*cnt[y[a]]));m=XGBClassifier(n_estimators=1500,max_depth=10,learning_rate=.04,subsample=.85,colsample_bytree=.85,objective='multi:softprob',num_class=14,tree_method='hist',device='cuda',eval_metric='mlogloss',early_stopping_rounds=100,n_jobs=8,random_state=42+n);m.fit(x[a],y[a],sample_weight=w,eval_set=[(x[b],y[b])],verbose=False);pred[b]=m.predict(x[b]);print(n,m.best_iteration,flush=True)
s=f1_score(ystr,np.asarray(ALL_CLASSES)[pred],labels=ALL_CLASSES,average='macro',zero_division=0);print('AE_prompt_hash',s,flush=True);out=Path('experiments/modeling_ae_q_prompt_hash');out.mkdir(exist_ok=True);(out/'score.txt').write_text(str(s)+'\n')
