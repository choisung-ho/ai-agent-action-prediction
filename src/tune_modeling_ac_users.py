#!/usr/bin/env python3
import csv,json,re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from train_modeling_b import ALL_CLASSES
out=Path('experiments/modeling_ac_tune2');out.mkdir(exist_ok=True);cache=out/'input.npz'
if cache.exists():z=np.load(cache,allow_pickle=True);x=z['x'];ystr=z['ystr'];groups=z['groups']
else:
 labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};samples=[json.loads(l) for l in open('data/train.jsonl',encoding='utf8') if l.strip()];rows=[s for s in samples if split[s['id']]=='val'];ystr=np.asarray([labels[s['id']] for s in rows]);groups=np.asarray([s['id'].split('-step_',1)[0] for s in rows]);rx=re.compile(r'[\w가-힣]+',re.UNICODE)
 def f(s):
  h=s.get('history') if isinstance(s.get('history'),list) else [];a=[str(q.get('name') or q.get('action') or 'none').lower() for q in h if isinstance(q,dict) and q.get('role')=='assistant_action'];v={}
  for p,k in enumerate(reversed(a[-20:]),1):v[f'pos{p}={k}']=1
  for n in (2,3,4):
   for i in range(max(0,len(a)-20),len(a)-n+1):v[f'ng{n}='+'>'.join(a[i:i+n])]=1
  for action in a:v[f'count={action}']=v.get(f'count={action}',0)+1
  v[f'n_actions={min(len(a),25)}']=1
  u=[str(q.get('content') or '').lower() for q in h if isinstance(q,dict) and q.get('role')=='user']
  for p,m in enumerate(reversed(u[-3:]),1):
   t=rx.findall(m)[:60]
   for w in t:
    if len(w)>1:v[f'u{p}:{w}']=1
   for i in range(len(t)-1):v[f'ub{p}:{t[i]}_{t[i+1]}']=1
  return v
 seq=FeatureHasher(n_features=1536,input_type='dict',alternate_sign=False).transform([f(s) for s in rows]).toarray().astype('float32');x=np.column_stack((np.load('experiments/modeling_w_xgb_tune/input.npz',allow_pickle=True)['x'],seq));np.savez_compressed(cache,x=x,ystr=ystr,groups=groups)
toid={c:i for i,c in enumerate(ALL_CLASSES)};y=np.asarray([toid[q] for q in ystr]);folds=list(GroupKFold(5).split(x,y,groups));rows=[]
for name,depth,power,lr in [('d8p1',8,1,.04),('d12p1',12,1,.035),('d10p15',10,1.5,.035),('d10p2',10,2,.03)]:
 pred=np.empty(len(y),dtype=int)
 for n,(a,b) in enumerate(folds,1):
  cnt=np.bincount(y[a],minlength=14);w=(len(a)/(14*cnt[y[a]]))**power;m=XGBClassifier(n_estimators=1600,max_depth=depth,learning_rate=lr,subsample=.85,colsample_bytree=.85,objective='multi:softprob',num_class=14,tree_method='hist',device='cuda',eval_metric='mlogloss',early_stopping_rounds=100,n_jobs=8,random_state=42+n);m.fit(x[a],y[a],sample_weight=w,eval_set=[(x[b],y[b])],verbose=False);pred[b]=m.predict(x[b])
 s=f1_score(ystr,np.asarray(ALL_CLASSES)[pred],labels=ALL_CLASSES,average='macro',zero_division=0);rows.append((name,s));print(name,s,flush=True)
(out/'results.csv').write_text('name,score\n'+'\n'.join(f'{a},{b}' for a,b in rows)+'\n')
