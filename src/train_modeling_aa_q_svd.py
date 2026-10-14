#!/usr/bin/env python3
"""Q/source XGBoost plus compact TF-IDF/SVD lexical-topic features."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from train_modeling_b import ALL_CLASSES
labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};samples=[json.loads(l) for l in open('data/train.jsonl',encoding='utf8') if l.strip()]
def text(s):
 h=s.get('history') if isinstance(s.get('history'),list) else [];p=[str(s.get('current_prompt') or '')]
 for x in h[-5:]:
  if isinstance(x,dict):p.append(str(x.get('role',''))+' '+str(x.get('name',''))+' '+str(x.get('result_summary',''))[:80])
 return ' '.join(p).lower()
tr=[s for s in samples if split[s['id']]=='train'];va=[s for s in samples if split[s['id']]=='val'];ystr=np.asarray([labels[s['id']] for s in va]);toid={c:i for i,c in enumerate(ALL_CLASSES)};y=np.asarray([toid[v] for v in ystr]);groups=np.asarray([s['id'].split('-step_',1)[0] for s in va]);cache=Path('experiments/modeling_aa_q_svd');cache.mkdir(exist_ok=True)
if (cache/'svd_val.npy').exists():sv=np.load(cache/'svd_val.npy')
else:
 vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=100000,sublinear_tf=True);a=vec.fit_transform([text(s) for s in tr]);b=vec.transform([text(s) for s in va]);svd=TruncatedSVD(n_components=128,n_iter=5,random_state=42);svd.fit(a);sv=svd.transform(b).astype('float32');np.save(cache/'svd_val.npy',sv)
x=np.column_stack((np.load('experiments/modeling_w_xgb_tune/input.npz',allow_pickle=True)['x'],sv));print(x.shape,flush=True);pred=np.empty(len(y),dtype=int)
for n,(a,b) in enumerate(GroupKFold(5).split(x,y,groups),1):
 cnt=np.bincount(y[a],minlength=14);w=(len(a)/(14*cnt[y[a]]));m=XGBClassifier(n_estimators=1400,max_depth=10,learning_rate=.04,subsample=.85,colsample_bytree=.85,objective='multi:softprob',num_class=14,tree_method='hist',device='cuda',eval_metric='mlogloss',early_stopping_rounds=100,n_jobs=8,random_state=42+n);m.fit(x[a],y[a],sample_weight=w,eval_set=[(x[b],y[b])],verbose=False);pred[b]=m.predict(x[b]);print(n,m.best_iteration,flush=True)
s=f1_score(ystr,np.asarray(ALL_CLASSES)[pred],labels=ALL_CLASSES,average='macro',zero_division=0);print('AA_svd',s,flush=True);(cache/'score.txt').write_text(str(s)+'\n')
