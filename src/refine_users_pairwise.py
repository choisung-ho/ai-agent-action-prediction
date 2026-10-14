#!/usr/bin/env python3
"""Apply targeted pairwise text refiners to the best user-sequence selector OOF."""
import csv,json
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.svm import LinearSVC
from train_modeling_b import ALL_CLASSES
from train_modeling_e import build_modeling_e_features
z=np.load('experiments/modeling_ac_tune2/input.npz',allow_pickle=True);x=z['x'];ys=z['ystr'];groups=z['groups'];toid={c:i for i,c in enumerate(ALL_CLASSES)};y=np.asarray([toid[v] for v in ys]);p=np.empty(len(y),dtype=int)
for n,(a,b) in enumerate(GroupKFold(5).split(x,y,groups),1):
 cnt=np.bincount(y[a],minlength=14);w=(len(a)/(14*cnt[y[a]]));m=XGBClassifier(n_estimators=1400,max_depth=10,learning_rate=.04,subsample=.85,colsample_bytree=.85,objective='multi:softprob',num_class=14,tree_method='hist',device='cuda',eval_metric='mlogloss',early_stopping_rounds=100,n_jobs=8,random_state=42+n);m.fit(x[a],y[a],sample_weight=w,eval_set=[(x[b],y[b])],verbose=False);p[b]=m.predict(x[b])
base=np.asarray(ALL_CLASSES)[p];print('base',f1_score(ys,base,labels=ALL_CLASSES,average='macro',zero_division=0),flush=True)
labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};samples=[json.loads(l) for l in open('data/train.jsonl',encoding='utf8') if l.strip()];ids,texts=build_modeling_e_features(samples);tr=[i for i,k in enumerate(ids) if split[k]=='train'];va=[i for i,k in enumerate(ids) if split[k]=='val']
for aa,bb in [('web_search','ask_user'),('ask_user','plan_task')]:
 idx=[i for i in tr if labels[ids[i]] in {aa,bb}];model=Pipeline([('f',FeatureUnion([('w',TfidfVectorizer(ngram_range=(1,3),min_df=2,max_features=100000,sublinear_tf=True)),('c',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=100000,sublinear_tf=True))])),('s',LinearSVC(C=.5,class_weight='balanced'))]);model.fit([texts[i] for i in idx],[labels[ids[i]] for i in idx]);q=model.predict([texts[i] for i in va]);cand=base.copy();mask=np.isin(cand,[aa,bb]);cand[mask]=q[mask];print(aa,bb,mask.sum(),f1_score(ys,cand,labels=ALL_CLASSES,average='macro',zero_division=0),flush=True);base=cand
Path('experiments/refine_users_pairwise').mkdir(exist_ok=True);Path('experiments/refine_users_pairwise/score.txt').write_text(str(f1_score(ys,base,labels=ALL_CLASSES,average='macro',zero_division=0))+'\n')
