#!/usr/bin/env python3
"""XGBoost selector with Q plus source-routed E/D/B margin features."""
import csv,json
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from data import load_jsonl,load_labels
from train_modeling_b import ALL_CLASSES,build_modeling_b_features,make_classifier
from train_modeling_d import fit_two_stage,predict_two_stage,GROUP_TO_LABELS
from train_modeling_e import build_modeling_e_features
from train_modeling_r_source_stack import source_of,fit_source_models,source_e_features

samples=load_jsonl('data/train.jsonl');lab=load_labels('data/train_labels.csv');split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};ids,e=build_modeling_e_features(samples);_,b=build_modeling_b_features(samples,history_n=3,include_meta=True);tr=np.asarray([i for i,z in enumerate(ids) if split[z]=='train']);va=np.asarray([i for i,z in enumerate(ids) if split[z]=='val']);ystr=np.asarray([lab[ids[i]] for i in va]);toid={c:i for i,c in enumerate(ALL_CLASSES)};y=np.asarray([toid[v] for v in ystr]);groups=np.asarray([ids[i].split('-step_',1)[0] for i in va]);src=np.asarray([source_of(z) for z in ids])
em=fit_source_models([e[i] for i in tr],np.asarray([lab[ids[i]] for i in tr]),src[tr]);ef,_,_=source_e_features(em,[e[i] for i in va],src[va])
dm={};bm={}
for kind in ('sim','au'):
 ii=tr[src[tr]==kind];yy=[lab[ids[i]] for i in ii];dm[kind]=fit_two_stage([b[i] for i in ii],yy,{'coarse_c':1.2,'fine_c':.8,'coarse_max_features':100000,'fine_max_features':80000});bm[kind]=make_classifier('B3');bm[kind].fit([b[i] for i in ii],yy)
def dec(model,texts,labels):
 z=np.asarray(model.decision_function(texts));z=np.column_stack((-z,z)) if z.ndim==1 else z;out=np.zeros((len(texts),len(labels)),dtype='float32');pos={c:i for i,c in enumerate(model.classes_)}
 for j,c in enumerate(labels):
  if c in pos:out[:,j]=z[:,pos[c]]
 return out
n=len(va);df=[];hardd=np.empty(n,dtype=object);hardb=np.empty(n,dtype=object)
for component,classes in [('coarse',list(GROUP_TO_LABELS)),('coordinate',GROUP_TO_LABELS['coordinate']),('execute',GROUP_TO_LABELS['execute']),('explore',GROUP_TO_LABELS['explore']),('modify',GROUP_TO_LABELS['modify'])]:
 block=np.zeros((n,len(classes)),dtype='float32')
 for kind in ('sim','au'):
  ix=np.flatnonzero(src[va]==kind);model=dm[kind]['coarse_model'] if component=='coarse' else dm[kind]['fine_models'][component];block[ix]=dec(model,[b[va[i]] for i in ix],classes)
 df.append(block)
bb=np.zeros((n,14),dtype='float32')
for kind in ('sim','au'):
 ix=np.flatnonzero(src[va]==kind);bb[ix]=dec(bm[kind],[b[va[i]] for i in ix],ALL_CLASSES);hardd[ix]=predict_two_stage(dm[kind],[b[va[i]] for i in ix]);hardb[ix]=bm[kind].predict([b[va[i]] for i in ix])
df.append(bb);df.append(np.column_stack([hardd==c for c in ALL_CLASSES]).astype('float32'));df.append(np.column_stack([hardb==c for c in ALL_CLASSES]).astype('float32'));extra=np.column_stack([ef]+df);x=np.column_stack((np.load('experiments/modeling_q/prepared_inputs.npz',allow_pickle=False)['x'],extra)).astype('float32');print('x',x.shape,flush=True);pred=np.empty(n,dtype=int)
for k,(a,c) in enumerate(GroupKFold(5).split(x,y,groups),1):
 cnt=np.bincount(y[a],minlength=14);w=(len(a)/(14*cnt[y[a]]));m=XGBClassifier(n_estimators=1400,max_depth=10,learning_rate=.04,subsample=.85,colsample_bytree=.85,objective='multi:softprob',num_class=14,tree_method='hist',device='cuda',eval_metric='mlogloss',early_stopping_rounds=100,n_jobs=8,random_state=42+k);m.fit(x[a],y[a],sample_weight=w,eval_set=[(x[c],y[c])],verbose=False);pred[c]=m.predict(x[c]);print(k,m.best_iteration,flush=True)
s=f1_score(ystr,np.asarray(ALL_CLASSES)[pred],labels=ALL_CLASSES,average='macro',zero_division=0);print('Y_source_all',s,flush=True);out=Path('experiments/modeling_y_source_all');out.mkdir(exist_ok=True);(out/'score.txt').write_text(str(s)+'\n')
