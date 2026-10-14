#!/usr/bin/env python3
"""Validate Q state selector plus source-routed E margins on the frozen split."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,classification_report
from sklearn.model_selection import GroupKFold
from data import load_jsonl,load_labels
from train_modeling_b import ALL_CLASSES
from train_modeling_e import build_modeling_e_features
from train_modeling_r_source_stack import source_of,fit_source_models,source_e_features

p=argparse.ArgumentParser();p.add_argument('--trees',type=int,default=300);p.add_argument('--leaf',type=int,default=10);p.add_argument('--out-dir',default='experiments/modeling_s_q_source');a=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
samples=load_jsonl('data/train.jsonl');labels=load_labels('data/train_labels.csv');split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};ids,texts=build_modeling_e_features(samples);tr=np.asarray([i for i,x in enumerate(ids) if split[x]=='train']);va=np.asarray([i for i,x in enumerate(ids) if split[x]=='val']);y=np.asarray([labels[ids[i]] for i in va]);groups=np.asarray([ids[i].split('-step_',1)[0] for i in va]);sources=np.asarray([source_of(x) for x in ids])
q=np.load('experiments/modeling_q/prepared_inputs.npz',allow_pickle=False);xq=q['x'];assert len(xq)==len(va)
models=fit_source_models([texts[i] for i in tr],np.asarray([labels[ids[i]] for i in tr]),sources[tr]);extra,schema,_=source_e_features(models,[texts[i] for i in va],sources[va]);x=np.column_stack((xq,extra)).astype(np.float32);print(f'q={xq.shape} source={extra.shape} combined={x.shape}',flush=True)
pred=np.empty(len(y),dtype=object);scores=[]
for n,(i,j) in enumerate(GroupKFold(5).split(x,y,groups),1):
 m=RandomForestClassifier(n_estimators=a.trees,min_samples_leaf=a.leaf,max_features=.7,class_weight='balanced_subsample',n_jobs=8,random_state=42).fit(x[i],y[i]);pred[j]=m.predict(x[j]);s=f1_score(y[j],pred[j],labels=ALL_CLASSES,average='macro',zero_division=0);scores.append(s);print(f'fold={n} score={s:.6f}',flush=True)
score=f1_score(y,pred,labels=ALL_CLASSES,average='macro',zero_division=0);print(f'S_q_plus_source_e={score:.6f} folds={scores}',flush=True);(out/'summary.json').write_text(json.dumps({'macro_f1':score,'folds':scores,'q_shape':list(xq.shape),'source_shape':list(extra.shape)},indent=2)+'\n');(out/'report.txt').write_text(classification_report(y,pred,labels=ALL_CLASSES,digits=4,zero_division=0))
