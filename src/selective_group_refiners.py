#!/usr/bin/env python3
import csv,json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from train_modeling_b import ALL_CLASSES
from train_modeling_o import rich_text
labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};samples=[json.loads(l) for l in open('data/train.jsonl',encoding='utf8') if l.strip()];tr=[i for i,s in enumerate(samples) if split[s['id']]=='train'];va=[i for i,s in enumerate(samples) if split[s['id']]=='val'];y=np.asarray([labels[samples[i]['id']] for i in va]);p=np.load('experiments/modeling_ac_tune2/oof_probabilities.npy');pred=np.asarray(ALL_CLASSES)[p.argmax(1)];sort=np.sort(p,axis=1);margin=sort[:,-1]-sort[:,-2]
def sc(a):return f1_score(y,a,labels=ALL_CLASSES,average='macro',zero_division=0)
texts=[rich_text(s,history_n=6,include_results=True,include_meta=True) for s in samples];groups=[['read_file','grep_search','list_directory','glob_pattern'],['run_bash','run_tests','lint_or_typecheck'],['ask_user','plan_task','web_search'],['edit_file','apply_patch']];thresholds=[0,.002,.005,.01,.02,.03,.05,.08,.12,.2]
print('base',sc(pred),flush=True)
for g in groups:
 idx=[i for i in tr if labels[samples[i]['id']] in g];m=Pipeline([('f',FeatureUnion([('w',TfidfVectorizer(ngram_range=(1,3),min_df=2,max_features=150000,sublinear_tf=True)),('c',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=150000,sublinear_tf=True))])),('s',LinearSVC(C=.5,class_weight='balanced'))]);m.fit([texts[i] for i in idx],[labels[samples[i]['id']] for i in idx]);gp=m.predict([texts[i] for i in va]);best=(sc(pred),None,pred)
 for t in thresholds:
  cand=pred.copy();mask=np.isin(cand,g)&(margin<=t);cand[mask]=gp[mask];v=sc(cand)
  if v>best[0]:best=(v,t,cand)
 print(g,best[0],best[1],flush=True);pred=best[2]
Path('experiments/selective_group_refiners').mkdir(exist_ok=True);Path('experiments/selective_group_refiners/score.txt').write_text(str(sc(pred))+'\n')
