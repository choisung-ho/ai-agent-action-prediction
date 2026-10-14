#!/usr/bin/env python3
"""Evaluate pair-specific word/character refiners on Q OOF predictions."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from train_modeling_b import ALL_CLASSES
from train_modeling_e import build_modeling_e_features

labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};samples=[json.loads(l) for l in open('data/train.jsonl',encoding='utf8') if l.strip()];ids,texts=build_modeling_e_features(samples);tr=[i for i,x in enumerate(ids) if split[x]=='train'];va=[i for i,x in enumerate(ids) if split[x]=='val'];y=np.asarray([labels[ids[i]] for i in va]);proba=np.load('experiments/modeling_q_calibration/oof_probabilities.npy');base=np.asarray(ALL_CLASSES)[np.argmax(proba,axis=1)];
pairs=[('read_file','list_directory'),('grep_search','glob_pattern'),('run_tests','lint_or_typecheck'),('ask_user','plan_task'),('edit_file','apply_patch'),('run_bash','run_tests'),('web_search','ask_user')]
def score(p):return f1_score(y,p,labels=ALL_CLASSES,average='macro',zero_division=0)
print('base',score(base),flush=True);current=base.copy();rows=[]
for a,b in pairs:
 idx=[i for i in tr if labels[ids[i]] in {a,b}];model=Pipeline([('f',FeatureUnion([('w',TfidfVectorizer(ngram_range=(1,3),min_df=2,max_features=100000,sublinear_tf=True)),('c',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=100000,sublinear_tf=True))])),('s',LinearSVC(C=.5,class_weight='balanced'))]);model.fit([texts[i] for i in idx],[labels[ids[i]] for i in idx]);pair=model.predict([texts[i] for i in va]);cand=current.copy();mask=np.isin(cand,[a,b]);cand[mask]=pair[mask];s=score(cand);rows.append((a,b,s,int(mask.sum())));print(a,b,s,mask.sum(),flush=True)
 if s>score(current):current=cand
print('greedy',score(current),flush=True);out=Path('experiments/pairwise_q_refine');out.mkdir(exist_ok=True);(out/'results.json').write_text(json.dumps({'base':score(base),'greedy':score(current),'rows':rows},indent=2))
