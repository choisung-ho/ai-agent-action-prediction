#!/usr/bin/env python3
"""Build a full-data, leakage-safe I3-style candidate from N OOF features."""
import csv,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import RandomForestClassifier
from train_modeling_b import ALL_CLASSES,build_modeling_b_features,make_classifier
from train_modeling_d import fit_two_stage
from train_modeling_e import build_modeling_e_features,fit_tree

ROOT=Path('.')
samples=[json.loads(x) for x in open('data/train.jsonl',encoding='utf8') if x.strip()]
labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))}
ids,e=build_modeling_e_features(samples);_,b=build_modeling_b_features(samples,history_n=3,include_meta=True); y=[labels[i] for i in ids]
print('Refit E/D/B on all rows...',flush=True)
e_model=fit_tree(e,y,c=.5,ngram_max=3)
d_model=fit_two_stage(b,y,{'coarse_c':1.2,'fine_c':.8,'coarse_max_features':100000,'fine_max_features':80000})
b_model=make_classifier('B3');b_model.fit(b,y)
print('Fit selector on saved leakage-safe OOF features...',flush=True)
x=np.load('experiments/modeling_n_refit/oof_stack_features.npy')
selector=RandomForestClassifier(n_estimators=120,min_samples_leaf=10,max_features=.7,class_weight='balanced_subsample',n_jobs=-1,random_state=42).fit(x,y)
schema=json.loads(Path('experiments/modeling_n_refit/feature_schema.json').read_text(encoding='utf8'))
bundle={'model_type':'stacked_selector_v1','e_model':e_model,'d_model':d_model,'b_model':b_model,'selector':selector,'feature_schema':schema,'all_classes':ALL_CLASSES}
out=Path('model_candidate_full');out.mkdir(exist_ok=True);joblib.dump(bundle,out/'best_model.pkl',compress=3)
(out/'best_model_metadata.json').write_text(json.dumps({'best_experiment':'N_full_refit_rf120','best_macro_f1':None,'validation_method':'base models refit on all 70000; selector fit on leakage-safe 5-fold OOF features','feature_builder':'stacked_multi_v1','feature_config':{},'model_type':'stacked_selector_v1','refit_full':True,'all_classes':ALL_CLASSES},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('candidate written',flush=True)
