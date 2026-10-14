#!/usr/bin/env python3
"""Generate full-data stacked-selector variants from the same saved OOF base features."""
import argparse,csv,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import RandomForestClassifier

p=argparse.ArgumentParser();p.add_argument('--trees',type=int,default=300);p.add_argument('--leaf',type=int,default=10);p.add_argument('--out-dir',default='model_candidate_rf300');a=p.parse_args()
labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};ids=[json.loads(l)['id'] for l in open('data/train.jsonl',encoding='utf8') if l.strip()];y=[labels[x] for x in ids];x=np.load('experiments/modeling_n_refit/oof_stack_features.npy')
bundle=joblib.load('model_candidate_full/best_model.pkl');print(f'fit RF trees={a.trees} leaf={a.leaf}',flush=True);bundle['selector']=RandomForestClassifier(n_estimators=a.trees,min_samples_leaf=a.leaf,max_features=.7,class_weight='balanced_subsample',n_jobs=-1,random_state=42).fit(x,y)
out=Path(a.out_dir);out.mkdir(exist_ok=True);joblib.dump(bundle,out/'best_model.pkl',compress=3);meta=json.loads(Path('model_candidate_full/best_model_metadata.json').read_text(encoding='utf8'));meta['best_experiment']=f'N_full_refit_rf{a.trees}_leaf{a.leaf}';(out/'best_model_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('done',flush=True)
