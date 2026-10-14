#!/usr/bin/env python3
"""Fast selector-only resume for Modeling N's already materialized OOF features."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupKFold
from train_modeling_b import ALL_CLASSES

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',default='data'); p.add_argument('--oof-dir',default='experiments/modeling_n_refit'); p.add_argument('--out-dir',default='experiments/modeling_n_refit'); a=p.parse_args()
    x=np.load(Path(a.oof_dir)/'oof_stack_features.npy'); labels={r['id']:r['action'] for r in csv.DictReader(open(Path(a.data_dir)/'train_labels.csv',encoding='utf8'))}
    ids=[json.loads(line)['id'] for line in open(Path(a.data_dir)/'train.jsonl',encoding='utf8') if line.strip()]; y=np.asarray([labels[i] for i in ids]); groups=np.asarray([i.split('-step_',1)[0] for i in ids]); folds=list(GroupKFold(5).split(x,y,groups))
    candidates={
      'N_lr_c01':LogisticRegression(C=0.1, max_iter=300, class_weight='balanced', n_jobs=-1, multi_class='ovr'),
      'N_lr_c1':LogisticRegression(C=1.0, max_iter=300, class_weight='balanced', n_jobs=-1, multi_class='ovr'),
    }
    rows=[]
    for name, proto in candidates.items():
      pred=np.empty(len(y),dtype=object); fold_scores=[]
      for tr,va in folds:
        model=proto.__class__(**proto.get_params()); model.fit(x[tr],y[tr]); pred[va]=model.predict(x[va]); fold_scores.append(f1_score(y[va],pred[va],labels=ALL_CLASSES,average='macro',zero_division=0))
      score=f1_score(y,pred,labels=ALL_CLASSES,average='macro',zero_division=0); print(f'{name} oof_macro_f1={score:.6f} folds={fold_scores}',flush=True); rows.append((name,score,pred,proto))
    best=max(rows,key=lambda r:r[1]); out=Path(a.out_dir); (out/'selector_resume_results.csv').write_text('experiment,oof_macro_f1\n'+'\n'.join(f'{n},{s}' for n,s,_,_ in rows)+'\n',encoding='utf8'); (out/'N_fast_classification_report.txt').write_text(classification_report(y,best[2],labels=ALL_CLASSES,digits=4,zero_division=0),encoding='utf8'); print(f'BEST={best[0]} score={best[1]:.6f}')
if __name__=='__main__': main()
