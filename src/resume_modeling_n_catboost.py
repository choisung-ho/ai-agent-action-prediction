#!/usr/bin/env python3
"""GPU CatBoost selector evaluation over Modeling N leakage-safe OOF features."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score,classification_report
from sklearn.model_selection import GroupKFold
from train_modeling_b import ALL_CLASSES

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--oof-dir',default='experiments/modeling_n_refit');p.add_argument('--out-dir',default='experiments/modeling_n_refit');a=p.parse_args()
 x=np.load(Path(a.oof_dir)/'oof_stack_features.npy'); labels={r['id']:r['action'] for r in csv.DictReader(open(Path(a.data_dir)/'train_labels.csv',encoding='utf8'))};ids=[json.loads(l)['id'] for l in open(Path(a.data_dir)/'train.jsonl',encoding='utf8') if l.strip()];y=np.array([labels[i] for i in ids]);groups=np.array([i.split('-step_',1)[0] for i in ids]);folds=list(GroupKFold(5).split(x,y,groups));pred=np.empty(len(y),dtype=object)
 for n,(tr,va) in enumerate(folds,1):
  model=CatBoostClassifier(iterations=700,depth=8,learning_rate=.08,loss_function='MultiClass',eval_metric='TotalF1:average=Macro',auto_class_weights='Balanced',l2_leaf_reg=6,random_seed=42,verbose=100,task_type='GPU',devices='0',allow_writing_files=False)
  model.fit(x[tr],y[tr],eval_set=(x[va],y[va]),early_stopping_rounds=100,use_best_model=True);pred[va]=model.predict(x[va]).reshape(-1);print(f'fold={n} score={f1_score(y[va],pred[va],labels=ALL_CLASSES,average="macro",zero_division=0):.6f}',flush=True)
 score=f1_score(y,pred,labels=ALL_CLASSES,average='macro',zero_division=0);print(f'N_catboost oof_macro_f1={score:.6f}',flush=True);out=Path(a.out_dir);(out/'N_catboost_report.txt').write_text(classification_report(y,pred,labels=ALL_CLASSES,digits=4,zero_division=0),encoding='utf8');(out/'N_catboost_score.txt').write_text(str(score)+'\n',encoding='utf8')
if __name__=='__main__':main()
