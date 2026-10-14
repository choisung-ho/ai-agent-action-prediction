#!/usr/bin/env python3
"""Build full-data OOF user/action/source XGBoost submission candidate."""
import csv,json,sys
from pathlib import Path
import joblib,numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from data import load_jsonl,load_labels
from train_modeling_e import build_modeling_e_features
from train_modeling_r_source_stack import source_of,fit_source_models,source_e_features
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from script import build_state_dict,build_user_sequence_hash,ALL_CLASSES_LIST

samples=load_jsonl('data/train.jsonl');labels_map=load_labels('data/train_labels.csv');ids,e_texts=build_modeling_e_features(samples);labels=np.asarray([labels_map[i] for i in ids]);toid={c:i for i,c in enumerate(ALL_CLASSES_LIST)};y=np.asarray([toid[c] for c in labels]);sources=np.asarray([source_of(i) for i in ids]);groups=np.asarray([i.split('-step_',1)[0] for i in ids])
base=np.load('experiments/modeling_n_refit/oof_stack_features.npy');bundle=joblib.load('model_candidate_q/best_model.pkl');state=bundle['state_vectorizer'].transform([build_state_dict(s) for s in samples]).astype('float32');print('base/state',base.shape,state.shape,flush=True)
folds=list(GroupKFold(5).split(np.zeros(len(ids)),y,groups));source_oof=None;schema=None
for n,(tr,va) in enumerate(folds,1):
 models=fit_source_models([e_texts[i] for i in tr],labels[tr],sources[tr]);current,current_schema,_=source_e_features(models,[e_texts[i] for i in va],sources[va])
 if source_oof is None:source_oof=np.zeros((len(ids),current.shape[1]),dtype='float32');schema=current_schema
 source_oof[va]=current;print('source fold',n,flush=True)
full_source_models=fit_source_models(e_texts,labels,sources);user=build_user_sequence_hash(samples);x=np.column_stack((base,state,source_oof,user)).astype('float32');print('x',x.shape,flush=True)
counts=np.bincount(y,minlength=14);weight=(len(y)/(14*counts[y]));selector=XGBClassifier(n_estimators=210,max_depth=10,learning_rate=.04,subsample=.85,colsample_bytree=.85,objective='multi:softprob',num_class=14,tree_method='hist',device='cuda',eval_metric='mlogloss',n_jobs=8,random_state=42);selector.fit(x,y,sample_weight=weight,verbose=False)
cal=json.loads(Path('experiments/modeling_ac_tune2/source_calibration.json').read_text());weights={'sim':[cal['sim'][c] for c in ALL_CLASSES_LIST],'au':[cal['au'][c] for c in ALL_CLASSES_LIST]}
candidate=dict(bundle);candidate.update({'model_type':'stacked_xgb_user_source_v1','selector':selector,'source_e_models':full_source_models,'source_e_schema':schema,'selector_feature_count':int(x.shape[1]),'source_class_weights':weights})
out=Path('model_candidate_user_xgb');out.mkdir(exist_ok=True);joblib.dump(candidate,out/'best_model.pkl',compress=3)
meta={'best_experiment':'AC_full_oof_user_action_source_xgb','best_macro_f1':0.742448245417774,'validation_method':'fixed common validation OOF selector with source-conditional macro-F1 calibration; full selector fit on 70k leakage-safe OOF base/source margins plus inference-safe state/hash features','feature_builder':'stacked_xgb_user_source_v1','model_type':'stacked_xgb_user_source_v1','refit_full':True,'selector_feature_count':int(x.shape[1]),'all_classes':ALL_CLASSES_LIST}
(out/'best_model_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('candidate',out,flush=True)
