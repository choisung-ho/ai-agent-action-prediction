#!/usr/bin/env python3
"""GPU CatBoost over Q state/margins plus frozen multilingual E5 embeddings."""
import csv,json,os
from pathlib import Path
import numpy as np,torch
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score,classification_report
from sklearn.model_selection import GroupKFold
from transformers import AutoModel,AutoTokenizer
from train_modeling_b import ALL_CLASSES
from train_modeling_g import model_text

out=Path('experiments/modeling_u_q_e5');out.mkdir(parents=True,exist_ok=True);labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};samples=[json.loads(l) for l in open('data/train.jsonl',encoding='utf8') if l.strip()];rows=[s for s in samples if split[s['id']]=='val'];y=np.asarray([labels[s['id']] for s in rows]);groups=np.asarray([s['id'].split('-step_',1)[0] for s in rows]);xq=np.load('experiments/modeling_q/prepared_inputs.npz',allow_pickle=False)['x'];assert len(xq)==len(rows)
cache=out/'e5_val.npy'
if cache.exists(): emb=np.load(cache)
else:
 device='cuda' if torch.cuda.is_available() else 'cpu';tok=AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small');model=AutoModel.from_pretrained('intfloat/multilingual-e5-small').to(device).eval();parts=[]
 with torch.inference_mode():
  for start in range(0,len(rows),256):
   b=tok([model_text(s) for s in rows[start:start+256]],padding=True,truncation=True,max_length=128,return_tensors='pt');b={k:v.to(device) for k,v in b.items()};h=model(**b).last_hidden_state;m=b['attention_mask'].unsqueeze(-1);z=(h*m).sum(1)/m.sum(1).clamp(min=1);z=torch.nn.functional.normalize(z,p=2,dim=1);parts.append(z.cpu().numpy().astype('float32'));print('embed',start,flush=True)
 emb=np.vstack(parts);np.save(cache,emb)
x=np.column_stack((xq,emb)).astype('float32');print('x',x.shape,flush=True);pred=np.empty(len(y),dtype=object)
for n,(tr,va) in enumerate(GroupKFold(5).split(x,y,groups),1):
 m=CatBoostClassifier(iterations=900,depth=8,learning_rate=.06,loss_function='MultiClass',eval_metric='TotalF1:average=Macro',auto_class_weights='Balanced',l2_leaf_reg=7,random_seed=42+n,task_type='GPU',devices='0',allow_writing_files=False,verbose=False);m.fit(x[tr],y[tr],eval_set=(x[va],y[va]),early_stopping_rounds=100,use_best_model=True);pred[va]=m.predict(x[va]).reshape(-1);print('fold',n,f1_score(y[va],pred[va],labels=ALL_CLASSES,average='macro',zero_division=0),flush=True)
s=f1_score(y,pred,labels=ALL_CLASSES,average='macro',zero_division=0);print('U_q_e5',s,flush=True);(out/'report.txt').write_text(classification_report(y,pred,labels=ALL_CLASSES,digits=4,zero_division=0));(out/'summary.json').write_text(json.dumps({'macro_f1':s,'shape':list(x.shape)},indent=2))
