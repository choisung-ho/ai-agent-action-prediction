#!/usr/bin/env python3
"""CUDA fine-tuning of E5-small using full recent conversational context."""
import argparse,csv,json,random,time
from collections import Counter
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset,DataLoader
from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup
from sklearn.metrics import f1_score,classification_report
from train_modeling_b import ALL_CLASSES

def text(s):
 h=s.get('history') if isinstance(s.get('history'),list) else [];parts=['[CURRENT] '+str(s.get('current_prompt') or '')]
 for x in h[-8:]:
  if not isinstance(x,dict):continue
  role=x.get('role','')
  if role=='user':parts.append('[USER] '+str(x.get('content') or ''))
  elif role=='assistant_action':parts.append('[ACTION] '+str(x.get('name') or '')+' [ARGS] '+json.dumps(x.get('args') or {},ensure_ascii=False)+' [RESULT] '+str(x.get('result_summary') or ''))
 meta=s.get('session_meta') if isinstance(s.get('session_meta'),dict) else {};work=meta.get('workspace') if isinstance(meta.get('workspace'),dict) else {};parts.append('[STATE] tier='+str(meta.get('user_tier'))+' turn='+str(meta.get('turn_index'))+' git='+str(work.get('git_dirty'))+' ci='+str(work.get('last_ci_status')))
 return 'query: '+' '.join(parts)
class DS(Dataset):
 def __init__(self,x,y):self.x=x;self.y=y
 def __len__(self):return len(self.x)
 def __getitem__(self,i):return self.x[i],self.y[i]
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.e=AutoModel.from_pretrained('intfloat/multilingual-e5-small');self.d=nn.Dropout(.15);self.h=nn.Linear(self.e.config.hidden_size,14)
 def forward(self,**b):
  o=self.e(**b).last_hidden_state;m=b['attention_mask'].unsqueeze(-1);z=(o*m).sum(1)/m.sum(1).clamp(min=1);z=F.normalize(z,p=2,dim=1);return self.h(self.d(z))
p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=6);p.add_argument('--out-dir',default='experiments/modeling_x_rich');a=p.parse_args();random.seed(42);np.random.seed(42);torch.manual_seed(42);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
samples=[json.loads(l) for l in open('data/train.jsonl',encoding='utf8') if l.strip()];labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};toid={c:i for i,c in enumerate(ALL_CLASSES)};tr=[i for i,s in enumerate(samples) if split[s['id']]=='train'];va=[i for i,s in enumerate(samples) if split[s['id']]=='val'];tx=[text(s) for s in samples];y=np.asarray([toid[labels[s['id']]] for s in samples]);tok=AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
def coll(rows):
 x,z=zip(*rows);b=tok(list(x),padding=True,truncation=True,max_length=256,return_tensors='pt');b['labels']=torch.tensor(z,dtype=torch.long);return b
train=DataLoader(DS([tx[i] for i in tr],y[tr]),batch_size=48,shuffle=True,collate_fn=coll,num_workers=0);valid=DataLoader(DS([tx[i] for i in va],y[va]),batch_size=128,shuffle=False,collate_fn=coll,num_workers=0);dev=torch.device('cuda');m=Net().to(dev);counts=np.bincount(y[tr],minlength=14);weights=torch.tensor(np.sqrt(len(tr)/(14*counts)),device=dev,dtype=torch.float32);opt=torch.optim.AdamW([{'params':m.e.parameters(),'lr':1.5e-5},{'params':m.h.parameters(),'lr':8e-4}],weight_decay=.01);sch=get_linear_schedule_with_warmup(opt,int(len(train)*a.epochs*.05),len(train)*a.epochs);scaler=torch.amp.GradScaler('cuda');best=0;rows=[]
for ep in range(1,a.epochs+1):
 m.train();st=time.time()
 for b in train:
  yy=b.pop('labels').to(dev);b={k:v.to(dev) for k,v in b.items()};opt.zero_grad(set_to_none=True)
  with torch.amp.autocast('cuda'): loss=F.cross_entropy(m(**b),yy,weight=weights)
  scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(m.parameters(),1.);scaler.step(opt);scaler.update();sch.step()
 m.eval();pp=[];yy=[]
 with torch.no_grad():
  for b in valid:
   target=b.pop('labels');b={k:v.to(dev) for k,v in b.items()};pp.extend(m(**b).argmax(1).cpu().tolist());yy.extend(target.tolist())
 s=f1_score(np.asarray(ALL_CLASSES)[yy],np.asarray(ALL_CLASSES)[pp],labels=ALL_CLASSES,average='macro',zero_division=0);rows.append({'epoch':ep,'macro_f1':s});print('epoch',ep,s,time.time()-st,flush=True);(out/f'epoch_{ep}_report.txt').write_text(classification_report(np.asarray(ALL_CLASSES)[yy],np.asarray(ALL_CLASSES)[pp],labels=ALL_CLASSES,digits=4,zero_division=0))
 if s>best:
  best=s
  # Unique checkpoint names avoid transient OneDrive file-handle collisions.
  torch.save({'encoder':m.e.state_dict(),'head':m.h.state_dict(),'score':s},out/f'best_epoch_{ep}.pt')
  tok.save_pretrained(out/'tokenizer')
(out/'results.json').write_text(json.dumps(rows,indent=2))
