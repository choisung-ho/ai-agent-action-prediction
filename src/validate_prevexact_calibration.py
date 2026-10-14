#!/usr/bin/env python3
import csv,json,random,numpy as np
from sklearn.model_selection import GroupKFold
labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};rows=[json.loads(l) for l in open('data/train.jsonl',encoding='utf8') if l.strip()];rows=[s for s in rows if split[s['id']]=='val'];cls=['read_file','grep_search','list_directory','glob_pattern','edit_file','write_file','apply_patch','run_bash','run_tests','lint_or_typecheck','ask_user','plan_task','web_search','respond_only'];toid={c:i for i,c in enumerate(cls)};y=np.asarray([toid[labels[s['id']]] for s in rows]);p=np.load('experiments/modeling_ac_tune2/oof_probabilities.npy');actions=['none']+cls;aid={c:i for i,c in enumerate(actions)}
def prev(s):
 h=s.get('history') if isinstance(s.get('history'),list) else [];a=[str(x.get('name') or x.get('action') or 'none') for x in h if isinstance(x,dict) and x.get('role')=='assistant_action'];return aid.get(a[-1] if a else 'none',0)
g=np.asarray([prev(s) for s in rows]);groups=np.asarray([s['id'].split('-step_',1)[0] for s in rows])
def score(w,idx):
 z=np.argmax(p[idx]*np.exp(w[g[idx]]),1);cm=np.zeros((14,14),dtype=np.int64);np.add.at(cm,(y[idx],z),1);tp=np.diag(cm);den=2*tp+cm.sum(0)-tp+cm.sum(1)-tp;return np.mean(np.divide(2*tp,den,out=np.zeros(14),where=den>0))
allpred=np.empty(len(y),dtype=int)
for fold,(tr,te) in enumerate(GroupKFold(5).split(p,y,groups),1):
 w=np.zeros((15,14));best=score(w,tr);random.seed(100+fold)
 for it in range(25000):
  i=random.randrange(15);j=random.randrange(14);old=w[i,j];d=random.choice((-.2,-.12,-.08,-.05,-.03,-.02,.02,.03,.05,.08,.12,.2));w[i,j]=old+d;v=score(w,tr)
  if v>=best:best=v
  else:w[i,j]=old
 allpred[te]=np.argmax(p[te]*np.exp(w[g[te]]),1);print(fold,'train',best,'test',score(w,te),flush=True)
cm=np.zeros((14,14),dtype=np.int64);np.add.at(cm,(y,allpred),1);tp=np.diag(cm);den=2*tp+cm.sum(0)-tp+cm.sum(1)-tp;print('nested_oof',np.mean(np.divide(2*tp,den,out=np.zeros(14),where=den>0)),flush=True)
