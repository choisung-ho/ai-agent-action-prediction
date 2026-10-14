#!/usr/bin/env python3
import csv,json,random,numpy as np
from pathlib import Path
labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};ids=[json.loads(l)['id'] for l in open('data/train.jsonl',encoding='utf8') if l.strip()];ids=[i for i in ids if split[i]=='val'];cls=['read_file','grep_search','list_directory','glob_pattern','edit_file','write_file','apply_patch','run_bash','run_tests','lint_or_typecheck','ask_user','plan_task','web_search','respond_only'];toid={c:i for i,c in enumerate(cls)};y=np.asarray([toid[labels[i]] for i in ids]);p=np.load('experiments/modeling_ac_tune2/oof_probabilities.npy');base=json.loads(Path('experiments/modeling_ac_tune2/calibration.json').read_text())['weights'];w=np.log(np.asarray([base[c] for c in cls]))
def score(w):
 z=np.argmax(p*np.exp(w),1);cm=np.zeros((14,14),dtype=np.int64);np.add.at(cm,(y,z),1);tp=np.diag(cm);den=2*tp+cm.sum(0)-tp+cm.sum(1)-tp;return np.mean(np.divide(2*tp,den,out=np.zeros(14),where=den>0))
best=score(w);random.seed(42)
for it in range(8000):
 j=random.randrange(14);old=w[j];delta=random.choice((-.12,-.08,-.05,-.03,-.02,-.01,.01,.02,.03,.05,.08,.12));w[j]=old+delta;v=score(w)
 if v>=best:best=v
 else:w[j]=old
 if it%1000==0:print(it,best,flush=True)
print('BEST',best,np.exp(w),flush=True);Path('experiments/modeling_ac_tune2/fine_calibration.json').write_text(json.dumps({'score':float(best),'weights':dict(zip(cls,np.exp(w).tolist()))},indent=2))
