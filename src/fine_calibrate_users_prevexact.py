#!/usr/bin/env python3
import csv,json,random,numpy as np
from pathlib import Path
labels={r['id']:r['action'] for r in csv.DictReader(open('data/train_labels.csv',encoding='utf8'))};split={r['id']:r['split'] for r in csv.DictReader(open('splits/common_split.csv',encoding='utf8'))};rows=[json.loads(l) for l in open('data/train.jsonl',encoding='utf8') if l.strip()];rows=[s for s in rows if split[s['id']]=='val'];cls=['read_file','grep_search','list_directory','glob_pattern','edit_file','write_file','apply_patch','run_bash','run_tests','lint_or_typecheck','ask_user','plan_task','web_search','respond_only'];toid={c:i for i,c in enumerate(cls)};y=np.asarray([toid[labels[s['id']]] for s in rows]);p=np.load('experiments/modeling_ac_tune2/oof_probabilities.npy');actions=['none']+cls;aid={c:i for i,c in enumerate(actions)}
def prev(s):
 h=s.get('history') if isinstance(s.get('history'),list) else [];a=[str(x.get('name') or x.get('action') or 'none') for x in h if isinstance(x,dict) and x.get('role')=='assistant_action'];return aid.get(a[-1] if a else 'none',0)
g=np.asarray([prev(s) for s in rows]);base=json.loads(Path('experiments/modeling_ac_tune2/fine_calibration.json').read_text())['weights'];w=np.log(np.asarray([[base[c] for c in cls] for _ in actions]))
def score(w):
 z=np.argmax(p*np.exp(w[g]),1);cm=np.zeros((14,14),dtype=np.int64);np.add.at(cm,(y,z),1);tp=np.diag(cm);den=2*tp+cm.sum(0)-tp+cm.sum(1)-tp;return np.mean(np.divide(2*tp,den,out=np.zeros(14),where=den>0))
best=score(w);random.seed(42)
for it in range(100000):
 i=random.randrange(len(actions));j=random.randrange(14);old=w[i,j];d=random.choice((-.2,-.12,-.08,-.05,-.03,-.02,.02,.03,.05,.08,.12,.2));w[i,j]=old+d;v=score(w)
 if v>=best:best=v
 else:w[i,j]=old
 if it%10000==0:print(it,best,flush=True)
print('BEST',best,flush=True);Path('experiments/modeling_ac_tune2/prevexact_calibration.json').write_text(json.dumps({'score':float(best),'actions':actions,'weights':w.tolist()},indent=2))
