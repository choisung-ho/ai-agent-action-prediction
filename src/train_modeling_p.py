#!/usr/bin/env python3
"""Modeling P: expanded structured state for the proven E hierarchy."""
from __future__ import annotations

import argparse, csv, json, time
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from train_modeling_b import ALL_CLASSES
from train_modeling_e import TREE, child_for_label, descendants, predict_tree, result_status, bucket


def args_():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',default='data'); p.add_argument('--split-path',default='splits/common_split.csv'); p.add_argument('--out-dir',default='experiments/modeling_p'); return p.parse_args()

def build(samples):
    texts=[]
    for s in samples:
        history=s.get('history') if isinstance(s.get('history'),list) else []
        actions=[x for x in history if isinstance(x,dict) and x.get('role')=='assistant_action']
        parts=['[CURRENT] '+str(s.get('current_prompt') or '').strip().lower(), f'hlen_{len(history)}', f'action_count_{len(actions)}']
        for pos, action in enumerate(reversed(actions[-8:]),1):
            name=str(action.get('name') or 'none').lower(); status=result_status(action.get('result_summary'))
            parts += [f'pa{pos}_{name}', f'pr{pos}_{status}', f'pair{pos}_{name}_{status}']
            raw=str(action.get('result_summary') or '').lower()
            for token in ('error','traceback','not found','fail','pass','patched','modified','listed','found','0 match','0 file'):
                if token in raw: parts.append(f'pres{pos}_{token.replace(" ","_")}')
            value=action.get('args',{})
            if isinstance(value,dict):
                for key, item in value.items():
                    if key in {'path','target','scope','pattern','cmd','n_files','query'}:
                        suffix=Path(str(item)).suffix.lower().lstrip('.') or 'none'
                        parts += [f'arg{pos}_{key}',f'ext{pos}_{suffix}']
        names=[str(a.get('name') or 'none').lower() for a in actions]
        for width in (2,3,4):
            for index in range(max(0,len(names)-width+1)):
                parts.append('seq'+str(width)+'_'+'_'.join(names[index:index+width]))
        meta=s.get('session_meta') if isinstance(s.get('session_meta'),dict) else {}; workspace=meta.get('workspace') if isinstance(meta.get('workspace'),dict) else {}
        language=workspace.get('language_mix') if isinstance(workspace.get('language_mix'),dict) else {}
        lang='_'.join(sorted(language,key=language.get,reverse=True)[:2]) if language else 'none'
        open_files=workspace.get('open_files') if isinstance(workspace.get('open_files'),list) else []
        parts += [f'turn_{meta.get("turn_index","unknown")}', f'tier_{meta.get("user_tier","unknown")}', f'pref_{meta.get("language_pref","unknown")}', f'lang_{lang}', f'git_{workspace.get("git_dirty","unknown")}', f'ci_{workspace.get("last_ci_status","unknown")}', f'open_{len(open_files)}', f'loc_{bucket(workspace.get("loc"),[(2000,"xs"),(8000,"s"),(20000,"m"),(50000,"l")])}', f'budget_{bucket(meta.get("budget_tokens_remaining"),[(20000,"xs"),(60000,"s"),(120000,"m"),(170000,"l")])}']
        texts.append(' '.join(parts))
    return texts

def fit(texts, labels, c, ngrams):
    models={}; defaults={}
    for node in TREE:
        allowed=descendants(node); idx=[i for i,y in enumerate(labels) if y in allowed]; targets=[child_for_label(node,labels[i]) for i in idx]
        defaults[node]=Counter(targets).most_common(1)[0][0]
        model=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,ngrams),min_df=2,max_features=300000,sublinear_tf=True,lowercase=False)),('clf',LinearSVC(C=c,class_weight='balanced'))])
        model.fit([texts[i] for i in idx],targets); models[node]=model
    return {'model_type':'hierarchical_tree_p1','tree':TREE,'models':models,'defaults':defaults}

def main():
    a=args_(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    samples=[json.loads(line) for line in open(Path(a.data_dir)/'train.jsonl',encoding='utf8') if line.strip()]
    labels={r['id']:r['action'] for r in csv.DictReader(open(Path(a.data_dir)/'train_labels.csv',encoding='utf8'))}; split={r['id']:r['split'] for r in csv.DictReader(open(a.split_path,encoding='utf8'))}
    texts=build(samples); y=[labels[s['id']] for s in samples]; tr=[i for i,s in enumerate(samples) if split[s['id']]=='train']; va=[i for i,s in enumerate(samples) if split[s['id']]=='val']
    for name,c,ngrams in [('P0',.25,3),('P1',.5,3),('P2',.7,4),('P3',1.,4)]:
        start=time.time(); m=fit([texts[i] for i in tr],[y[i] for i in tr],c,ngrams); p=predict_tree(m,[texts[i] for i in va]); score=f1_score([y[i] for i in va],p,labels=ALL_CLASSES,average='macro',zero_division=0)
        print(f'{name} macro_f1={score:.6f} seconds={time.time()-start:.1f}',flush=True); (out/f'{name}_report.txt').write_text(classification_report([y[i] for i in va],p,labels=ALL_CLASSES,digits=4,zero_division=0),encoding='utf8')

if __name__=='__main__': main()
