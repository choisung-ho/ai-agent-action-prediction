#!/usr/bin/env python3
"""Fast GPU linear/MLP probes over frozen Qwen embeddings (experiment-only)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score

from train_modeling_b import ALL_CLASSES


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-dir", default="experiments/qwen15_embeddings")
    p.add_argument("--out-dir", default="experiments/qwen15_head")
    p.add_argument("--hidden", type=int, default=0, help="0 is linear; otherwise a GELU MLP")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class Head(torch.nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = torch.nn.Linear(input_dim, len(ALL_CLASSES)) if hidden <= 0 else torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, hidden), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden, len(ALL_CLASSES)),
        )
    def forward(self, x): return self.net(x)


def main():
    a = args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    root=Path(a.embedding_dir); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    x=np.asarray(np.load(root/'qwen_embeddings.npy',mmap_mode='r'),dtype=np.float32)
    labels=np.load(root/'labels.npy'); split=np.load(root/'split.npy')
    lookup={label:i for i,label in enumerate(ALL_CLASSES)}; y=np.asarray([lookup[v] for v in labels])
    tr=np.flatnonzero(split=='train'); va=np.flatnonzero(split=='val')
    # Embeddings are individually L2-normalized; centering is fitted on train
    # only to avoid a subtle held-out distribution leak.
    mean=x[tr].mean(axis=0,keepdims=True); scale=x[tr].std(axis=0,keepdims=True).clip(1e-4)
    x=(x-mean)/scale
    dev='cuda'; xt=torch.tensor(x[tr],dtype=torch.float32,device=dev); yt=torch.tensor(y[tr],dtype=torch.long,device=dev); xv=torch.tensor(x[va],dtype=torch.float32,device=dev)
    counts=np.bincount(y[tr],minlength=len(ALL_CLASSES)); weights=torch.tensor(np.sqrt(len(tr)/(len(ALL_CLASSES)*counts)),dtype=torch.float32,device=dev)
    m=Head(x.shape[1],a.hidden,a.dropout).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    best=-1; remaining=a.patience; records=[]
    gen=torch.Generator(device=dev).manual_seed(a.seed)
    for epoch in range(1,a.epochs+1):
        m.train(); order=torch.randperm(len(tr),generator=gen,device=dev); loss=0
        for start in range(0,len(tr),a.batch_size):
            ix=order[start:start+a.batch_size]; opt.zero_grad(set_to_none=True); logits=m(xt[ix]); val=F.cross_entropy(logits,yt[ix],weight=weights,label_smoothing=.01); val.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();loss+=val.item()*len(ix)
        m.eval()
        with torch.inference_mode(): logits=m(xv).float().cpu().numpy()
        pred=logits.argmax(1); score=f1_score(labels[va],np.asarray(ALL_CLASSES)[pred],labels=ALL_CLASSES,average='macro',zero_division=0)
        record={'epoch':epoch,'macro_f1':score,'loss':loss/len(tr)};records.append(record);print(json.dumps(record),flush=True)
        if score>best:
            best=score;remaining=a.patience; torch.save({'state_dict':m.state_dict(),'mean':mean,'scale':scale,'args':vars(a),'score':score},out/'best.pt');np.savez_compressed(out/'best_val_logits.npz',logits=logits,target=y[va]);(out/'best_report.txt').write_text(classification_report(labels[va],np.asarray(ALL_CLASSES)[pred],labels=ALL_CLASSES,digits=4,zero_division=0),encoding='utf8')
        else:
            remaining-=1
            if remaining<=0:break
    (out/'results.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':main()
