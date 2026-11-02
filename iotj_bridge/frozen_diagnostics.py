#!/usr/bin/env python3
"""Frozen local critic/adapter/replay gradient diagnostics on IEEE AC case39/118.
Conditional empirical estimates only; not uniform theoretical bounds.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
try:
 from .core import Client,flat,assign,aggregate
 from .run_experiment import Grid
except ImportError:
 from core import Client,flat,assign,aggregate
 from run_experiment import Grid

def reference(c):
    items=c.data
    s=torch.tensor(np.stack([r[0] for r in items]),dtype=torch.float32)
    a=torch.tensor(np.array([r[1] for r in items]),dtype=torch.float32).reshape(-1,1)
    with torch.no_grad():
        adv=(c.q(s,a)-c.q(s,c.actor(s))).flatten()
        idx=(adv>0).nonzero(as_tuple=False).flatten()
    if not len(idx):return None
    for p in c.backbone.parameters():
        if p.grad is not None:p.grad.zero_()
    w=adv[idx];w=w/(w.sum()+1e-12)
    loss=-(w*c.q(s[idx],c.actor(s[idx])).flatten()).sum()
    g=torch.autograd.grad(loss,list(c.backbone.parameters()))
    return torch.cat([v.detach().flatten() for v in g]).numpy()

def run(case=39,seed=42,episodes=12,draws=16,clients_n=5):
    torch.set_num_threads(1)
    env=Grid(case,clients_n,seed,'dim')
    obs=env.reset()
    clients=[Client(len(o),seed*100+i) for i,o in enumerate(obs)]
    theta=flat(clients[0].backbone)
    for c in clients:assign(c.backbone,theta)
    globalproto=None
    for ep in range(episodes):
        obs=env.reset();done=False;step=0
        while not done:
            actions=[]
            for i,(c,o) in enumerate(zip(clients,obs)):
                with torch.no_grad():nom=float(c.actor(torch.tensor(o[None,:],dtype=torch.float32))[0,0])
                noise=np.random.default_rng(seed*100000+ep*1000+step*clients_n+i).normal(0,.12)
                actions.append(float(np.clip(nom+noise,0,1)))
            nxt,rewards,done,info=env.step(actions)
            cat='stable' if info['safe'] and info['restored_pct']>=95 else ('recovering' if info['restored_pct']>=75 else 'unsafe')
            for i,c in enumerate(clients):c.append(obs[i],actions[i],rewards[i],nxt[i],done,cat)
            obs=nxt;step+=1
        uploads=[]
        for i,c in enumerate(clients):
            c.adapt(seed*1000+ep*10+i,globalproto,steps=1,batch_size=16)
            uploads.append(c.share(theta,seed*1000+ep*10+i,mode='adv',steps=2,batch_size=16))
        theta,globalproto,_,_=aggregate(theta,uploads,globalproto)
        for c in clients:assign(c.backbone,theta)
    # Single frozen snapshot: no more critic/encoder/local buffer updates.
    refs=[reference(c) for c in clients]
    if any(r is None for r in refs):
        return {'case':case,'seed':seed,'status':'reference_high_advantage_set_empty'}
    gref=np.mean(np.stack(refs),axis=0)
    directions=[];subterms=[]
    eff_lr=2*clients[0].lr
    d=len(theta)
    for repetition in range(draws):
        ms=[c.share(theta,seed*4000+repetition*clients_n+i,steps=2,batch_size=32) for i,c in enumerate(clients)]
        if any(m is None for m in ms):continue
        dense=np.mean(np.stack([m['full_delta'] for m in ms]),axis=0)
        sparse=[]
        clipped=[]
        for m in ms:
            a=np.zeros(d,np.float32);a[m['idx']]=m['val'];sparse.append(a)
            clipped.append(a*min(1.,.075/max(np.linalg.norm(a),1e-12)))
        spart=np.mean(np.stack(sparse),axis=0);cpart=np.mean(np.stack(clipped),axis=0)
        new,_,weights,_=aggregate(theta,ms,globalproto)
        weighted=(new-theta).numpy()
        direction=-weighted/eff_lr
        directions.append(direction)
        subterms.append({'adam_local_drift_l2':float(np.linalg.norm(-dense/eff_lr-gref)),
             'sparsification_l2':float(np.linalg.norm(spart-dense)/eff_lr),
             'clipping_l2':float(np.linalg.norm(cpart-spart)/eff_lr),
             'adaptive_weighting_l2':float(np.linalg.norm(weighted-cpart)/eff_lr)})
    if len(directions)<2:return {'case':case,'seed':seed,'status':'insufficient_replicates','count':len(directions)}
    arr=np.stack(directions);mean=arr.mean(0)
    out={'case':case,'seed':seed,'status':'ok','episodes':episodes,'draws':len(directions),
         'reference_gradient_l2':float(np.linalg.norm(gref)),
         'estimated_bias_l2':float(np.linalg.norm(mean-gref)),
         'estimated_conditional_variance_trace':float(np.var(arr,axis=0,ddof=1).sum()),
         'effective_direction_scale':eff_lr,
         'diagnostic_scope':'fixed critic, replay, adapters; Adam2, Topk, clip, data-dependent weights'}
    for key in subterms[0]:out[key]=float(np.mean([x[key] for x in subterms]))
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',type=int,choices=[39,118],required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--episodes',type=int,default=12)
    p.add_argument('--draws',type=int,default=16)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=run(a.case,a.seed,a.episodes,a.draws)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))
    if out['status']!='ok':raise SystemExit(2)
if __name__=='__main__':main()
