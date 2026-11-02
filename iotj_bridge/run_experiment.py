#!/usr/bin/env python3
"""Reproducible development bridge for AEFC-FRL on AC IEEE39/118.
Not the original MathWorks dynamic grid experiment. No frequency dynamics.
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import torch
try:
    from .core import Client,flat,assign,aggregate
except ImportError:
    from core import Client,flat,assign,aggregate

class Grid:
    def __init__(self,case,clients,seed,hetero):
        import pandapower as pp
        import pandapower.networks as nw
        self.pp=pp;self.net=nw.case39() if case==39 else nw.case118()
        assert len(self.net.bus)==case
        self.case=case;self.clients=clients;self.seed=seed;self.hetero=hetero
        self.groups=np.array_split(np.arange(len(self.net.load)),clients)
        self.original=self.net.load.p_mw.to_numpy(copy=True)
        self.reset()
    def reset(self):
        self.served=np.full(self.clients,.55)
        self.net.load.loc[:,'scaling']=.55;self.step_idx=0
        self._solve()
        return self.observations()
    def _solve(self):
        try:
            self.pp.runpp(self.net,init='flat',numba=False,max_iteration=25)
            self.valid=bool(self.net.converged)
        except Exception:self.valid=False
        if self.valid:
            self.vm=self.net.res_bus.vm_pu.to_numpy(copy=True)
            loading=np.r_[self.net.res_line.loading_percent.to_numpy(),
                         self.net.res_trafo.loading_percent.to_numpy()]
            self.thermal_excess=float(np.maximum(loading-100,0).sum())
            self.voltage_excess=float((np.maximum(.9-self.vm,0)+np.maximum(self.vm-1.1,0)).sum())
        else:
            self.vm=np.ones(len(self.net.bus))
            self.thermal_excess=1e3;self.voltage_excess=1e3
    def observations(self):
        out=[]
        for i,group in enumerate(self.groups):
            buses=self.net.load.bus.iloc[group].to_numpy(dtype=int)
            volt=self.vm[buses] if self.valid else np.ones(len(buses))
            v=np.asarray([self.served[i],volt.mean(),volt.min(),volt.max(),
                         self.thermal_excess/100.,self.step_idx/8.],dtype=np.float32)
            if self.hetero=='dim' and i%2:v=v[:3]
            if self.hetero=='mask' and i%2:v[2]=0
            if self.hetero=='shift' and i%2:v[1:3]+=.04
            if self.hetero=='delay' and i%2:v[1:3]=1.
            out.append(v)
        return out
    def step(self,actions):
        action=np.asarray(actions,float)
        if self.hetero=='saturation':action[1::2]=np.minimum(action[1::2],.25)
        old=self.served.copy()
        self.served=np.clip(old+.08*np.clip(action,0,1),0,1)
        for i,group in enumerate(self.groups):
            self.net.load.iloc[group,self.net.load.columns.get_loc('scaling')]=self.served[i]
        self.step_idx+=1;self._solve()
        restored=100*float(np.sum([self.original[g].sum()*self.served[i]
                                   for i,g in enumerate(self.groups)])/self.original.sum())
        safe=bool(self.valid and self.thermal_excess<1e-9 and self.voltage_excess<1e-9)
        reward=restored/100-.15*self.thermal_excess-.25*self.voltage_excess
        if not self.valid:reward=-10.
        info={'restored_pct':restored,'safe':safe,'ac_converged':self.valid,
              'voltage_violation_pu_sum':self.voltage_excess,
              'thermal_excess_pct_sum':self.thermal_excess}
        return self.observations(),[reward-.05*(self.served[i]-old[i])**2
                                   for i in range(self.clients)],self.step_idx>=8 or not self.valid,info

def single_run(case,mode,hetero,poison,seed,episodes,clients_n,weight_cap=None,disable_proto=False):
    torch.set_num_threads(1)
    env=Grid(case,clients_n,seed,hetero)
    obs=env.reset()
    learners=[Client(len(o),seed*100+i) for i,o in enumerate(obs)]
    theta=flat(learners[0].backbone)
    for c in learners:assign(c.backbone,theta)
    previous=None;rows=[];total=0
    for episode in range(episodes):
        obs=env.reset();done=False;idx=0;load_path=[];safe_path=[]
        while not done:
            action=[]
            for i,(c,o) in enumerate(zip(learners,obs)):
                with torch.no_grad():
                    nominal=float(c.actor(torch.tensor(o[None,:],dtype=torch.float32))[0,0])
                noise=np.random.default_rng(seed*100000+episode*1000+idx*clients_n+i).normal(0,.12)
                action.append(float(np.clip(nominal+noise,0,1)))
            next_obs,reward,done,last=env.step(action)
            category=('stable' if last['safe'] and last['restored_pct']>=95
                      else 'recovering' if last['restored_pct']>=75 else 'unsafe')
            for i,c in enumerate(learners):
                c.append(obs[i],action[i],reward[i],next_obs[i],done,category)
            obs=next_obs;idx+=1;load_path.append(last['restored_pct']);safe_path.append(int(last['safe']))
        updates=[]
        for i,c in enumerate(learners):
            c.adapt(seed*1000+episode*10+i,previous,steps=1,batch_size=16,
                    align=not disable_proto)
            updates.append(c.share(theta,seed*1000+episode*10+i,mode=mode,batch_size=16,
                                   ratio=.2,steps=2))
        compromised=(clients_n-1,) if poison!='none' else ()
        theta,previous,weights,bytes_this=aggregate(theta,updates,previous,
                                                     bad_indices=compromised,attack=poison,weight_cap=weight_cap)
        for c in learners:assign(c.backbone,theta)
        total+=bytes_this
        rows.append({'case':case,'seed':seed,'method':mode,'hetero':hetero,
                     'attack':poison,'weight_cap':str(weight_cap),'disable_proto':int(disable_proto),'episode':episode+1,'restored_pct':last['restored_pct'],'recovery_auc_pct':float(np.mean(load_path)),
                     'safe_90_dwell2':int(any(load_path[j]>=90 and load_path[j+1]>=90 and safe_path[j] and safe_path[j+1] for j in range(len(load_path)-1))),
                     'safe':int(last['safe']),'ac_converged':int(last['ac_converged']),
                     'voltage_violation_pu_sum':last['voltage_violation_pu_sum'],
                     'thermal_excess_pct_sum':last['thermal_excess_pct_sum'],
                     'uplink_bytes':total,'bad_weight':sum(weights[k] for k in compromised)})
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',type=int,choices=[39,118],required=True)
    p.add_argument('--mode',choices=['adv','ordinary','random','td','uniform_high'],default='adv')
    p.add_argument('--hetero',choices=['base','dim','mask','delay','shift','saturation','no_proto'],default='base')
    p.add_argument('--attack',choices=['none','joint','utility','proto','update'],default='none')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--episodes',type=int,default=4)
    p.add_argument('--clients',type=int,default=5)
    p.add_argument('--weight-cap',type=float,default=None)
    p.add_argument('--disable-proto',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    records=single_run(a.case,a.mode,a.hetero,a.attack,a.seed,a.episodes,a.clients,a.weight_cap,a.disable_proto)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys())
        w.writeheader();w.writerows(records)
    print(json.dumps({'record_count':len(records),'last':records[-1]}))
if __name__=='__main__':main()
