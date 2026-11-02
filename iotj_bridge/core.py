"""Controlled AEFC-FRL sharing prototype (not the PATBU/Gate AEFC).
Separate local encoder/actor head and critic; only actor backbone federated.
This is a development-stage implementation, not a publication-grade reproduction.
"""
from __future__ import annotations
import io, math, struct
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def flat(module):
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])
def assign(module,vector):
    off=0
    with torch.no_grad():
        for p in module.parameters():
            n=p.numel();p.copy_(vector[off:off+n].view_as(p));off+=n
    assert off==len(vector)

class Client:
    def __init__(self,obs_dim,seed,latent=12,hidden=24,lr=.002):
        torch.manual_seed(seed)
        self.encoder=nn.Sequential(nn.Linear(obs_dim,latent),nn.Tanh())
        self.backbone=nn.Sequential(nn.Linear(latent,hidden),nn.ReLU(),nn.Linear(hidden,latent),nn.Tanh())
        self.actor_head=nn.Sequential(nn.Linear(latent,1),nn.Sigmoid())
        self.critic=nn.Sequential(nn.Linear(obs_dim+1,hidden),nn.ReLU(),nn.Linear(hidden,1))
        self.opt_local=torch.optim.Adam(list(self.encoder.parameters())+list(self.actor_head.parameters()),lr=lr)
        self.opt_critic=torch.optim.Adam(self.critic.parameters(),lr=lr)
        self.data=[];self.lr=lr
    def actor(self,s):return self.actor_head(self.backbone(self.encoder(s)))
    def q(self,s,a):return self.critic(torch.cat((s,a),dim=1))
    def append(self,s,a,r,ns,done,category):
        self.data.append((np.asarray(s,np.float32),float(a),float(r),np.asarray(ns,np.float32),bool(done),category))
        if len(self.data)>2000:self.data.pop(0)
    def batch(self,size,seed):
        g=np.random.default_rng(seed)
        sel=g.integers(0,len(self.data),size=size)
        s,a,r,ns,d,cat=zip(*(self.data[k] for k in sel))
        return (torch.from_numpy(np.stack(s)),torch.tensor(a,dtype=torch.float32).reshape(-1,1),
                torch.tensor(r,dtype=torch.float32).reshape(-1,1),
                torch.from_numpy(np.stack(ns)),torch.tensor(d,dtype=torch.float32).reshape(-1,1),list(cat))
    def adapt(self,seed,prototypes=None,steps=1,batch_size=32,align=True):
        if not self.data:return
        for p in self.backbone.parameters():p.requires_grad_(False)
        for j in range(steps):
            s,a,r,ns,d,cats=self.batch(batch_size,seed+j)
            with torch.no_grad():target=r+.97*(1-d)*self.q(ns,self.actor(ns))
            self.opt_critic.zero_grad()
            loss=F.mse_loss(self.q(s,a),target)
            loss.backward();self.opt_critic.step()
            self.opt_local.zero_grad()
            loss=-self.q(s,self.actor(s)).mean()
            if prototypes and align:
                emb=self.encoder(s);pen=[]
                for key,p in prototypes.items():
                    idx=[k for k,cat in enumerate(cats) if cat==key]
                    if idx:pen.append(F.mse_loss(emb[idx].mean(0),p))
                if pen:loss+=.01*torch.stack(pen).mean()
            loss.backward();self.opt_local.step()
        for p in self.backbone.parameters():p.requires_grad_(True)
    def prototypes(self,count=96):
        if not self.data:return {}
        groups={}
        for s,_,_,_,_,c in self.data[-count:]:groups.setdefault(c,[]).append(s)
        with torch.no_grad():
            return {k:self.encoder(torch.from_numpy(np.stack(v))).mean(0).numpy().copy()
                    for k,v in groups.items()}
    def share(self,theta,seed,mode='adv',ratio=.2,steps=2,batch_size=32,mask='magnitude'):
        assign(self.backbone,theta)
        if not self.data:return None
        for p in list(self.encoder.parameters())+list(self.actor_head.parameters()):p.requires_grad_(False)
        s,a,r,ns,d,_=self.batch(batch_size,seed)
        with torch.no_grad():
            advantage=(self.q(s,a)-self.q(s,self.actor(s))).flatten()
            td=(r+.97*(1-d)*self.q(ns,self.actor(ns))-self.q(s,a)).abs().flatten()
            high=(advantage>0).nonzero(as_tuple=False).flatten()
        if mode=='adv' or mode=='uniform_high':selected=high
        elif mode=='ordinary':selected=torch.arange(batch_size)
        elif mode=='random':selected=torch.randperm(batch_size,generator=torch.Generator().manual_seed(seed+1000))[:len(high)]
        elif mode=='td':selected=torch.argsort(td,descending=True,stable=True)[:len(high)]
        else:raise ValueError(mode)
        if len(selected)==0:
            for p in list(self.encoder.parameters())+list(self.actor_head.parameters()):p.requires_grad_(True)
            return None
        weight=advantage[selected].clamp_min(0) if mode=='adv' else torch.ones(len(selected))
        weight=weight/(weight.sum()+1e-12)
        opt=torch.optim.Adam(self.backbone.parameters(),lr=self.lr)
        for step in range(steps):
            loss=-(weight*self.q(s[selected],self.actor(s[selected])).flatten()).sum()
            opt.zero_grad();loss.backward();opt.step()
        delta=flat(self.backbone)-theta
        k=max(1,math.ceil(ratio*len(delta)))
        if mask=='random':
            indices=torch.randperm(len(delta),generator=torch.Generator().manual_seed(seed))[:k]
        else:indices=torch.argsort(delta.abs(),descending=True,stable=True)[:k]
        indices=indices.sort().values
        msg={'idx':indices.numpy().astype(np.int32),'val':delta[indices].numpy().astype(np.float32),
             'n':len(theta),'utility':float(advantage[high].mean()) if len(high) else 0.,
             'prototypes':self.prototypes(),'selected':len(selected),'full_delta':delta.numpy()}
        for p in list(self.encoder.parameters())+list(self.actor_head.parameters()):p.requires_grad_(True)
        assign(self.backbone,theta)
        return msg

def serialized_bytes(m):
    f=io.BytesIO()
    f.write(struct.pack('<IIf',int(m['n']),len(m['idx']),float(m['utility'])))
    f.write(np.asarray(m['idx'],dtype='<i4').tobytes())
    f.write(np.asarray(m['val'],dtype='<f4').tobytes())
    f.write(struct.pack('<I',len(m['prototypes'])))
    for category,values in sorted(m['prototypes'].items()):
        b=category.encode('utf8');values=np.asarray(values,dtype='<f4')
        f.write(struct.pack('<II',len(b),len(values)));f.write(b);f.write(values.tobytes())
    return len(f.getvalue())

def aggregate(theta,messages,previous=None,clip=.075,utility_cap=1.,eps=.001,
              bad_indices=(),attack='none',weight_cap=None):
    packed=[];tx=0
    for i,m in enumerate(messages):
        if m is None:continue
        m={**m,'val':m['val'].copy(),'prototypes':{k:v.copy() for k,v in m['prototypes'].items()}}
        if i in bad_indices:
            if attack in ('utility','joint'):m['utility']=utility_cap
            if attack in ('proto','joint') and previous:
                m['prototypes']={k:v.clone().numpy() for k,v in previous.items()}
            if attack in ('update','joint'):
                m['val']=np.asarray(-np.sign(m['val']+1e-10)*np.minimum(np.abs(m['val'])*2+.01,clip/max(1,np.sqrt(len(m['val'])))),dtype=np.float32)
        tx+=serialized_bytes(m)
        if np.isfinite(m['val']).all() and math.isfinite(m['utility']):packed.append((i,m))
    if not packed:return theta.clone(),previous,[0.]*len(messages),tx
    norms=[];scores=[];updates=[];proto=[]
    for i,m in packed:
        u=torch.zeros_like(theta)
        u[torch.from_numpy(m['idx'].astype(np.int64))]=torch.from_numpy(m['val'])
        u*=min(1.,clip/(float(u.norm())+1e-12))
        shared=set(previous or {}).intersection(m['prototypes'])
        drift=float(np.mean([1-float((torch.from_numpy(m['prototypes'][key])@previous[key])/
                              (torch.from_numpy(m['prototypes'][key]).norm()*previous[key].norm()+1e-9))
                             for key in shared])) if shared else 0.
        score=min(utility_cap,max(0.,m['utility']))/(float(u.norm())+eps)*math.exp(-drift)
        updates.append(u);scores.append(score);proto.append(m['prototypes'])
    weights=np.asarray(scores);weights=weights/weights.sum() if weights.sum()>0 else np.ones(len(weights))/len(weights)
    if weight_cap is not None:
        # Explicitly separate capped mitigation from the original AEFC rule.
        cap=float(weight_cap)
        if not (1./len(weights)<=cap<=1.):
            raise ValueError('weight_cap must be feasible for active clients')
        remainder=1.0;next_w=np.zeros(len(weights));free=set(range(len(weights)))
        while free:
            mass=sum(weights[j] for j in free)
            over=[j for j in free if weights[j]/mass*remainder>cap+1e-12] if mass>0 else []
            if not over:
                for j in free:next_w[j]=remainder*weights[j]/mass if mass>0 else remainder/len(free)
                break
            for j in over:next_w[j]=cap;remainder-=cap;free.remove(j)
        weights=next_w
    result=theta.clone();all_weights=[0.]*len(messages)
    for (i,m),w,u in zip(packed,weights,updates):
        result+=float(w)*u;all_weights[i]=float(w)
    nextp={}
    for key in ('unsafe','recovering','stable'):
        group=[(float(w),torch.from_numpy(p[key])) for w,p in zip(weights,proto) if key in p]
        if group:nextp[key]=sum(w*vec for w,vec in group)/sum(w for w,_ in group)
        elif previous and key in previous:nextp[key]=previous[key]
    return result,nextp,all_weights,tx
