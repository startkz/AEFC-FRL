"""No-oracle traffic federation: independent component-level AEFC-FRL implementation.

This is NOT the historical linear Traffic-PoC, nor the IEEE39 Simulink plant.
The env dynamics deliberately preserve the historical queue/phase/attack equations.
Audit output records per-episode safety, recovery, censoring and exact uplink payloads.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, struct, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEEDS = (11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67)
METHODS = ("AEFC-FRL", "FedRL", "AEFC-no-adapter", "AEFC-no-proto")
OBS = 5
LATENT = 16
HORIZON = 60
N_CLIENTS = 5


def sseed(seed: int, cid: int, episode: int, stage: int, suffix: int = 0) -> int:
    return int((seed * 10000019 + cid * 100003 + episode * 113 + stage * 65537 + suffix) % (2**32))


class TrafficEnv:
    def __init__(self, cid: int, seed: int, episode: int, stage: int):
        self.cid = cid
        self.rng = np.random.default_rng(sseed(seed, cid, episode, stage, 3))
        self.arrivals = np.zeros((HORIZON, 2))
        self.noise = self.rng.normal(0, 1, (HORIZON, 2))
        base = np.array([2.2 + .15 * cid, 1.8 + .1 * (4 - cid)])
        for t in range(HORIZON):
            self.arrivals[t] = self.rng.poisson(base + (np.array([1.4, .8]) if 15 <= t < 32 else 0))
        self.q = np.array([8.0 + cid, 6.0 + .5 * cid], dtype=float)
        self.phase = self.prev_action = 0
        self.t = 0
        self.local_bias = (cid - 2) / 5.0

    def obs(self):
        # No attack flag, time-to-attack, or true queue oracle is exposed.
        o = self.q.copy()
        if 15 <= self.t < 32:
            o = np.maximum(0, o + np.array([4., -2.]) + self.noise[self.t])
        return np.asarray([1., o[0]/30., o[1]/30., float(self.phase), self.local_bias], np.float32)

    def step(self, action: int):
        assert action in (0, 1) and self.t < HORIZON
        t = self.t
        self.q += self.arrivals[t]
        applied = self.prev_action if 15 <= t < 32 and self.cid % 2 == 0 else action
        self.prev_action = action
        service = np.array([5.4, 1.]) if applied == 0 else np.array([1., 5.4])
        self.q = np.maximum(0., self.q - service)
        changed = int(applied != self.phase)
        self.phase = applied
        queue = float(self.q.sum())
        congestion = bool(self.q.max() > 35.)
        reward = -queue - .6*changed - 2.*float(congestion)
        self.t += 1
        return reward, queue, congestion, bool(self.t >= HORIZON)


class Replay:
    def __init__(self, capacity=20000, alpha=.6):
        self.capacity, self.alpha, self.pos = capacity, alpha, 0
        self.data, self.priorities = [], np.zeros(capacity, dtype=np.float64)

    def add(self, record):
        p = max(float(self.priorities[:len(self.data)].max()) if self.data else 1., 1e-4)
        if len(self.data) < self.capacity: self.data.append(record)
        else: self.data[self.pos] = record
        self.priorities[self.pos] = p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, rng, batch=64, beta=.4):
        n = len(self.data)
        if n == 0: raise RuntimeError("empty replay")
        probs = np.maximum(self.priorities[:n], 1e-6) ** self.alpha
        probs /= probs.sum()
        indices = rng.choice(n, size=batch, p=probs, replace=True)
        w = (n*probs[indices])**(-beta)
        w /= w.max()
        records = [self.data[int(i)] for i in indices]
        o,a,r,o2,d,cat=zip(*records)
        return (torch.tensor(np.stack(o)), torch.tensor(a, dtype=torch.long),
                torch.tensor(r, dtype=torch.float32),torch.tensor(np.stack(o2)),
                torch.tensor(d, dtype=torch.float32),torch.tensor(cat, dtype=torch.long),
                torch.tensor(w, dtype=torch.float32),indices)

    def update(self, indices, delta):
        for i,p in zip(indices, np.asarray(delta, dtype=float)):
            self.priorities[int(i)] = abs(p)+1e-5


class LocalActor(nn.Module):
    def __init__(self, no_adapter=False):
        super().__init__()
        self.no_adapter = no_adapter
        self.encoder = None if no_adapter else nn.Sequential(nn.Linear(OBS,LATENT),nn.Tanh())
        self.backbone = nn.Sequential(nn.Linear(LATENT,32), nn.ReLU(),nn.Linear(32,LATENT),nn.Tanh())
        self.head = nn.Linear(LATENT,2)

    def encode(self, obs):
        if self.no_adapter:
            return F.pad(obs,(0,LATENT-OBS))
        return self.encoder(obs)

    def forward(self, obs):
        return self.head(self.backbone(self.encode(obs)))


class Client:
    def __init__(self, cid, seed, method, lr, batch, per_alpha):
        self.cid, self.method, self.seed, self.lr, self.batch = cid, method, seed, lr, batch
        self.actor = LocalActor(no_adapter=method=="AEFC-no-adapter")
        self.critic = nn.Sequential(nn.Linear(OBS,64),nn.ReLU(),nn.Linear(64,2))
        self.critic_target = copy.deepcopy(self.critic)
        self.replay = Replay(alpha=per_alpha)
        self.rng = np.random.default_rng(sseed(seed,cid,0,3))
        self.local_params = list(self.actor.head.parameters()) + (list(self.actor.encoder.parameters()) if self.actor.encoder is not None else [])
        self.opt_local = torch.optim.Adam(self.local_params,lr=lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(),lr=lr)

    def set_global(self, global_backbone):
        self.actor.backbone.load_state_dict(global_backbone.state_dict())

    def action(self, obs, rng, explore=True):
        with torch.no_grad():
            z = self.actor(torch.from_numpy(obs).reshape(1,-1))
            p = z.softmax(-1).numpy()[0]
        return int(rng.choice(2,p=p)) if explore else int(np.argmax(p))

    def episode(self, seed, index, stage, explore=True, train=True):
        env=TrafficEnv(self.cid,seed,index,stage)
        rng=np.random.default_rng(sseed(seed,self.cid,index,stage,5))
        total=0.; queues=[]; violations=0; recovered=-1; safe_run=0; dwell=-1; physical_trace=[]
        while env.t<HORIZON:
            t=env.t; obs=env.obs(); act=self.action(obs,rng,explore)
            applied = env.prev_action if 15 <= t < 32 and self.cid % 2 == 0 else act
            exogenous_arrivals = env.arrivals[t].copy()
            reward,q,unsafe,done=env.step(act)
            physical_trace.append([t, float(env.q[0]), float(env.q[1]), int(env.phase), int(act),
                                   int(applied), float(exogenous_arrivals[0]), float(exogenous_arrivals[1]),
                                   float(reward)])
            nxt=env.obs() if not done else np.zeros(OBS,dtype=np.float32)
            if train:self.replay.add((obs,act,reward,nxt,float(done),int(obs[3])))
            total+=reward;queues.append(q);violations+=int(unsafe)
            # Historical diagnostic only: success inside [15,32), not post-attack recovery.
            if 15<=t<32 and q<18 and recovered<0: recovered=t-15+1
            # A separate 5-step post-attack safe-dwell endpoint, right-censored at 60.
            if t>=32 and q<18 and not unsafe: safe_run+=1
            else: safe_run=0
            if dwell<0 and safe_run>=5: dwell=t-4
        return {"method":self.method,"seed":seed,"client":self.cid,"episode":index,
                "stage":"train" if stage==0 else "heldout", "return":total,
                "avg_queue":float(np.mean(queues)),"terminal_queue":float(queues[-1]),
                "attack_window_success":int(recovered>0),"attack_window_recovery_step":recovered,
                "safe_dwell5":int(dwell>=0),"recovery_step":dwell,"right_censored":int(dwell<0),
                "unsafe_steps":violations,"physical_trace":physical_trace}

    def prototypes(self, last=512):
        groups={0:[],1:[]}
        for rec in self.replay.data[-last:]:groups[int(rec[5])].append(rec[0])
        proto={}
        with torch.no_grad():
            for key,arr in groups.items():
                if arr:proto[key]=(self.actor.encode(torch.tensor(np.stack(arr))).mean(0).numpy().astype('<f4'),len(arr))
        return proto

    def local_update(self, proto_global, step, align=.01):
        if not self.replay.data:return
        data=self.replay.sample(self.rng,batch=self.batch,beta=min(1.,.4+.6*step/80))
        obs,a,rew,nxt,done,cat,w,idx=data
        with torch.no_grad():target=rew + .97*(1-done)*self.critic_target(nxt).max(1).values
        q=self.critic(obs);picked=q.gather(1,a[:,None]).flatten()
        td=target-picked
        self.opt_critic.zero_grad();(w*td.square()).mean().backward();self.opt_critic.step()
        self.replay.update(idx,td.detach().numpy())
        with torch.no_grad():q=self.critic(obs);pi=self.actor(obs).softmax(-1);adv=(q.gather(1,a[:,None]).flatten()-(pi*q).sum(-1)).detach()
        for p in self.actor.backbone.parameters():p.requires_grad_(False)
        logits=self.actor(obs)
        lp=F.log_softmax(logits,-1).gather(1,a[:,None]).flatten()
        loss=-(w*adv*lp).mean()-.005*torch.distributions.Categorical(logits=logits).entropy().mean()
        if self.method not in ("AEFC-no-proto","FedRL") and proto_global:
            emb=self.actor.encode(obs);terms=[]
            for key,p in proto_global.items():
                mask=(cat==key)
                if mask.any():terms.append((emb[mask].mean(0)-p).square().mean())
            if terms:loss+=align*torch.stack(terms).mean()
        self.opt_local.zero_grad();loss.backward();self.opt_local.step()
        for p in self.actor.backbone.parameters():p.requires_grad_(True)
        with torch.no_grad():
            for t,s in zip(self.critic_target.parameters(),self.critic.parameters()):t.mul_(.995).add_(s,alpha=.005)

    def share(self, theta, kappa, round_i):
        self.set_global(theta)
        obs,a,r,nxt,d,cat,w,idx=self.replay.sample(self.rng,batch=self.batch)
        with torch.no_grad():
            q=self.critic(obs); probs=self.actor(obs).softmax(-1)
            adv=q.gather(1,a[:,None]).flatten()-(q*probs).sum(-1)
            high=adv>0
        if self.method=="FedRL":mask=torch.ones_like(high)
        else:mask=high
        if not mask.any(): mask=torch.zeros_like(high)
        for p in self.local_params:p.requires_grad_(False)
        optimizer=torch.optim.Adam(self.actor.backbone.parameters(),lr=self.lr)
        if mask.any():
            for _ in range(2):
                logits=self.actor(obs[mask]);lp=F.log_softmax(logits,-1).gather(1,a[mask,None]).flatten()
                weighted=adv[mask].detach()
                # The selected sample weights are reused for both Adam steps.
                loss=-(lp*weighted).mean()
                optimizer.zero_grad();loss.backward();optimizer.step()
        delta=torch.nn.utils.parameters_to_vector(self.actor.backbone.parameters()).detach()-torch.nn.utils.parameters_to_vector(theta.parameters()).detach()
        if self.method=="FedRL":
            vals=delta.numpy().astype('<f4');packet=struct.pack('<I',len(vals))+vals.tobytes()
            message={"delta":delta.clone(),"utility":1.,"prototypes":{},"bytes":len(packet),"selected":int(mask.sum())}
        else:
            n=len(delta);k=max(1,math.ceil(kappa*n))
            ids=torch.argsort(delta.abs(),descending=True,stable=True)[:k].sort().values.numpy().astype('<i4')
            vals=delta.numpy()[ids].astype('<f4')
            utility=float(adv[high].mean()) if high.any() else 0.
            packet=struct.pack('<IIf',n,k,utility)+ids.tobytes()+vals.tobytes()
            proto={} if self.method=="AEFC-no-proto" else self.prototypes()
            for key,(vec,count) in sorted(proto.items()):packet+=struct.pack('<II',key,count)+vec.tobytes()
            message={"indices":ids,"values":vals,"n":n,"utility":max(0.,utility),
                     "prototypes":proto,"bytes":len(packet),"selected":int(high.sum())}
        for p in self.local_params:p.requires_grad_(True)
        self.set_global(theta)
        return message


def aggregate(messages, method, template, malicious_cid, max_norm=.45):
    n=torch.nn.utils.parameters_to_vector(template.parameters()).numel()
    vectors=[];weights=[]
    for cid,m in enumerate(messages):
        if method=="FedRL":v=m["delta"].numpy().copy();weight=1.
        else:
            v=np.zeros(n,dtype=np.float32);v[m["indices"]]=m["values"]
            weight=max(0.02,float(m["utility"]))
        if cid==malicious_cid:v=-3*v
        magnitude=float(np.linalg.norm(v))
        if method!="FedRL" and magnitude>max_norm:v=v*(max_norm/magnitude)
        vectors.append(v);weights.append(weight)
    weights=np.asarray(weights,float);weights=weights/weights.sum()
    delta=(np.stack(vectors,axis=0)*weights[:,None]).sum(axis=0)
    return torch.from_numpy(delta.astype(np.float32)),{
        "attacker_client":malicious_cid,"applied_norm":float(np.linalg.norm(delta)),
        "max_client_clipped_norm":float(max(np.linalg.norm(v) for v in vectors))}


def train(args):
    torch.set_num_threads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    method=args.method
    clients=[Client(i,args.seed,method,args.lr,args.batch,.6) for i in range(5)]
    global_backbone=copy.deepcopy(clients[0].actor.backbone)
    for c in clients:c.set_global(global_backbone)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    train_file=out/'training_episodes.jsonl';round_file=out/'rounds.jsonl'
    all_train=[];all_rounds=[];global_proto={}
    for round_i in range(args.rounds):
        for c in clients:c.set_global(global_backbone)
        for c in clients:
            for local_ep in range(args.local_episodes):
                ep=round_i*args.local_episodes+local_ep
                all_train.append(c.episode(args.seed,ep,0,explore=True,train=True))
        for c in clients:
            for _ in range(args.local_steps):c.local_update(global_proto,round_i)
        messages=[c.share(global_backbone,args.kappa,round_i) for c in clients]
        diff,agg_diag=aggregate(messages,method,global_backbone,args.seed%5)
        with torch.no_grad():
            v=torch.nn.utils.parameters_to_vector(global_backbone.parameters())+diff
            torch.nn.utils.vector_to_parameters(v,global_backbone.parameters())
        all_proto={0:[],1:[]}
        if method not in ('FedRL','AEFC-no-proto'):
            for m in messages:
                for key,(vec,count) in m['prototypes'].items():all_proto[key].append((vec,count))
            global_proto={k:torch.tensor(np.average(np.stack([v for v,_ in arr]),axis=0,weights=[num for _,num in arr]),dtype=torch.float32) for k,arr in all_proto.items() if arr}
        else:global_proto={}
        all_rounds.append({"round":round_i+1,"bytes_per_client":[m['bytes'] for m in messages],
                           "total_application_uplink_bytes":sum(m['bytes'] for m in messages),
                           "selected_samples":[m['selected'] for m in messages],**agg_diag})
    test=[]
    for c in clients:
        c.set_global(global_backbone)
        for ep in range(args.eval_episodes):test.append(c.episode(args.seed,ep,1,explore=False,train=False))
    with train_file.open('w') as f:
        for r in all_train:f.write(json.dumps(r,allow_nan=False)+'\n')
    with round_file.open('w') as f:
        for r in all_rounds:f.write(json.dumps(r,allow_nan=False)+'\n')
    with (out/'heldout_episodes.jsonl').open('w') as f:
        for r in test:f.write(json.dumps(r,allow_nan=False)+'\n')
    checkpoint={"global_backbone":global_backbone.state_dict(),"clients":[{"encoder":c.actor.encoder.state_dict() if c.actor.encoder is not None else None,"head":c.actor.head.state_dict(),"critic":c.critic.state_dict()} for c in clients],
                "method":method,"seed":args.seed}
    torch.save(checkpoint,out/'model.pt')
    summary={"method":method,"seed":args.seed,"training_episodes":len(all_train),"federated_rounds":len(all_rounds),
             "heldout_episodes":len(test),"application_uplink_bytes":sum(x['total_application_uplink_bytes'] for x in all_rounds),
             "heldout_success_fraction":float(np.mean([x['attack_window_success'] for x in test])),
             "heldout_safe_dwell5_fraction":float(np.mean([x['safe_dwell5'] for x in test])),
             "heldout_mean_queue":float(np.mean([x['avg_queue'] for x in test])),
             "heldout_mean_return":float(np.mean([x['return'] for x in test])),
             "heldout_censored_count":sum(x['right_censored'] for x in test),
             "train_censored_count":sum(x['right_censored'] for x in all_train)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    manifest={"source":"startkz/AEFC-FRL","protocol":"traffic-v1-no-oracle-full-backbone","git_sha":os.environ.get('GITHUB_SHA','local'),
              "method":method,"seed":args.seed,"config":vars(args),"sha256":sha,
              "limitations":["IEEE39 Simulink results not reproduced","No guaranteed policy improvement","Attack-window success is not post-attack physical recovery"]}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,sort_keys=True,allow_nan=False))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--rounds',type=int,default=80)
    p.add_argument('--local-episodes',type=int,default=4)
    p.add_argument('--local-steps',type=int,default=4)
    p.add_argument('--eval-episodes',type=int,default=25)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--kappa',type=float,default=.2)
    p.add_argument('--out',required=True)
    a=p.parse_args();assert a.seed>=0 and a.local_episodes>0 and a.rounds>0 and 0<a.kappa<=1
    train(a)

if __name__=='__main__':main()