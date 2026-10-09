#!/usr/bin/env python3
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path('results/joint20')
GEN=Path('results/generated')
GEN.mkdir(parents=True,exist_ok=True)

rows=list(csv.DictReader((ROOT/'summary.csv').open()))
num=['safety_violation','min_cbf_margin','false_authorization_rate','max_belief_distortion','max_policy_drift','recovery_success','recovery_time','shield_intervention_ratio','mean_aggregate_deviation','communication_bytes']
for r in rows:
    r['seed']=int(r['seed'])
    for k in num:
        r[k]=float(r[k])
        if not math.isfinite(r[k]):
            raise SystemExit(f'Non-finite {k} in mechanism summary; generation refused.')

g=defaultdict(list)
for r in rows:
    g[(r['scenario'],r['method'])].append(r)
for key in g:
    g[key]=sorted(g[key],key=lambda x:x['seed'])

def mean_ci(xs):
    xs=np.asarray(xs,float)
    m=float(xs.mean())
    se=float(xs.std(ddof=1)/math.sqrt(len(xs))) if len(xs)>1 else 0.0
    return m,m-1.96*se,m+1.96*se

def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return p,max(0.0,center-half),min(1.0,center+half)

def fmt_mean(xs,binary=False,bounded=False):
    xs=np.asarray(xs,float)
    if binary:
        m,lo,hi=wilson(int(round(xs.sum())),len(xs))
    else:
        m,lo,hi=mean_ci(xs)
        if bounded:
            lo=max(0.0,lo); hi=min(1.0,hi)
    return m,lo,hi,f"{m:.3f} [{lo:.3f}, {hi:.3f}]"

methods=['FedRL','RobustAgg','AEFC-no-PATBU','AEFC-no-Gate','AEFC-no-TGOPA','AEFC-no-Shield','AEFC-Full']
metrics=[('safety_violation','SVR',True),('min_cbf_margin','Min CBF',False),('false_authorization_rate','FAR','bounded'),('max_belief_distortion','Belief dist.',False),('max_policy_drift','Policy drift',False),('recovery_success','Rec. success',True),('recovery_time','Rec. time',False)]
with (GEN/'joint_main.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['method']+[b for _,b,_ in metrics])
    for m in methods:
        rr=g[('joint',m)]
        w.writerow([m]+[fmt_mean([x[a] for x in rr],binary is True,binary=='bounded')[3] for a,_,binary in metrics])

# The original reduced-order runner uses a method-dependent RNG stream.
# Preserve descriptive differences but do not report paired p-values.
desc_metrics=['max_belief_distortion','max_policy_drift','recovery_time','mean_aggregate_deviation']
full=g[('joint','AEFC-Full')]
with (GEN/'joint_statistics.csv').open('w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['metric','comparison','mean_difference_full_minus_comparison','inference_note'])
    for metric in desc_metrics:
        x=np.asarray([r[metric] for r in full],float)
        for m in methods[:-1]:
            y=np.asarray([r[metric] for r in g[('joint',m)]],float)
            w.writerow([metric,m,float(x.mean()-y.mean()),'descriptive_only_method_dependent_rng'])

tex=[r"\begin{table*}[t]",r"\centering",r"\caption{Mechanism-level joint-corruption results over 20 stochastic runs per method. Values are means with 95\% confidence intervals; Wilson intervals are used for binary rates. This table is not an IEEE 39-bus result.}",r"\label{tab:joint-mechanism}",r"\scriptsize",r"\begin{tabular}{lcccccc}",r"\toprule",r"Method & SVR $\downarrow$ & FAR $\downarrow$ & Belief dist. $\downarrow$ & Policy drift $\downarrow$ & Rec. success $\uparrow$ & Rec. time $\downarrow$\\",r"\midrule"]
for m in methods:
    rr=g[('joint',m)]; vals=[]
    for key,binary in [('safety_violation',True),('false_authorization_rate','bounded'),('max_belief_distortion',False),('max_policy_drift',False),('recovery_success',True),('recovery_time',False)]:
        vals.append(fmt_mean([x[key] for x in rr],binary is True,binary=='bounded')[3])
    name=r"\textbf{AEFC-Full}" if m=='AEFC-Full' else m
    tex.append(name+' & '+' & '.join(vals)+r"\\")
tex += [r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
(GEN/'Joint-Table-Auto.tex').write_text('\n'.join(tex)+'\n')

lines=[r"\subsection{Mechanism-Level 20-Run Joint-Corruption Results}",r"The executable bounded-authority pipeline was evaluated under the joint physical/knowledge attack on the released reduced-order CPS integration backend. This backend exercises PATBU, the risk gate, trust-gated adaptation, Byzantine-resilient coordination, the robust execution shield, and post-shield control corruption. It is used to validate mechanism composition and is not presented as an IEEE 39-bus result. All values below are generated from 20 stochastic runs per method and immutable raw JSONL traces tied to one commit and configuration.",r"\input{results/generated/Joint-Table-Auto.tex}"]
for key,label,binary in [('safety_violation','episode safety-violation rate',True),('false_authorization_rate','false-authorization rate','bounded'),('max_belief_distortion','maximum belief distortion',False),('max_policy_drift','maximum deployed-policy drift',False),('recovery_success','recovery success rate',True),('recovery_time','post-attack recovery time',False)]:
    mf,lf,hf,_=fmt_mean([x[key] for x in full],binary is True,binary=='bounded')
    base=g[('joint','FedRL')]
    mb,lb,hb,_=fmt_mean([x[key] for x in base],binary is True,binary=='bounded')
    lines.append(f"For the joint attack, AEFC-Full yields {label} {mf:.4f} (95\\% CI [{lf:.4f}, {hf:.4f}]), versus {mb:.4f} ([{lb:.4f}, {hb:.4f}]) for FedRL.")
lines.append(r"The ablations expose the intended layer separation. Removing the shield causes physical safety violations despite small policy drift, whereas removing trust-gated adaptation sharply increases false authorization and policy drift while the shield can still preserve the physical invariant. Removing PATBU increases belief distortion. These observations are mechanism-level evidence for the compositional design, not a substitute for native IEEE 39-bus validation.")
lines.append(r"The original reduced-order runner derives its stochastic stream from both the seed and the method identifier. Accordingly, these original 20-run values are used descriptively and are not assigned a strict common-random-number paired significance claim. The fairness-calibrated E3 experiment supplies method-independent common random numbers and the corresponding paired Wilcoxon/Holm inference.")
(GEN/'Results-Auto.tex').write_text('\n\n'.join(lines)+'\n')

fig,ax=plt.subplots(figsize=(7.2,3.6)); means=[]; los=[]; his=[]
for m in methods:
    xs=np.asarray([x['safety_violation'] for x in g[('joint',m)]])
    mean,lo,hi=wilson(int(round(xs.sum())),len(xs)); means.append(mean); los.append(mean-lo); his.append(hi-mean)
ax.bar(np.arange(len(methods)),means)
ax.errorbar(np.arange(len(methods)),means,yerr=np.vstack([los,his]),fmt='none',capsize=3)
ax.set_xticks(np.arange(len(methods)),methods,rotation=35,ha='right'); ax.set_ylabel('Episode safety-violation rate'); ax.set_ylim(0,1.05); fig.tight_layout(); fig.savefig(GEN/'joint_safety.pdf'); plt.close(fig)
manifest=json.loads((ROOT/'manifest.json').read_text()); manifest['generated_from']='results/joint20/summary.csv'; manifest['statistics']='descriptive 20-run means/CIs for original mechanism experiment; strict CRN paired inference is supplied by supplemental E3'; (GEN/'generation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
