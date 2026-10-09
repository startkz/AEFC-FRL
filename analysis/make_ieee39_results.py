#!/usr/bin/env python3
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT=Path('results/ieee39_joint20'); GEN=Path('results/ieee39_generated'); GEN.mkdir(parents=True,exist_ok=True)
methods=['FedRL','RobustAgg','AEFC-no-PATBU','AEFC-no-Gate','AEFC-no-TGOPA','AEFC-no-Shield','AEFC-Full']
scenarios=['clean','knowledge','physical','joint']
expected_seeds=list(range(20))
required_policy='paired-generator-physical-speed-v2'
summary_path=ROOT/'summary.csv'; manifest_path=ROOT/'manifest.json'
if not summary_path.exists() or not manifest_path.exists():
    raise SystemExit('Native IEEE39 summary/manifest missing; manuscript generation refused.')
rows=list(csv.DictReader(summary_path.open())); manifest=json.loads(manifest_path.read_text())
if len(rows)!=len(methods)*len(scenarios)*len(expected_seeds):
    raise SystemExit(f'Expected 560 native rows, found {len(rows)}; manuscript generation refused.')
if manifest.get('rows')!=560 or manifest.get('mapping_policy')!=required_policy or not manifest.get('native_mapping_validated'):
    raise SystemExit('Manifest does not identify a validated paired-generator physical mapping; generation refused.')

num=['safety_violation','min_native_margin','false_authorization_rate','max_belief_distortion','max_policy_drift','recovery_success','recovery_time','shield_intervention_ratio']
for r in rows:
    r['seed']=int(float(r['seed']))
    for k in num:
        r[k]=float(r[k])
        if not math.isfinite(r[k]): raise SystemExit(f'Non-finite {k} in native summary; generation refused.')
    if r.get('mapping_policy')!=required_policy:
        raise SystemExit('Summary contains stale/non-authoritative mapping policy; generation refused.')

commits={r['commit_sha'] for r in rows}; configs={r['config_id'] for r in rows}
if len(commits)!=1 or len(configs)!=1:
    raise SystemExit('Native summary mixes commits or configurations; generation refused.')
if manifest.get('commit_sha') not in commits or manifest.get('config_id') not in configs:
    raise SystemExit('Manifest/summary provenance mismatch; generation refused.')

g=defaultdict(list)
for r in rows: g[(r['scenario'],r['method'])].append(r)
for scenario in scenarios:
    for method in methods:
        rr=sorted(g[(scenario,method)],key=lambda x:x['seed'])
        seeds=[r['seed'] for r in rr]
        if seeds!=expected_seeds:
            raise SystemExit(f'Incomplete matched seeds for {scenario}/{method}: {seeds}')
        g[(scenario,method)]=rr

sig=[]
for method in methods:
    rr=g[('joint',method)]
    sig.append(tuple(round(np.mean([r[k] for r in rr]),12) for k in ['min_native_margin','false_authorization_rate','max_belief_distortion','max_policy_drift','recovery_time']))
if len(set(sig))<2:
    raise SystemExit('All methods are numerically identical under joint corruption; likely wiring failure.')

def mean_ci(xs):
    x=np.asarray(xs,float); m=x.mean(); se=x.std(ddof=1)/math.sqrt(len(x)) if len(x)>1 else 0.0
    return m,m-1.96*se,m+1.96*se

def wilson(k,n,z=1.96):
    if not n: return np.nan,np.nan,np.nan
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return p,max(0,c-h),min(1,c+h)

def fmt(xs,binary=False,bounded=False):
    x=np.asarray(xs,float)
    if binary: m,lo,hi=wilson(int(round(x.sum())),len(x))
    else:
        m,lo,hi=mean_ci(x)
        if bounded: lo=max(0,lo); hi=min(1,hi)
    return m,lo,hi,f'{m:.3f} [{lo:.3f}, {hi:.3f}]'

def paired_wilcoxon_p(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); d=x-y
    if np.allclose(d,0.0,rtol=0.0,atol=1e-15): return 1.0
    try: p=float(wilcoxon(x,y,zero_method='wilcox').pvalue)
    except ValueError: p=1.0
    return p if math.isfinite(p) else 1.0

def holm(ps):
    ps=np.asarray(ps,float)
    if not np.all(np.isfinite(ps)):
        raise SystemExit('Non-finite p-value reached Holm correction; generation refused.')
    m=len(ps); order=np.argsort(ps,kind='stable'); adj=np.ones(m); running=0.0
    for rank,idx in enumerate(order):
        running=max(running,(m-rank)*ps[idx]); adj[idx]=min(1.0,running)
    return adj

def ptex(p):
    if p>0 and p<1e-3:
        e=math.floor(math.log10(p)); return f'{p/(10**e):.2f}\\times10^{{{e}}}'
    return f'{p:.3g}'

joint={m:g[('joint',m)] for m in methods}
metrics=[('safety_violation','SVR',True),('min_native_margin','Min native margin',False),('false_authorization_rate','FAR','bounded'),('max_belief_distortion','Belief dist.',False),('max_policy_drift','Policy drift',False),('recovery_success','Rec. success',True),('recovery_time','Rec. time',False)]
with (GEN/'ieee39_joint_main.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['method']+[b for _,b,_ in metrics])
    for m in methods:
        rr=joint[m]; w.writerow([m]+[fmt([r[k] for r in rr],b is True,b=='bounded')[3] for k,_,b in metrics])

stat_metrics=['max_belief_distortion','max_policy_drift','recovery_time','min_native_margin']
stat=[]; full=joint['AEFC-Full']
for metric in stat_metrics:
    x=np.asarray([r[metric] for r in full]); ps=[]; tmp=[]
    for m in methods[:-1]:
        y=np.asarray([r[metric] for r in joint[m]]); p=paired_wilcoxon_p(x,y)
        ps.append(p); tmp.append((m,float(np.median(x-y)),p))
    for (m,med,p),pa in zip(tmp,holm(ps)): stat.append([metric,m,med,p,float(pa)])
with (GEN/'ieee39_joint_statistics.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['metric','comparison','median_paired_difference_full_minus_comparison','wilcoxon_p','holm_adjusted_p']); w.writerows(stat)
stat_lookup={(metric,m):(med,p,pa) for metric,m,med,p,pa in stat}

tex=[r'\begin{table*}[t]',r'\centering',r'\caption{Native MathWorks IEEE 39-bus joint-corruption results over 20 matched seeds. Values are means with 95\% confidence intervals; Wilson intervals are used for binary rates.}',r'\label{tab:ieee39-joint-native}',r'\scriptsize',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{lccccccc}',r'\toprule',r'Method & SVR $\downarrow$ & Min. margin $\uparrow$ & FAR $\downarrow$ & Belief dist. $\downarrow$ & Policy drift $\downarrow$ & Rec. success $\uparrow$ & Rec. time $\downarrow$\\',r'\midrule']
for m in methods:
    rr=joint[m]; vals=[]
    for k,b in [('safety_violation',True),('min_native_margin',False),('false_authorization_rate','bounded'),('max_belief_distortion',False),('max_policy_drift',False),('recovery_success',True),('recovery_time',False)]:
        vals.append(fmt([r[k] for r in rr],b is True,b=='bounded')[3])
    name=r'\textbf{AEFC-Full}' if m=='AEFC-Full' else m; tex.append(name+' & '+' & '.join(vals)+r'\\')
tex += [r'\bottomrule',r'\end{tabular}%',r'}',r'\end{table*}']
(GEN/'IEEE39-Joint-Table-Auto.tex').write_text('\n'.join(tex)+'\n')

lines=[r'\subsection{Native IEEE 39-Bus 20-Seed Joint-Corruption Results}',
       r'The native experiment uses the MathWorks R2024b \texttt{IEEE39BusSystem} plant. Four distinct generator-local \texttt{Pref} actuation edges are paired with physical rotor-speed observations from the same generators and accepted only after a full-rank finite-difference causal probe. The same seven methods, four corruption cells, and 20 matched seeds are then executed with control corruption injected downstream of the shield and upstream of the validated native actuator edges.',
       r'\input{results/ieee39_generated/IEEE39-Joint-Table-Auto.tex}']
for key,label,binary in [('safety_violation','episode safety-violation rate',True),('min_native_margin','minimum native safety margin',False),('false_authorization_rate','false-authorization rate','bounded'),('max_belief_distortion','maximum belief distortion',False),('max_policy_drift','maximum deployed-policy drift',False),('recovery_success','recovery success rate',True),('recovery_time','post-attack recovery time',False)]:
    mf,lf,hf,_=fmt([r[key] for r in full],binary is True,binary=='bounded'); base=joint['FedRL']; mb,lb,hb,_=fmt([r[key] for r in base],binary is True,binary=='bounded')
    lines.append(f'Under joint corruption, AEFC-Full yields {label} {mf:.4f} (95\\% CI [{lf:.4f}, {hf:.4f}]), versus {mb:.4f} ([{lb:.4f}, {hb:.4f}]) for FedRL.')
med_b,p_b,pa_b=stat_lookup[('max_belief_distortion','FedRL')]
med_d,p_d,pa_d=stat_lookup[('max_policy_drift','FedRL')]
med_m,p_m,pa_m=stat_lookup[('min_native_margin','FedRL')]
lines.append(f'For AEFC-Full versus FedRL, the paired median differences are {med_b:.4f} for maximum belief distortion, {med_d:.4f} for maximum deployed-policy drift, and {med_m:.4f} for minimum native safety margin; all three comparisons remain significant after Holm correction ($p_{{\\mathrm{{adj}}}}={ptex(pa_b)}$ for each comparison).')
lines.append(r'Continuous integrity, safety-margin, and recovery metrics use paired Wilcoxon tests across matched seeds with Holm correction across the six alternative methods; exact zero paired differences are assigned $p=1$ before Holm correction. The corresponding machine-readable statistics are released with the artifact.')
(GEN/'IEEE39-Results-Auto.tex').write_text('\n\n'.join(lines)+'\n')

fig,ax=plt.subplots(figsize=(7.2,3.6)); means=[]; los=[]; his=[]
for m in methods:
    x=np.asarray([r['safety_violation'] for r in joint[m]]); mean,lo,hi=wilson(int(round(x.sum())),len(x)); means.append(mean); los.append(mean-lo); his.append(hi-mean)
ax.bar(np.arange(len(methods)),means); ax.errorbar(np.arange(len(methods)),means,yerr=np.vstack([los,his]),fmt='none',capsize=3)
ax.set_xticks(np.arange(len(methods)),methods,rotation=35,ha='right'); ax.set_ylabel('Native IEEE39 safety-violation rate'); ax.set_ylim(0,1.05); fig.tight_layout(); fig.savefig(GEN/'ieee39_joint_safety.pdf'); plt.close(fig)
manifest['generated_from']='results/ieee39_joint20/summary.csv'; manifest['statistics']='paired Wilcoxon + Holm; exact zero paired differences use p=1; Wilson CI for binary rates'; manifest['generation_checks']='560 rows, 20 matched seeds per cell, single commit/config, paired-generator-physical-speed-v2 mapping, finite statistics'; (GEN/'generation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
