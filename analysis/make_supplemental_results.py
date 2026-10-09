#!/usr/bin/env python3
import csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr

ROOT = Path("results")
GEN = ROOT / "supplemental_generated"; GEN.mkdir(parents=True, exist_ok=True)
METHODS7 = ["FedRL","RobustAgg","AEFC-no-PATBU","AEFC-no-Gate","AEFC-no-TGOPA","AEFC-no-Shield","AEFC-Full"]
E1_METHODS = ["FedRL","RobustAgg","AEFC-no-PATBU","AEFC-no-Gate","AEFC-Full"]


def read_csv(path):
    if not Path(path).exists():
        raise SystemExit(f"Required supplemental input missing: {path}")
    return list(csv.DictReader(Path(path).open()))


def f(r, k):
    return float(r[k])


def mean_ci(xs):
    x=np.asarray(xs,float); m=float(x.mean()); se=float(x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else 0.0
    return m,m-1.96*se,m+1.96*se


def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return p,max(0,c-h),min(1,c+h)


def paired_p(x,y):
    d=np.asarray(x,float)-np.asarray(y,float)
    if np.allclose(d,0,rtol=0,atol=1e-15): return 1.0
    try: p=float(wilcoxon(x,y,zero_method="wilcox").pvalue)
    except ValueError: p=1.0
    return p if math.isfinite(p) else 1.0


def holm(ps):
    ps=np.asarray(ps,float)
    if not np.all(np.isfinite(ps)): raise SystemExit("Non-finite p-value in supplemental Holm correction")
    m=len(ps); order=np.argsort(ps,kind="stable"); adj=np.ones(m); running=0.0
    for rank,idx in enumerate(order):
        running=max(running,(m-rank)*ps[idx]); adj[idx]=min(1.0,running)
    return adj


def fmt(m): return f"{m:.4f}"
def safe_spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if np.allclose(a,a[0]) or np.allclose(b,b[0]): return float("nan"),float("nan")
    r=spearmanr(a,b); return float(r.statistic),float(r.pvalue)


def by(rows, *keys):
    d=defaultdict(list)
    for r in rows: d[tuple(r[k] for k in keys)].append(r)
    return d

# ---------- authoritative inputs ----------
main_native=read_csv(ROOT/"ieee39_joint20"/"summary.csv")
main_mech=read_csv(ROOT/"joint20"/"summary.csv")
cal=read_csv(ROOT/"supplemental_mechanism"/"calibration"/"summary.csv")
if len(cal)!=80 or {int(float(r["seed"])) for r in cal}!=set(range(20)):
    raise SystemExit("Mechanism calibration must contain 4 methods x 20 seeds")
if not all(str(r.get("common_random_numbers","")).lower() in ("true","1") for r in cal):
    raise SystemExit("Mechanism calibration is not marked common-random-number")

# ---------- E1: attack-intensity robustness ----------
intensity_rows=[]
# gamma=0 reuses authoritative clean cell; gamma=1 will be replaced by a supplemental reproduction check.
for r in main_native:
    if r["scenario"]=="clean" and r["method"] in E1_METHODS:
        rr=dict(r); rr["gamma"]="0"; intensity_rows.append(rr)
for gamma,tag in [(0.5,"0p5"),(1.0,"1"),(1.5,"1p5"),(2.0,"2")]:
    rows=read_csv(ROOT/"supplemental_native"/"intensity"/tag/"summary.csv")
    if len(rows)!=100: raise SystemExit(f"Intensity gamma={gamma} must contain 100 rows")
    for r in rows:
        rr=dict(r); rr["gamma"]=str(gamma); intensity_rows.append(rr)

# Reproduction guard: gamma=1 supplemental must reproduce authoritative joint run for the shared methods.
main_joint={(r["method"],int(float(r["seed"]))):r for r in main_native if r["scenario"]=="joint" and r["method"] in E1_METHODS}
supp_one={(r["method"],int(float(r["seed"]))):r for r in intensity_rows if float(r["gamma"])==1.0}
check_metrics=["safety_violation","min_native_margin","false_authorization_rate","max_belief_distortion","max_policy_drift","recovery_success","recovery_time","shield_intervention_ratio"]
max_repro=0.0
for key,old in main_joint.items():
    new=supp_one[key]
    for m in check_metrics: max_repro=max(max_repro,abs(float(old[m])-float(new[m])))
if max_repro>1e-9:
    raise SystemExit(f"Supplemental controller gamma=1 does not reproduce Main Experiment (max diff={max_repro})")

int_g=by(intensity_rows,"gamma","method")
e1=[]
for gamma in [0,0.5,1,1.5,2]:
    for method in E1_METHODS:
        rr=int_g[(str(gamma) if gamma in (0,1,2) else str(gamma),method)]
        # tolerate Python string formatting differences
        if not rr:
            rr=[r for r in intensity_rows if abs(float(r["gamma"])-gamma)<1e-12 and r["method"]==method]
        if len(rr)!=20: raise SystemExit(f"E1 incomplete for gamma={gamma}, method={method}: {len(rr)}")
        row={"gamma":gamma,"method":method}
        for metric in ["min_native_margin","max_belief_distortion","max_policy_drift","shield_intervention_ratio","recovery_time"]:
            row[metric],row[metric+"_lo"],row[metric+"_hi"]=mean_ci([f(x,metric) for x in rr])
        row["safety_violation_rate"]=sum(f(x,"safety_violation") for x in rr)/20
        e1.append(row)
with (GEN/"E1_attack_intensity.csv").open("w",newline="") as fp:
    w=csv.DictWriter(fp,fieldnames=list(e1[0])); w.writeheader(); w.writerows(e1)

for metric,ylabel,name in [
    ("min_native_margin","Minimum native safety margin","E1_attack_intensity_margin.pdf"),
    ("max_belief_distortion","Maximum belief distortion","E1_attack_intensity_belief.pdf"),
    ("max_policy_drift","Maximum deployed-policy movement","E1_attack_intensity_policy.pdf")]:
    fig,ax=plt.subplots(figsize=(6.8,4.0))
    for method in E1_METHODS:
        rr=[r for r in e1 if r["method"]==method]
        ax.plot([r["gamma"] for r in rr],[r[metric] for r in rr],marker="o",label=method)
    ax.set_xlabel("Joint-corruption scale gamma"); ax.set_ylabel(ylabel); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(GEN/name); plt.close(fig)

# ---------- E2a: PATBU local causal contribution ----------
patbu=[]; ps=[]; tmp=[]
for gamma in [0.5,1.0,1.5,2.0]:
    full=sorted([r for r in intensity_rows if abs(float(r["gamma"])-gamma)<1e-12 and r["method"]=="AEFC-Full"],key=lambda r:int(float(r["seed"])))
    no=sorted([r for r in intensity_rows if abs(float(r["gamma"])-gamma)<1e-12 and r["method"]=="AEFC-no-PATBU"],key=lambda r:int(float(r["seed"])))
    x=np.asarray([f(r,"max_belief_distortion") for r in full]); y=np.asarray([f(r,"max_belief_distortion") for r in no])
    p=paired_p(x,y); ps.append(p); tmp.append((gamma,float(np.median(x-y)),float(x.mean()),float(y.mean()),p))
for item,pa in zip(tmp,holm(ps)):
    gamma,med,mf,mn,p=item; patbu.append({"gamma":gamma,"full_mean_belief_distortion":mf,"no_patbu_mean_belief_distortion":mn,"median_paired_full_minus_no_patbu":med,"wilcoxon_p":p,"holm_adjusted_p":float(pa)})
with (GEN/"E2_PATBU_causal_ablation.csv").open("w",newline="") as fp:
    w=csv.DictWriter(fp,fieldnames=list(patbu[0])); w.writeheader(); w.writerows(patbu)

# ---------- E2b: Gate authority stress ----------
gate_rows=[]
for alpha,tag in [(0.0,"0"),(0.25,"0p25"),(0.5,"0p5"),(0.75,"0p75"),(1.0,"1")]:
    rows=read_csv(ROOT/"supplemental_native"/"gate"/tag/"summary.csv")
    if len(rows)!=40: raise SystemExit(f"Gate alpha={alpha} must contain 40 rows")
    for r in rows:
        rr=dict(r); rr["alpha"]=alpha; gate_rows.append(rr)
gg=by(gate_rows,"alpha","method"); gate_out=[]
for alpha in [0,0.25,0.5,0.75,1.0]:
    for method in ["AEFC-no-Gate","AEFC-Full"]:
        rr=[r for r in gate_rows if abs(float(r["alpha"])-alpha)<1e-12 and r["method"]==method]
        dangerous=sum(int(float(r["dangerous_proposal_count"])) for r in rr)
        authorized=sum(int(float(r["dangerous_authorization_count"])) for r in rr)
        pooled_far=authorized/dangerous if dangerous else 0.0
        gate_out.append({"alpha":alpha,"method":method,"dangerous_proposals":dangerous,"dangerous_authorizations":authorized,"pooled_false_authorization_rate":pooled_far,
                         "mean_gate_authorization_rate":float(np.mean([f(r,"gate_authorization_rate") for r in rr])),
                         "mean_shield_intervention_ratio":float(np.mean([f(r,"shield_intervention_ratio") for r in rr])),
                         "mean_shield_intervention":float(np.mean([f(r,"mean_shield_intervention") for r in rr])),
                         "mean_min_native_margin":float(np.mean([f(r,"min_native_margin") for r in rr]))})
with (GEN/"E2_Gate_authority_stress.csv").open("w",newline="") as fp:
    w=csv.DictWriter(fp,fieldnames=list(gate_out[0])); w.writeheader(); w.writerows(gate_out)
fig,ax=plt.subplots(figsize=(6.6,4.0))
for method in ["AEFC-no-Gate","AEFC-Full"]:
    rr=[r for r in gate_out if r["method"]==method]
    ax.plot([r["alpha"] for r in rr],[r["pooled_false_authorization_rate"] for r in rr],marker="o",label=method)
ax.set_xlabel("Targeted unsafe-proposal blend alpha"); ax.set_ylabel("Pooled false-authorization rate"); ax.set_ylim(-0.02,1.02); ax.legend(); fig.tight_layout(); fig.savefig(GEN/"E2_gate_false_authorization.pdf"); plt.close(fig)

# ---------- E3: fairness-calibrated FedRL baselines ----------
cal_g=by(cal,"method"); e3=[]
metrics=["max_belief_distortion","max_policy_drift","recovery_time","min_cbf_margin"]
for method in ["FedRL","FedRL-Clip","RobustFedRL-Clip","AEFC-Full"]:
    rr=sorted(cal_g[(method,)],key=lambda r:int(r["seed"]))
    row={"method":method}
    for metric in metrics:
        row[metric],row[metric+"_lo"],row[metric+"_hi"]=mean_ci([f(r,metric) for r in rr])
    row["safety_violation_rate"]=sum(f(r,"safety_violation") for r in rr)/20
    row["false_authorization_rate"]=float(np.mean([f(r,"false_authorization_rate") for r in rr]))
    e3.append(row)
with (GEN/"E3_baseline_calibration.csv").open("w",newline="") as fp:
    w=csv.DictWriter(fp,fieldnames=list(e3[0])); w.writeheader(); w.writerows(e3)
full=sorted(cal_g[("AEFC-Full",)],key=lambda r:int(r["seed"])); stat=[]
for metric in metrics:
    ps=[]; tmp=[]; x=np.asarray([f(r,metric) for r in full])
    for method in ["FedRL","FedRL-Clip","RobustFedRL-Clip"]:
        y=np.asarray([f(r,metric) for r in sorted(cal_g[(method,)],key=lambda r:int(r["seed"]))]); p=paired_p(x,y)
        ps.append(p); tmp.append((method,float(np.median(x-y)),p))
    for (method,med,p),pa in zip(tmp,holm(ps)):
        stat.append({"metric":metric,"comparison":method,"median_paired_full_minus_comparison":med,"wilcoxon_p":p,"holm_adjusted_p":float(pa)})
with (GEN/"E3_baseline_calibration_statistics.csv").open("w",newline="") as fp:
    w=csv.DictWriter(fp,fieldnames=list(stat[0])); w.writeheader(); w.writerows(stat)

# quantify how much the old method-dependent-RNG FedRL magnitude differs from fair-CRN FedRL.
old_fed=[r for r in main_mech if r["scenario"]=="joint" and r["method"]=="FedRL"]
old_policy=float(np.mean([f(r,"max_policy_drift") for r in old_fed])); fair_policy=[r for r in e3 if r["method"]=="FedRL"][0]["max_policy_drift"]
old_belief=float(np.mean([f(r,"max_belief_distortion") for r in old_fed])); fair_belief=[r for r in e3 if r["method"]=="FedRL"][0]["max_belief_distortion"]
(GEN/"E3_original_vs_fairness_calibration.json").write_text(json.dumps({"old_mechanism_fedrl_policy_drift":old_policy,"fair_crn_fedrl_policy_drift":fair_policy,"old_mechanism_fedrl_belief_distortion":old_belief,"fair_crn_fedrl_belief_distortion":fair_belief},indent=2)+"\n")

# ---------- E4: cross-backend direction/ranking consistency ----------
mech_joint={m:[r for r in main_mech if r["scenario"]=="joint" and r["method"]==m] for m in METHODS7}
native_joint={m:[r for r in main_native if r["scenario"]=="joint" and r["method"]==m] for m in METHODS7}
e4=[]
for label,mk,nk in [("belief_distortion","max_belief_distortion","max_belief_distortion"),("policy_movement","max_policy_drift","max_policy_drift"),("continuous_margin","min_cbf_margin","min_native_margin")]:
    a=[float(np.mean([f(r,mk) for r in mech_joint[m]])) for m in METHODS7]
    b=[float(np.mean([f(r,nk) for r in native_joint[m]])) for m in METHODS7]
    rho,p=safe_spearman(a,b)
    e4.append({"metric":label,"spearman_rho":rho,"p_value":p,"mechanism_means":json.dumps(dict(zip(METHODS7,a))),"native_means":json.dumps(dict(zip(METHODS7,b)))})
with (GEN/"E4_cross_backend_consistency.csv").open("w",newline="") as fp:
    w=csv.DictWriter(fp,fieldnames=list(e4[0])); w.writeheader(); w.writerows(e4)

# ---------- E5: one-at-a-time native sensitivity ----------
def param_rows(kind, values, default_value):
    out=[]
    # default is authoritative Main Experiment joint AEFC-Full.
    base=[r for r in main_native if r["scenario"]=="joint" and r["method"]=="AEFC-Full"]
    sets=[(default_value,base)]
    for value,tag in values:
        sets.append((value,read_csv(ROOT/"supplemental_native"/kind/tag/"summary.csv")))
    for value,rr in sorted(sets,key=lambda x:x[0]):
        if len(rr)!=20: raise SystemExit(f"E5 {kind}={value} requires 20 seeds")
        row={"parameter":kind,"value":value}
        for metric in ["min_native_margin","max_belief_distortion","max_policy_drift","shield_intervention_ratio","recovery_time"]:
            row[metric],row[metric+"_lo"],row[metric+"_hi"]=mean_ci([f(r,metric) for r in rr])
        out.append(row)
    return out

e5=[]
e5 += param_rows("qmin",[(0.10,"0p1"),(0.30,"0p3")],0.18)
e5 += param_rows("rho",[(0.40,"0p4"),(0.70,"0p7")],0.55)
e5 += param_rows("trust",[(0.020,"0p02"),(0.060,"0p06")],0.035)
with (GEN/"E5_parameter_sensitivity.csv").open("w",newline="") as fp:
    w=csv.DictWriter(fp,fieldnames=list(e5[0])); w.writeheader(); w.writerows(e5)
for kind,xlabel in [("qmin","q_min"),("rho","rho_max"),("trust","Delta_theta")]:
    rr=[r for r in e5 if r["parameter"]==kind]
    fig,ax=plt.subplots(figsize=(6.4,3.8))
    ax.plot([r["value"] for r in rr],[r["max_policy_drift"] for r in rr],marker="o",label="Policy movement")
    ax.plot([r["value"] for r in rr],[r["max_belief_distortion"] for r in rr],marker="s",label="Belief distortion")
    ax.set_xlabel(xlabel); ax.set_ylabel("Integrity metric"); ax.legend(); fig.tight_layout(); fig.savefig(GEN/f"E5_{kind}_sensitivity.pdf"); plt.close(fig)

# ---------- professional auto-LaTeX analysis ----------
def find_e1(gamma,method): return next(r for r in e1 if r["gamma"]==gamma and r["method"]==method)
def find_e3(method): return next(r for r in e3 if r["method"]==method)
def padj(metric,method): return next(r["holm_adjusted_p"] for r in stat if r["metric"]==metric and r["comparison"]==method)

lines=[]
lines.append(r"\subsection{Supplemental Robustness and Component Validation}")
lines.append(r"The authoritative $7\times4\times20$ native matrix is kept unchanged. The following experiments are supplemental and use separate output directories and provenance records. Their purpose is to test robustness, component-local causal effects, baseline fairness, backend consistency, and parameter sensitivity rather than to retune the Main Experiment.")

full05=find_e1(0.5,"AEFC-Full"); full20=find_e1(2,"AEFC-Full"); fed20=find_e1(2,"FedRL")
lines.append("\\textbf{E1---Attack-intensity robustness.} The supplemental controller reproduces the Main Experiment at $\\gamma=1$ to numerical tolerance (maximum summary discrepancy %.2e). Across $\\gamma\\in\\{0,0.5,1,1.5,2\\}$, AEFC-Full changes from minimum native margin %s at $\\gamma=0.5$ to %s at $\\gamma=2$, while its maximum belief distortion changes from %s to %s. At $\\gamma=2$, FedRL has minimum native margin %s and maximum belief distortion %s. The interpretation therefore relies on continuous degradation curves rather than manufacturing a binary safety-violation gap." % (max_repro,fmt(full05["min_native_margin"]),fmt(full20["min_native_margin"]),fmt(full05["max_belief_distortion"]),fmt(full20["max_belief_distortion"]),fmt(fed20["min_native_margin"]),fmt(fed20["max_belief_distortion"])))

pworst=patbu[-1]
lines.append("\\textbf{E2---PATBU and Gate local causal ablations.} PATBU is evaluated at the evidence-to-belief boundary, not by final plant safety. At $\\gamma=2$, AEFC-Full and AEFC-no-PATBU have mean maximum belief distortions %s and %s, respectively, with paired median Full-minus-no-PATBU difference %s (Holm-adjusted $p=%s$). The Gate experiment injects a targeted unsafe learned proposal before authorization while leaving post-shield corruption unchanged; this separates action authorization from the downstream shield." % (fmt(pworst["full_mean_belief_distortion"]),fmt(pworst["no_patbu_mean_belief_distortion"]),fmt(pworst["median_paired_full_minus_no_patbu"]),f'{pworst["holm_adjusted_p"]:.3g}'))
full_gate=next(r for r in gate_out if r["alpha"]==1.0 and r["method"]=="AEFC-Full"); nog_gate=next(r for r in gate_out if r["alpha"]==1.0 and r["method"]=="AEFC-no-Gate")
lines.append("At the strongest Gate stress ($\\alpha=1$), AEFC-Full has pooled false-authorization rate %s versus %s without the Gate; the corresponding mean shield-intervention ratios are %s and %s. If both FAR values are identical, the result should be interpreted as an insufficiently discriminating authorization stress rather than as evidence of no Gate contribution." % (fmt(full_gate["pooled_false_authorization_rate"]),fmt(nog_gate["pooled_false_authorization_rate"]),fmt(full_gate["mean_shield_intervention_ratio"]),fmt(nog_gate["mean_shield_intervention_ratio"])))

fed=find_e3("FedRL"); clip=find_e3("FedRL-Clip"); robustclip=find_e3("RobustFedRL-Clip"); fullc=find_e3("AEFC-Full")
lines.append("\\textbf{E3---Fairness-calibrated baselines.} With method-independent common random numbers, FedRL, FedRL-Clip, RobustFedRL-Clip, and AEFC-Full have mean maximum policy movements %s, %s, %s, and %s, respectively. The Full-versus-RobustFedRL-Clip comparison has Holm-adjusted $p=%s$. Thus the security comparison is not based solely on the numerically aggressive unconstrained FedRL update." % (fmt(fed["max_policy_drift"]),fmt(clip["max_policy_drift"]),fmt(robustclip["max_policy_drift"]),fmt(fullc["max_policy_drift"]),f'{padj("max_policy_drift","RobustFedRL-Clip"):.3g}'))

cons=', '.join([f'{r["metric"]}: rho={r["spearman_rho"]:.3f}' if math.isfinite(r["spearman_rho"]) else f'{r["metric"]}: uninformative (constant)' for r in e4])
lines.append("\\textbf{E4---Cross-backend consistency.} Absolute metric magnitudes are not equated across the reduced-order and native plants. Instead, method-level direction/ranking consistency is assessed with Spearman correlation: %s. Metrics that are constant in one backend are explicitly treated as uninformative rather than assigned an artificial correlation." % cons)

for kind,label in [("qmin",r"$q_{\min}$"),("rho",r"$\rho_{\max}$"),("trust",r"$\Delta_\theta$")]:
    rr=[r for r in e5 if r["parameter"]==kind]; vals=', '.join([f'{r["value"]:.3g}: drift={r["max_policy_drift"]:.4f}, margin={r["min_native_margin"]:.4f}' for r in rr])
    lines.append(f"\\textbf{{E5---{label} sensitivity.}} {vals}.")
lines.append(r"These supplemental experiments do not replace the authoritative 560-run Main Experiment. They are used to qualify component attribution and robustness claims and are reported with the same code-to-claim discipline.")
(GEN/"Supplemental-Results-Auto.tex").write_text("\n\n".join(lines)+"\n")

manifest={"generated_from":["results/ieee39_joint20/summary.csv","results/joint20/summary.csv","results/supplemental_native","results/supplemental_mechanism/calibration/summary.csv"],"main_experiment_untouched":True,"gamma1_reproduction_max_abs_difference":max_repro,"statistics":"paired Wilcoxon + Holm; Spearman ranking consistency; 95% normal CIs for continuous summaries","experiments":["E1 attack intensity","E2 PATBU/Gate local causal ablation","E3 fairness-calibrated baselines","E4 cross-backend consistency","E5 parameter sensitivity"]}
(GEN/"generation_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"status":"ok","gamma1_reproduction_max_abs_difference":max_repro,"generated":str(GEN)},indent=2))
