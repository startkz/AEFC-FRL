"""Paired seed-level bootstrap analysis; never treat held-out episodes as seeds."""
import csv, json, sys, argparse
from pathlib import Path
import numpy as np
METHODS=('AEFC-FRL','FedRL','AEFC-no-adapter','AEFC-no-proto')
METRICS=('heldout_success_fraction','heldout_safe_dwell5_fraction','heldout_mean_queue','heldout_mean_return','application_uplink_bytes')
def ci(x,seed=91723,B=10000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed)
    res=x[rng.integers(0,len(x),(B,len(x)))].mean(axis=1)
    return float(res.mean()),float(np.quantile(res,.025)),float(np.quantile(res,.975))
def main():
    p=argparse.ArgumentParser();p.add_argument('--per-seed',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    rows=list(csv.DictReader(open(a.per_seed))); assert len(rows)==60
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    stats=[];paired=[]
    for m in METHODS:
        v=sorted((r for r in rows if r['method']==m),key=lambda x:int(x['seed']))
        assert len(v)==15
        for metric in METRICS:
            x=np.array([float(r[metric]) for r in v]);_,lo,hi=ci(x)
            stats.append({'method':m,'metric':metric,'n_seeds':15,'mean':x.mean(),'ci95_low':lo,'ci95_high':hi})
    for m in METHODS[1:]:
        for metric in METRICS:
            d={name:{int(r['seed']):float(r[metric]) for r in rows if r['method']==name} for name in ('AEFC-FRL',m)}
            assert set(d['AEFC-FRL'])==set(d[m])
            diff=np.array([d['AEFC-FRL'][s]-d[m][s] for s in sorted(d[m])]);_,lo,hi=ci(diff)
            paired.append({'comparison':'AEFC-FRL minus '+m,'metric':metric,'paired_n_seeds':15,
                           'mean_difference':diff.mean(),'ci95_low':lo,'ci95_high':hi})
    for name,data in [('summary.csv',stats),('paired.csv',paired)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,list(data[0]));writer.writeheader();writer.writerows(data)
    print('Wrote seed-level summary and matched-seed intervals: no independence inflation')
if __name__=='__main__':main()