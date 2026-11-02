#!/usr/bin/env python3
"""Validate standard IEEE39/118 AC grid cases via actual power flow.
This is model/interface validation, not AEFC learning or dynamic recovery.
"""
import argparse, json, sys, hashlib, platform
from pathlib import Path
import numpy as np

def probe(case, seed=42):
    import pandapower as pp
    import pandapower.networks as pn
    net=(pn.case39() if case==39 else pn.case118())
    assert len(net.bus)==case, (case,len(net.bus))
    original_load=net.load.p_mw.copy()
    groups=np.array_split(np.arange(len(net.load)),min(10,len(net.load)))
    out={'case':case,'backend':'pandapower AC power flow, PYPOWER IEEE case',
         'buses':len(net.bus),'loads':len(net.load),'lines':len(net.line),
         'trafos':len(net.trafo),'generators':len(net.gen),
         'seed':seed,'python':platform.python_version(),
         'pandapower':pp.__version__,'dynamic_frequency_validated':False,
         'epochs':[]}
    rng=np.random.default_rng(seed)
    for j, restoration in enumerate((0.65,0.75,0.85,0.95,1.0)):
        # Deterministic, paired state for each model in a fixed recovery sweep.
        # These load curtailments are *simulated controlled curtailments*, not
        # evidence of a physical network being compromised.
        scaling=np.zeros(len(net.load))
        for group in groups:
            scaling[group]=np.clip(restoration+rng.normal(0,0.005),0,1)
        net.load.loc[:,'scaling']=scaling
        converged=True;failure=None
        try:pp.runpp(net,init='flat',numba=False,max_iteration=30)
        except Exception as e:converged=False;failure=type(e).__name__
        if converged:
            vm=net.res_bus.vm_pu.to_numpy()
            loading=np.r_[net.res_line.loading_percent.to_numpy(),
                           net.res_trafo.loading_percent.to_numpy()]
            volt=float(np.maximum(.9-vm,0).sum()+np.maximum(vm-1.1,0).sum())
            overload=float(np.maximum(loading-100,0).sum())
            served_pct=100*float((original_load.to_numpy()*scaling).sum()/max(original_load.sum(),1e-10))
        else:volt=None;overload=None;served_pct=None
        out['epochs'].append({'restoration_input':restoration,
                              'pf_converged':converged,
                              'pf_error_type':failure,
                              'served_pct':served_pct,
                              'voltage_violation_pu_sum':volt,
                              'thermal_excess_pct_sum':overload})
    out['all_power_flows_converged']=all(r['pf_converged'] for r in out['epochs'])
    # Explicit source of the safety limits; these are study limits,
    # not universal grid standards.
    out['safety_limits']={'v_min_pu':.9,'v_max_pu':1.1,
                          'line_trafo_loading_max_pct':100}
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,
                    choices=[39,118],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    result=probe(args.case,args.seed)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'case':args.case,'all_power_flows_converged':result['all_power_flows_converged'],
          'last':result['epochs'][-1]}))
    if not result['all_power_flows_converged']:sys.exit(2)
if __name__=='__main__':main()
