#!/usr/bin/env python3
"""Check real per-run CSV metadata without manufacturing results."""
import csv,sys,math
from pathlib import Path
REQUIRED={'method','seed','scenario','clients','malicious_clients',
          'train_rounds','train_episodes','recovered_load_pct',
          'uplink_bytes','safe_success','reached_target','target_round'}
def validate(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames is None: raise ValueError('Empty CSV')
        miss=REQUIRED-set(reader.fieldnames)
        if miss: raise ValueError('Missing columns: '+str(sorted(miss)))
        rows=list(reader)
    if not rows: raise ValueError('No executed runs')
    ids=set()
    for r in rows:
        ident=(r['method'],r['scenario'],r['seed'])
        if ident in ids: raise ValueError('Duplicate run '+str(ident))
        ids.add(ident)
        n=int(r['clients']);m=int(r['malicious_clients'])
        t=int(r['train_rounds']);ep=int(r['train_episodes'])
        byt=int(r['uplink_bytes']);load=float(r['recovered_load_pct'])
        reached=int(r['reached_target']);end=int(r['target_round'])
        if not 0<=m<=n or n==0: raise ValueError('Invalid actual M/N')
        if t<=0 or ep<=0 or byt<0 or not math.isfinite(load) or not 0<=load<=100:
            raise ValueError('Invalid budget/traffic/load')
        if reached not in (0,1) or int(r['safe_success']) not in (0,1):
            raise ValueError('Invalid binary outcome')
        if reached and not 0<=end<=t:
            raise ValueError('Convergence round exceeds budget')
        if not reached and end not in (-1,t):
            raise ValueError('Censoring record required')
    print('VALID STRUCTURE:',len(rows),'runs;',len(ids),'unique IDs')
    print('NOT VERIFIED: simulator identity, per-run traces, optimizer and safety')
if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('usage: python review_2026/audit_runs.py runs.csv')
    validate(sys.argv[1])
