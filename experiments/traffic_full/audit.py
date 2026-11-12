"""Fail-closed independent acceptance for the 15-seed traffic AEFC/FedRL matrix."""
import argparse, csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np

SEEDS=(11,13,17,19,23,29,31,37,41,43,47,53,59,61,67)
METHODS=('AEFC-FRL','FedRL','AEFC-no-adapter','AEFC-no-proto')
FILES=('summary.json','rounds.jsonl','training_episodes.jsonl','heldout_episodes.jsonl','model.pt')

def readlines(path):
    with path.open() as f:
        for line in f:
            if line.strip():yield json.loads(line)

def validate_row(row):
    assert row['right_censored']==int(row['recovery_step']<0)
    assert row['safe_dwell5']==int(row['recovery_step']>=0)
    assert row['attack_window_success']==int(row['attack_window_recovery_step']>=1)
    assert row['unsafe_steps']>=0 and row['unsafe_steps']<=60
    assert row['recovery_step'] == -1 or 32 <= row['recovery_step'] <= 55
    assert row['attack_window_recovery_step']==-1 or 1<=row['attack_window_recovery_step']<=17
    # Independently reconstruct physical service, reward, congestion and both endpoints.
    steps=row.get('physical_trace')
    assert isinstance(steps,list) and len(steps)==60,'missing full physical trace'
    cid=int(row['client'])
    q=np.array([8.+cid,6.+0.5*cid]); phase=0; prev=0
    sums=[]; total_reward=0.; unsafe_count=0; historical=-1; dwell=-1; run=0
    for t,rec in enumerate(steps):
        step,qa,qb,phase_record,proposed,applied,arr_a,arr_b,reward=rec
        assert int(step)==t and proposed in (0,1) and applied in (0,1)
        expected_applied = prev if 15<=t<32 and cid%2==0 else proposed
        assert applied==expected_applied,'delayed actuator mismatch'
        assert phase_record==applied
        assert arr_a>=0 and arr_b>=0 and float(arr_a).is_integer() and float(arr_b).is_integer()
        service=np.array([5.4,1.]) if applied==0 else np.array([1.,5.4])
        q=np.maximum(0.,q+np.array([arr_a,arr_b])-service)
        assert np.allclose(q,[qa,qb],rtol=0,atol=1e-8),'physical queue mismatch'
        unsafe=bool(q.max()>35.)
        expect_reward=-float(q.sum())-0.6*int(applied!=phase)-2.*int(unsafe)
        assert abs(reward-expect_reward)<1e-8,'reward mismatch'
        total_reward+=reward; unsafe_count+=int(unsafe);sums.append(float(q.sum()))
        if 15<=t<32 and q.sum()<18 and historical<0:historical=t-15+1
        if t>=32 and q.sum()<18 and not unsafe:run+=1
        else:run=0
        if dwell<0 and run>=5:dwell=t-4
        phase=applied; prev=proposed
    assert row['attack_window_recovery_step']==historical
    assert row['recovery_step']==dwell
    assert row['unsafe_steps']==unsafe_count
    assert abs(row['return']-total_reward)<1e-8
    assert abs(row['avg_queue']-float(np.mean(sums)))<1e-8
    assert abs(row['terminal_queue']-sums[-1])<1e-8
    for k in ('return','avg_queue','terminal_queue'): assert math.isfinite(row[k])

def audit(root, expected_rounds=80, expected_local_eps=4, expected_eval=25):
    summaries=[]; errors=[]; manifest_hash=[]
    for method in METHODS:
        for seed in SEEDS:
            folder=root/method/str(seed)
            try:
                assert folder.exists(),f'absent {folder}'
                meta=json.loads((folder/'manifest.json').read_text())
                assert meta['method']==method and meta['seed']==seed
                assert all(f in meta['sha256'] for f in FILES)
                for file in FILES:
                    digest=hashlib.sha256((folder/file).read_bytes()).hexdigest()
                    assert meta['sha256'][file]==digest,(file,'sha256 mismatch')
                manifest_hash.append({'method':method,'seed':seed,'manifest_sha256':hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()})
                train=list(readlines(folder/'training_episodes.jsonl'))
                held=list(readlines(folder/'heldout_episodes.jsonl'))
                rounds=list(readlines(folder/'rounds.jsonl'))
                summary=json.loads((folder/'summary.json').read_text())
                assert len(train)==5*expected_rounds*expected_local_eps
                assert len(held)==5*expected_eval
                assert len(rounds)==expected_rounds
                assert summary['training_episodes']==len(train)
                assert summary['federated_rounds']==len(rounds)
                assert summary['heldout_episodes']==len(held)
                assert all(r['method']==method and r['seed']==seed for r in train+held)
                assert len({(r['client'],r['episode']) for r in train})==len(train)
                assert len({(r['client'],r['episode']) for r in held})==len(held)
                for x in train+held:validate_row(x)
                assert summary['train_censored_count']==sum(x['right_censored'] for x in train)
                assert summary['heldout_censored_count']==sum(x['right_censored'] for x in held)
                assert abs(summary['heldout_success_fraction']-np.mean([x['attack_window_success'] for x in held]))<1e-12
                assert abs(summary['heldout_safe_dwell5_fraction']-np.mean([x['safe_dwell5'] for x in held]))<1e-12
                assert abs(summary['heldout_mean_queue']-np.mean([x['avg_queue'] for x in held]))<1e-9
                assert summary['application_uplink_bytes']==sum(r['total_application_uplink_bytes'] for r in rounds)
                assert all(len(x['bytes_per_client'])==5 and sum(x['bytes_per_client'])==x['total_application_uplink_bytes'] for x in rounds)
                assert all(r['round']==i+1 and r['attacker_client']==seed%5 for i,r in enumerate(rounds))
                assert all(r['total_application_uplink_bytes']>0 and math.isfinite(r['applied_norm']) for r in rounds)
                summaries.append(summary)
            except Exception as e:errors.append({'method':method,'seed':seed,'error':str(e) or repr(e)})
    result={'passed':not errors and len(summaries)==len(SEEDS)*len(METHODS),
            'expected_cells':len(SEEDS)*len(METHODS),'audited_cells':len(summaries),
            'training_episode_count':sum(x['training_episodes'] for x in summaries),
            'heldout_episode_count':sum(x['heldout_episodes'] for x in summaries),
            'errors':errors,'platform':'no-oracle two-phase stochastic queue; NOT Simulink IEEE39',
            'provenance':'Independent validation of JSON/pt manifest integrity; not mathematical performance proof'}
    return result,summaries,manifest_hash

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--report',required=True)
    p.add_argument('--rounds',type=int,default=80);p.add_argument('--local-episodes',type=int,default=4)
    p.add_argument('--eval-episodes',type=int,default=25);a=p.parse_args()
    res,rows,hashes=audit(Path(a.root),a.rounds,a.local_episodes,a.eval_episodes)
    dest=Path(a.report);dest.mkdir(parents=True,exist_ok=True)
    (dest/'audit.json').write_text(json.dumps(res,indent=2)+'\n')
    for name,data in [('per_seed.csv',rows),('manifest_sha256.csv',hashes)]:
        if data:
            with (dest/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps(res));sys.exit(0 if res['passed'] else 2)
if __name__=='__main__':main()