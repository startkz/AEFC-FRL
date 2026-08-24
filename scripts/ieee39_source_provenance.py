#!/usr/bin/env python3
"""Parse legacy IEEE39 MDL facts without loading ARTEMIS/RT-LAB libraries."""
from __future__ import annotations
import argparse, hashlib, json, re, zipfile
from pathlib import Path

OPEN_RE = re.compile(r'^\s*([A-Za-z][A-Za-z0-9_]*)\s*\{\s*$')
PARAM_RE = re.compile(r'^\s*([^\s{}]+)\s+(.*)$')
Q_RE = re.compile(r'^"(.*)"\s*$')
GEN_RE = re.compile(r'/GT\s*(\d+)')


def decode_value(raw: str) -> str:
    raw = raw.strip()
    m = Q_RE.match(raw)
    return m.group(1).replace('\\n', ' ') if m else raw


def as_float(expr: str):
    try:
        return float(expr)
    except Exception:
        return None


def parse_blocks(text: str):
    stack, blocks = [], []
    model_name = 'IEEE39bus'
    for lineno, line in enumerate(text.splitlines(), 1):
        m = OPEN_RE.match(line)
        if m:
            kind = m.group(1)
            parent_system = ''
            for ctx in reversed(stack):
                if ctx['kind'] == 'System':
                    parent_system = ctx.get('path', '')
                    break
            if not parent_system:
                parent_system = model_name
            ctx = {'kind': kind, 'params': {}, 'line': lineno,
                   'parent_system': parent_system, 'path': ''}
            if kind == 'System':
                for pctx in reversed(stack):
                    if pctx['kind'] == 'Block':
                        ctx['path'] = pctx.get('path') or pctx.get('parent_system', '')
                        break
            stack.append(ctx)
            continue
        if re.match(r'^\s*}\s*$', line):
            if stack:
                ctx = stack.pop()
                if ctx['kind'] == 'Block':
                    blocks.append(ctx)
            continue
        if not stack:
            continue
        pm = PARAM_RE.match(line)
        if not pm:
            continue
        key, value = pm.group(1), decode_value(pm.group(2))
        ctx = stack[-1]
        ctx['params'][key] = value
        if key == 'Name':
            if ctx['kind'] == 'Model':
                model_name = value
            elif ctx['kind'] == 'System' and not ctx.get('path'):
                ctx['path'] = value
            elif ctx['kind'] == 'Block':
                ctx['path'] = (ctx.get('parent_system') or model_name).rstrip('/') + '/' + value
    return blocks


def generator_from_path(path: str):
    m = GEN_RE.search(path)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--expected-git-blob', default='41db586d592851c4a81205a4cc5c7c770b7a0c48')
    args = ap.parse_args()
    with zipfile.ZipFile(args.zip) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith('ieee39bus.mdl')]
        if len(members) != 1:
            raise SystemExit(f'Expected exactly one IEEE39bus.mdl, found {members}')
        member = members[0]
        raw = zf.read(member)
    text = raw.decode('windows-1252', errors='replace')
    blocks = parse_blocks(text)

    generators = []
    for g in range(1, 11):
        local = [b for b in blocks if generator_from_path(b['path']) == g]
        machines = [b for b in local if 'synchronous machine' in
                    (b['params'].get('SourceType','') + ' ' + b['params'].get('SourceBlock','')).lower()]
        buses = [b for b in local if 'load flow bus' in
                 (b['params'].get('SourceType','') + ' ' + b['params'].get('SourceBlock','')).lower()]
        if len(machines) != 1:
            raise SystemExit(f'G{g}: expected exactly one synchronous machine, got {[b["path"] for b in machines]}')
        if len(buses) != 1:
            raise SystemExit(f'G{g}: expected exactly one generator Load Flow Bus, got {[b["path"] for b in buses]}')
        m, v = machines[0], buses[0]
        vb, vlf = as_float(v['params'].get('Vbase','')), as_float(v['params'].get('VLF',''))
        if vb is None or vb <= 0 or vlf is None:
            raise SystemExit(f'G{g}: invalid Vbase/VLF')
        generators.append({
            'generator': g,
            'machine_path': m['path'],
            'machine_source': m['params'].get('SourceBlock',''),
            'machine_source_type': m['params'].get('SourceType',''),
            'nominal_parameters': m['params'].get('NominalParameters',''),
            'mechanical_parameters': m['params'].get('Mechanical',''),
            'initial_conditions': m['params'].get('InitialConditions',''),
            'rotor_type': m['params'].get('RotorType',''),
            'loadflow_path': v['path'],
            'vbase_expr': v['params'].get('Vbase',''), 'vbase_volts': vb,
            'vlf_expr': v['params'].get('VLF',''), 'vlf_pu': vlf,
            'vref_expr': v['params'].get('Vref',''),
        })

    opcomm = []
    for b in blocks:
        if b['params'].get('SourceBlock') == 'rtlab/OpComm':
            p = b['params']
            opcomm.append({'path': b['path'], 'source_type': p.get('SourceType',''),
                           'st': p.get('st',''), 'subsys_rate': p.get('subsys_rate',''),
                           'nbport': p.get('nbport',''), 'synchronization': p.get('Synchronization',''),
                           'interpolation': p.get('Interpolation','')})
    if len(opcomm) != 8:
        raise SystemExit(f'Expected 8 source OpComm blocks, got {len(opcomm)}')

    bp = []
    m = re.search(r'BreakpointsForDimension1\s+"\[\s*48\.00([^\"]*)\]"', text)
    if m:
        bp = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', '48.00' + m.group(1))]
    minv = [float(x) for x in re.findall(r'MinimumVoltage\s+"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"', text)]
    out = {
        'schema': 'aefc.ieee39.source_provenance.v1',
        'parser': 'raw_mdl_brace_stack_no_library_loading',
        'source_zip': str(args.zip).replace('\\','/'),
        'source_git_blob_expected': args.expected_git_blob,
        'source_mdl_member': member,
        'source_mdl_sha256': hashlib.sha256(raw).hexdigest(),
        'nominal_frequency_hz': 50.0,
        'native_step_s': 25e-6,
        'generators': generators,
        'opcomm': opcomm,
        'native_limits': {
            'load_shedding_breakpoints_hz': bp,
            'dynamic_load_minimum_voltage_values_pu': sorted(set(minv)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'IEEE39 source provenance: generators={len(generators)} opcomm={len(opcomm)} -> {args.output}')
    for g in generators:
        print(f"  G{g['generator']}: Vbase={g['vbase_volts']:.9g} VLF={g['vlf_pu']:.9g} machine={g['machine_path']}")

if __name__ == '__main__':
    main()
