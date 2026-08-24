function ieee39_dpl_equivalence_audit
% Read-only equivalence audit for the 34 legacy ARTEMIS DPL replacements.
% This audit does not tune the plant. It distinguishes parameter/units/
% topology preservation from the harder algorithm/initialization equivalence
% between the legacy ARTEMIS travelling-wave implementation and R2024a SPS.

repoRoot=pwd;
outRoot=fullfile(repoRoot,'build','ieee39_r2024a');
sourceDir=fullfile(outRoot,'source','model');
sourceMdl=fullfile(sourceDir,'IEEE39bus.mdl');
modelPath=fullfile(outRoot,'migrated','IEEE39bus_R2024a.slx');
migPath=fullfile(outRoot,'migration_report.json');
assert(exist(sourceMdl,'file')==2,'Source IEEE39bus.mdl not found.');
assert(exist(modelPath,'file')==2,'Migrated IEEE39 model not found.');
assert(exist(migPath,'file')==2,'Migration report not found.');
mig=jsondecode(fileread(migPath));
assert(isfield(mig,'line_replacements') && numel(mig.line_replacements)==34, ...
    'Migration report must contain 34 line replacements.');

addpath(sourceDir);
oldDir=pwd; dirCleanup=onCleanup(@()restore_dir(oldDir)); %#ok<NASGU>
cd(sourceDir);
% Resolve the released line-length expressions in the same workspace used by
% the model callbacks. The script contains only released model constants.
run('IEEE39BusLineLength.m');
[~,mdl,~]=fileparts(modelPath);
load_system(modelPath);
modelCleanup=onCleanup(@()safe_close(mdl)); %#ok<NASGU>

src=parse_artemis_line_records(sourceMdl);
assert(numel(src)==34,'Expected 34 ARTEMIS DPL records, found %d.',numel(src));
Ts=25e-6;
records=cell(1,numel(src));
paramPass=true; physicalPass=true; topologyObservable=true; topologyPass=true;
for k=1:numel(src)
    s=src(k);
    % Use the exact in-place block path recorded by the migration itself.
    % Do not depend on release-specific SourceType/SourceBlock strings of the
    % native SPS mask, which changed across MATLAB releases.
    b=block_from_migration_report(mig.line_replacements,s.Name,mdl);
    assert(getSimulinkBlockHandle(b)>0,'Migrated SPS DPL path does not exist: %s',b);

    m=struct;
    m.Resistance=get_param(b,'Resistance');
    m.Inductance=get_param(b,'Inductance');
    m.Capacitance=get_param(b,'Capacitance');
    m.Length=get_param(b,'Length');
    m.Frequency=get_param(b,'Frequency');
    try, m.Phases=get_param(b,'Phases'); catch, m.Phases=''; end
    try, m.Measurements=get_param(b,'Measurements'); catch, m.Measurements=''; end

    sr=resolve_numeric(s.Resistance,b); mr=resolve_numeric(m.Resistance,b);
    sl=resolve_numeric(s.Inductance,b); ml=resolve_numeric(m.Inductance,b);
    sc=resolve_numeric(s.Capacitance,b); mc=resolve_numeric(m.Capacitance,b);
    sx=resolve_numeric(s.Length,b); mx=resolve_numeric(m.Length,b);
    sf=resolve_numeric(s.Frequency,b); mf=resolve_numeric(m.Frequency,b);

    shapeOK=numel(sr)==2 && numel(sl)==2 && numel(sc)==2 && isscalar(sx) && isscalar(sf);
    eqR=near_equal(sr,mr); eqL=near_equal(sl,ml); eqC=near_equal(sc,mc);
    eqLen=near_equal(sx,mx); eqF=near_equal(sf,mf);
    eq=shapeOK && eqR && eqL && eqC && eqLen && eqF;
    paramPass=paramPass && eq;

    posOK=shapeOK && all(isfinite([sr sl sc sx sf])) && all(sr>=0) && all(sl>0) && all(sc>0) && sx>0 && abs(sf-50)<1e-12;
    if posOK
        v1=1/sqrt(sl(1)*sc(1)); v0=1/sqrt(sl(2)*sc(2));
        tau1=sx/v1; tau0=sx/v0;
        z1=sqrt(sl(1)/sc(1)); z0=sqrt(sl(2)/sc(2));
        delaySamples=tau1/Ts;
        % Released IEEE39BusLineLength.m explicitly targets an overhead-line
        % positive-sequence propagation speed close to 2.9e5 km/s.
        propagationOK=v1>=2.0e5 && v1<=4.0e5 && tau1>0 && delaySamples>0;
    else
        v1=NaN; v0=NaN; tau1=NaN; tau0=NaN; z1=NaN; z0=NaN; delaySamples=NaN; propagationOK=false;
    end
    physicalPass=physicalPass && posOK && propagationOK;

    [pcOK,pcObservable,pcSummary]=port_connectivity(b);
    topologyObservable=topologyObservable && pcObservable;
    if pcObservable, topologyPass=topologyPass && pcOK; end

    records{k}=struct( ...
        'name',s.Name,'source_simulation_mode',s.SimulationMode,'migrated_block',b, ...
        'source_raw',s,'migrated_raw',m, ...
        'source_numeric',struct('R_ohm_per_km',sr,'L_h_per_km',sl,'C_f_per_km',sc,'length_km',sx,'frequency_hz',sf), ...
        'migrated_numeric',struct('R_ohm_per_km',mr,'L_h_per_km',ml,'C_f_per_km',mc,'length_km',mx,'frequency_hz',mf), ...
        'parameter_equivalent',eq,'shape_positive_zero_sequence',shapeOK, ...
        'positive_sequence_velocity_km_s',v1,'zero_sequence_velocity_km_s',v0, ...
        'positive_sequence_delay_s',tau1,'zero_sequence_delay_s',tau0, ...
        'positive_sequence_delay_samples',delaySamples, ...
        'positive_sequence_characteristic_impedance_ohm',z1, ...
        'zero_sequence_characteristic_impedance_ohm',z0, ...
        'propagation_plausible',propagationOK, ...
        'port_connectivity_observable',pcObservable,'port_connectivity_complete',pcOK, ...
        'port_connectivity',pcSummary);
end

vel=cellfun(@(x)x.positive_sequence_velocity_km_s,records);
dly=cellfun(@(x)x.positive_sequence_delay_samples,records);
report=struct;
report.release=version('-release');
report.read_only=true;
report.source_line_count=numel(src);
report.migrated_line_count=numel(records);
report.block_location_source='migration_report.line_replacements[].block';
report.parameter_equivalence_passed=paramPass;
report.sequence_order='[positive, zero]';
report.units=struct('resistance','ohm/km','inductance','H/km','capacitance','F/km','length','km');
report.positive_sequence_velocity_range_km_s=[min(vel) max(vel)];
report.positive_sequence_delay_sample_range=[min(dly) max(dly)];
report.propagation_plausibility_passed=physicalPass;
report.topology_connectivity_observable=topologyObservable;
report.topology_connectivity_passed=topologyPass;
report.source_algorithm='ARTEMIS model (legacy released block setting)';
report.migrated_algorithm='native R2024a SPS Distributed Parameters Line';
report.algorithm_equivalence_demonstrated=false;
report.initialization_equivalence_demonstrated=false;
report.dynamic_equivalence_ready=false;
report.interpretation=['Matching R/L/C/length/frequency and connected ports establish parameter-level migration fidelity only. ' ...
    'They do not establish transient equivalence because the legacy ARTEMIS implementation and native SPS line can differ in numerical algorithm, network decoupling, and initialization.'];
report.lines=records;
report.timestamp_utc=char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
write_report(fullfile(outRoot,'dpl_equivalence_audit.json'),report);

fprintf('IEEE39 DPL audit: lines=%d param=%d propagation=%d topology_obs=%d topology=%d dynamic_ready=%d\n', ...
    numel(src),paramPass,physicalPass,topologyObservable,topologyPass,report.dynamic_equivalence_ready);
fprintf('  v_pos=[%.6g, %.6g] km/s delay/Ts=[%.6g, %.6g]\n',min(vel),max(vel),min(dly),max(dly));
if ~paramPass || ~physicalPass || (topologyObservable && ~topologyPass)
    error('AEFC:IEEE39DPLEquivalenceFailed','DPL parameter/propagation/topology audit failed.');
end
end

function b=block_from_migration_report(repls,name,mdl)
hits={};
for i=1:numel(repls)
    p=repls(i).block;
    if endsWith(p,['/' name])
        hits{end+1}=p; %#ok<AGROW>
    end
end
assert(numel(hits)==1,'Migration report path not unique for %s (found %d).',name,numel(hits));
p=hits{1}; prefix='IEEE39bus';
assert(startsWith(p,prefix),'Unexpected migration block path: %s',p);
b=[mdl extractAfter(p,strlength(prefix))];
end

function records=parse_artemis_line_records(path)
txt=fileread(path); parts=regexp(txt,'(?m)^\s*Block \{','split');
records=struct('Name',{},'SimulationMode',{},'Resistance',{},'Inductance',{},'Capacitance',{},'Length',{},'Frequency',{},'Phases',{},'Measurements',{});
for i=1:numel(parts)
    p=parts{i};
    if contains(p,'SourceBlock') && contains(p,'op_dpl_lib/Distributed Parameters Line')
        r.Name=qvalue(p,'Name'); r.SimulationMode=qvalue(p,'SimulationMode');
        r.Resistance=qvalue(p,'Resistance'); r.Inductance=qvalue(p,'Inductance');
        r.Capacitance=qvalue(p,'Capacitance'); r.Length=qvalue(p,'Length');
        r.Frequency=qvalue(p,'Frequency'); r.Phases=qvalue(p,'Phases');
        r.Measurements=qvalue(p,'Measurements'); records(end+1)=r; %#ok<AGROW>
    end
end
end

function v=qvalue(txt,key)
t=regexp(txt,['(?m)^\s*' regexptranslate('escape',key) '\s+"([^"]*)"'],'tokens','once');
assert(~isempty(t),'Missing MDL parameter %s.',key); v=t{1};
end

function v=resolve_numeric(expr,b)
try
    v=slResolve(expr,b);
catch
    try, v=evalin('base',expr); catch ME, error('Could not resolve "%s" on %s: %s',expr,b,ME.message); end
end
v=double(v(:).');
end

function tf=near_equal(a,b)
if numel(a)~=numel(b) || any(~isfinite(a)) || any(~isfinite(b)), tf=false; return; end
den=max(abs(a),1e-15); tf=all(abs(a-b)./den <= 1e-10);
end

function [ok,observable,s]=port_connectivity(b)
ok=false; observable=false; s=struct('port_count',0,'connected_port_count',0,'details',{{}});
try
    pc=get_param(b,'PortConnectivity');
    if isempty(pc), return; end
    observable=true; details=cell(1,numel(pc)); connected=0;
    for i=1:numel(pc)
        src=[]; dst=[]; typ=''; pos=[];
        try, src=pc(i).SrcBlock; catch, end
        try, dst=pc(i).DstBlock; catch, end
        try, typ=pc(i).Type; catch, end
        try, pos=pc(i).Position; catch, end
        c=(~isempty(src) && any(src~=-1)) || (~isempty(dst) && any(dst~=-1));
        connected=connected+c;
        details{i}=struct('index',i,'type',typ,'position',pos,'src_block',src,'dst_block',dst,'connected',c);
    end
    s=struct('port_count',numel(pc),'connected_port_count',connected,'details',{details});
    % A three-phase DPL exposes six electrical terminals. If R2024a exposes
    % connectivity through PortConnectivity, all six must remain connected.
    ok=(numel(pc)>=6 && connected>=6);
catch
    observable=false; ok=false;
end
end

function restore_dir(p), try, cd(p); catch, end, end
function safe_close(mdl), try, if bdIsLoaded(mdl), close_system(mdl,0); end, catch, end, end
function write_report(p,r), fid=fopen(p,'w'); assert(fid>0); fwrite(fid,jsonencode(r,'PrettyPrint',true)); fclose(fid); end
