function ieee39_migration_physical_gate
% Strict clean-plant gate for the R2024a migration.
% Uses raw-MDL observation provenance, including each V_bus_G* measurement
% block's own voltage base.  No FRL action is injected here.

repoRoot=pwd;
outRoot=fullfile(repoRoot,'build','ieee39_r2024a');
sourceDir=fullfile(outRoot,'source','model');
modelPath=fullfile(outRoot,'migrated','IEEE39bus_R2024a.slx');
provPath=fullfile(repoRoot,'build','ieee39_source_provenance.json');
assert(exist(modelPath,'file')==2,'Migrated IEEE39 model not found.');
assert(exist(provPath,'file')==2,'IEEE39 source provenance not found.');
prov=jsondecode(fileread(provPath));
assert(isfield(prov,'voltage_observations') && numel(prov.voltage_observations)==10, ...
    'Voltage-observation provenance v2 is required.');

addpath(sourceDir);
oldDir=pwd; dirCleanup=onCleanup(@()restore_dir(oldDir)); %#ok<NASGU>
cd(sourceDir);
[~,mdl,~]=fileparts(modelPath); load_system(modelPath);
modelCleanup=onCleanup(@()safe_close(mdl)); %#ok<NASGU>

report=struct;
report.release=version('-release');
report.source_schema=prov.schema;
report.source_git_blob=prov.source_git_blob_expected;
report.clean_only=true;
report.frl_actions_injected=false;
report.bounds=struct('wm_min_pu',0.5,'wm_max_pu',1.5,'voltage_min_pu',0.2,'voltage_max_pu',2.0);
report.horizons_s=[0.005 0.01 0.02 0.05];

% Validate the actual R2024a observation semantics before interpreting the
% numerical traces.  This prevents a Bus Selector/tag/base mismatch from being
% mislabeled as physical instability.
sem=cell(1,10); semPass=true; measBases=zeros(1,10);
for g=1:10
    wtag=sprintf('Wm_G%d',g); vtag=sprintf('V_bus_G%d',g);
    [wOK,wInfo]=audit_wm_tag(mdl,wtag);
    pv=prov.voltage_observations(g);
    assert(pv.generator==g && strcmp(pv.tag,vtag),'Voltage provenance ordering mismatch for G%d.',g);
    [vOK,vInfo]=audit_voltage_measurement(mdl,pv);
    measBases(g)=pv.vbase_volts;
    sem{g}=struct('generator',g,'wm',wInfo,'voltage',vInfo,'passed',wOK && vOK);
    semPass=semPass && wOK && vOK;
end
report.observation_semantics=sem;
report.observation_semantics_passed=semPass;
report.measurement_voltage_base_volts=measBases;
report.loadflow_voltage_base_volts=arrayfun(@(x)x.vbase_volts,prov.generators);
report.voltage_base_policy='Normalize raw phase-to-ground V_bus_G* by the source Three-Phase V-I Measurement Vbase/sqrt(3), not by generator Load Flow Bus Vbase.';

assert(semPass,'Observation-semantics gate failed; clean traces are not interpretable.');
taps=install_logging_taps(mdl); tapCleanup=onCleanup(@()remove_logging_taps(taps)); %#ok<NASGU>
runs=cell(1,numel(report.horizons_s));
for i=1:numel(report.horizons_s)
    runs{i}=run_clean(mdl,report.horizons_s(i),measBases,report.bounds);
end
report.clean_baseline=runs;
report.clean_baseline_passed=all(cellfun(@(x)x.passed,runs));
idx=find(~cellfun(@(x)x.passed,runs),1,'first');
if isempty(idx), report.first_unstable_horizon_s=[]; else, report.first_unstable_horizon_s=report.horizons_s(idx); end
report.full_stack_frl_admitted=report.observation_semantics_passed && report.clean_baseline_passed;
report.timestamp_utc=char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
write_report(fullfile(outRoot,'migration_physical_gate.json'),report);

fprintf('IEEE39 migration physical gate: obs=%d clean=%d FRL_admitted=%d\n',semPass,report.clean_baseline_passed,report.full_stack_frl_admitted);
for i=1:numel(runs)
    r=runs{i};
    fprintf('  clean %.4fs: passed=%d wm=[%.6g, %.6g] vpu=[%.6g, %.6g] finite=%d\n', ...
        r.horizon_s,r.passed,r.global_wm_min,r.global_wm_max,r.global_vpu_min,r.global_vpu_max,r.all_finite);
end
if ~report.full_stack_frl_admitted
    error('AEFC:IEEE39MigrationPhysicalGateFailed', ...
        'R2024a migrated plant is not admitted to full-stack FRL; inspect migration_physical_gate.json.');
end
end

function [ok,info]=audit_wm_tag(mdl,tag)
g=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block','BlockType','Goto','GotoTag',tag);
info=struct('tag',tag,'goto_blocks',{g},'producer_block','','producer_block_type','', ...
    'producer_port',NaN,'signal_name','','rotor_speed_pu_semantics',false);
if numel(g)~=1, ok=false; return; end
try
    ph=get_param(g{1},'PortHandles'); lh=get_param(ph.Inport(1),'Line');
    assert(lh~=-1,'Goto input is unconnected.');
    sph=get_param(lh,'SrcPortHandle');
    if isempty(sph) || sph==-1
        sbh=get_param(lh,'SrcBlockHandle');
        if isempty(sbh) || sbh==-1, error('Cannot resolve Goto source.'); end
        sb=getfullname(sbh); sport=NaN;
    else
        sb=get_param(sph,'Parent');
        try, sport=str2double(get_param(sph,'PortNumber')); catch, sport=NaN; end
    end
    bt=get_param(sb,'BlockType'); sig='';
    try, sig=get_param(lh,'Name'); catch, end
    if isempty(sig) && ~isempty(sph) && sph~=-1
        try, slh=get_param(sph,'Line'); if slh~=-1, sig=get_param(slh,'Name'); end, catch, end
    end
    semantic=contains(lower(sig),'rotor speed') && contains(lower(sig),'wm') && contains(lower(sig),'pu');
    info.producer_block=sb; info.producer_block_type=bt; info.producer_port=sport; info.signal_name=sig; info.rotor_speed_pu_semantics=semantic;
    ok=semantic && (strcmpi(bt,'BusSelector') || contains(lower(get_param(sb,'Name')),'bus'));
catch ME
    info.error=full_error(ME); ok=false;
end
end

function [ok,info]=audit_voltage_measurement(mdl,pv)
allb=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block'); hits={};
for i=1:numel(allb)
    try
        if strcmp(get_param(allb{i},'LabelV'),pv.tag), hits{end+1}=allb{i}; end %#ok<AGROW>
    catch
    end
end
info=struct('tag',pv.tag,'source_measurement_path',pv.measurement_path,'migrated_measurement_blocks',{hits}, ...
    'source_vbase_volts',pv.vbase_volts,'migrated_vbase_volts',NaN,'voltage_measurement','', ...
    'vpu','','measurement_semantics_preserved',false);
if numel(hits)~=1, ok=false; return; end
b=hits{1};
try
    vm=get_param(b,'VoltageMeasurement'); vu=get_param(b,'Vpu'); vbexpr=get_param(b,'Vbase'); vb=resolve_numeric(vbexpr,b);
    vb=vb(1);
    sameBase=isfinite(vb) && abs(vb-pv.vbase_volts)<=max(1e-9,abs(pv.vbase_volts)*1e-12);
    semantic=strcmpi(strtrim(vm),'phase-to-ground') && strcmpi(strtrim(vu),'off') && sameBase;
    info.migrated_vbase_volts=vb; info.voltage_measurement=vm; info.vpu=vu; info.measurement_semantics_preserved=semantic;
    ok=semantic;
catch ME
    info.error=full_error(ME); ok=false;
end
end

function taps=install_logging_taps(mdl)
taps=cell(1,20); k=0;
for g=1:10
    k=k+1; taps{k}=make_tap(mdl,sprintf('Wm_G%d',g),sprintf('aefc_mig_wm_g%d',g),k);
    k=k+1; taps{k}=make_tap(mdl,sprintf('V_bus_G%d',g),sprintf('aefc_mig_vabc_g%d',g),k);
end
end

function t=make_tap(mdl,tag,varName,k)
fn=sprintf('AEFC_Mig_From_%02d',k); ln=sprintf('AEFC_Mig_Log_%02d',k); fp=[mdl '/' fn]; lp=[mdl '/' ln]; y=30+30*k;
safe_delete(lp); safe_delete(fp);
add_block('simulink/Signal Routing/From',fp,'GotoTag',tag,'Position',[40 y 120 y+14]);
add_block('simulink/Sinks/To Workspace',lp,'VariableName',varName,'SaveFormat','Timeseries','Position',[180 y-2 300 y+16]);
add_line(mdl,[fn '/1'],[ln '/1'],'autorouting','on');
t=struct('from_path',fp,'log_path',lp);
end

function remove_logging_taps(taps)
for i=numel(taps):-1:1
    try, safe_delete(taps{i}.log_path); safe_delete(taps{i}.from_path); catch, end
end
end

function r=run_clean(mdl,h,vbase,bounds)
r=struct('horizon_s',h,'passed',false,'all_finite',false,'global_wm_min',NaN,'global_wm_max',NaN, ...
    'global_vpu_min',NaN,'global_vpu_max',NaN,'generators',{{}});
try
    set_param(mdl,'StopTime',sprintf('%.17g',h)); out=sim(mdl,'ReturnWorkspaceOutputs','on');
    gens=cell(1,10); allFinite=true; wmin=nan(1,10); wmax=nan(1,10); vmin=nan(1,10); vmax=nan(1,10);
    for g=1:10
        wm=data_after_t0(out.get(sprintf('aefc_mig_wm_g%d',g))); wm=double(wm(:));
        va=reshape_vabc(data_after_t0(out.get(sprintf('aefc_mig_vabc_g%d',g))));
        vpu=sqrt(mean(double(va).^2,2))/(vbase(g)/sqrt(3));
        finite=all(isfinite(wm)) && all(isfinite(vpu)); allFinite=allFinite && finite;
        wmin(g)=min(wm); wmax(g)=max(wm); vmin(g)=min(vpu); vmax(g)=max(vpu);
        sane=finite && wmin(g)>=bounds.wm_min_pu && wmax(g)<=bounds.wm_max_pu && ...
            vmin(g)>=bounds.voltage_min_pu && vmax(g)<=bounds.voltage_max_pu;
        gens{g}=struct('generator',g,'measurement_vbase_volts',vbase(g), ...
            'wm_initial',wm(1),'wm_final',wm(end),'wm_min',wmin(g),'wm_max',wmax(g), ...
            'vpu_initial',vpu(1),'vpu_final',vpu(end),'vpu_min',vmin(g),'vpu_max',vmax(g),'passed',sane);
    end
    r.generators=gens; r.all_finite=allFinite; r.global_wm_min=min(wmin); r.global_wm_max=max(wmax);
    r.global_vpu_min=min(vmin); r.global_vpu_max=max(vmax); r.passed=allFinite && all(cellfun(@(x)x.passed,gens));
catch ME
    r.error=full_error(ME);
end
end

function d=data_after_t0(x)
if isa(x,'timeseries')
    d=double(x.Data); t=double(x.Time(:)); idx=find(t>0); if ~isempty(idx), d=d(idx,:,:,:,:); end
elseif isstruct(x) && isfield(x,'signals')
    d=double(x.signals.values); if size(d,1)>1, d=d(2:end,:,:,:,:); end
else
    try, d=double(x.Data); catch, d=double(x); end
    if size(d,1)>1, d=d(2:end,:,:,:,:); end
end
assert(~isempty(d),'Logged signal has no samples after t=0.');
end

function v=reshape_vabc(v)
v=squeeze(double(v)); if isvector(v) && numel(v)==3, v=reshape(v,1,3); end
if size(v,2)~=3 && size(v,1)==3, v=v.'; end
assert(size(v,2)==3,'V_bus_G* must contain three phase columns.');
end

function v=resolve_numeric(expr,b)
try, v=slResolve(expr,b); catch, v=evalin('base',expr); end
v=double(v(:).');
end
function safe_delete(p), try, h=getSimulinkBlockHandle(p); if h>0, delete_block(p); end, catch, end, end
function restore_dir(p), try, cd(p); catch, end, end
function safe_close(mdl), try, if bdIsLoaded(mdl), close_system(mdl,0); end, catch, end, end
function s=full_error(ME), s=ME.message; try, for i=1:numel(ME.cause), s=[s ' | cause: ' full_error(ME.cause{i})]; end, catch, end, end %#ok<AGROW>
function write_report(p,r), fid=fopen(p,'w'); assert(fid>0); fwrite(fid,jsonencode(r,'PrettyPrint',true)); fclose(fid); end
