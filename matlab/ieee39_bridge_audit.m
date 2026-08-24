function ieee39_bridge_audit
% Strict R2024a bridge audit for the migrated IEEE39 plant.
% A successful Simulink return code is not sufficient. The unforced plant
% must remain finite and physically plausible before any action-response,
% persistent bridge, calibration, or full-stack experiment is admitted.

repoRoot=pwd;
outRoot=fullfile(repoRoot,'build','ieee39_r2024a');
sourceModelDir=fullfile(outRoot,'source','model');
sourceMdl=fullfile(sourceModelDir,'IEEE39bus.mdl');
modelPath=fullfile(outRoot,'migrated','IEEE39bus_R2024a.slx');
migrationPath=fullfile(outRoot,'migration_report.json');
assert(exist(modelPath,'file')==2,'Migrated IEEE39 model not found.');
assert(exist(sourceMdl,'file')==2,'Source IEEE39 MDL not found.');
assert(exist(migrationPath,'file')==2,'Migration report not found.');
mig=jsondecode(fileread(migrationPath));
assert(isfield(mig,'source_generator_assets'),'Migration report lacks frozen source-generator assets.');
addpath(sourceModelDir);
[~,mdl,~]=fileparts(modelPath); load_system(modelPath);
cleanupObj=onCleanup(@()safe_close_model(mdl)); %#ok<NASGU>

report=struct;
report.release=version('-release'); report.matlab_version=version;
report.model='build/ieee39_r2024a/migrated/IEEE39bus_R2024a.slx';
report.source_model='build/ieee39_r2024a/source/model/IEEE39bus.mdl';
report.source_git_blob=mig.source_git_blob;
report.opcomm_surrogate=mig.opcomm_surrogate;
report.native_step_s=mig.native_step_s;
report.timestamp_utc=char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
report.coupled_plant=true; report.agent_count=10;
report.audit_policy='clean baseline sanity must pass before action-response or full-stack execution';
report.sanity_bounds=struct('wm_min_pu',0.5,'wm_max_pu',1.5,'voltage_min_pu',0.2,'voltage_max_pu',2.0);
report.diagnostic_horizons_s=[0.005 0.01 0.02 0.05];

controls=discover_generator_controls(mdl); report.generator_controls=controls;
report.control_map_complete=all(arrayfun(@(x)~isempty(x.pref_path) && ~isempty(x.vref_path) && x.pref_numeric && x.vref_numeric,controls));
obs=cell(1,10); obsComplete=true;
for g=1:10
    wtag=sprintf('Wm_G%d',g); vtag=sprintf('V_bus_G%d',g);
    wg=find_tag_producers(mdl,wtag); vg=find_tag_producers(mdl,vtag);
    if isempty(wg) || isempty(vg), obsComplete=false; end
    obs{g}=struct('generator',g,'speed_tag',wtag,'speed_producers',{wg},'voltage_tag',vtag,'voltage_producers',{vg});
end
report.observation_anchors=obs; report.observation_map_complete=obsComplete;

% Consume physical facts captured from the original MDL before legacy links
% were detached. Do not infer these from migrated R2024a library metadata.
assets=mig.source_generator_assets;
report.machines=assets;
report.machine_count=sum(arrayfun(@(x)~isempty(x.machine_path),assets));
report.machine_map_complete=(numel(assets)==10 && report.machine_count==10);
vbase=[assets.vbase_volts]; vlf=[assets.vlf_pu];
report.generator_vbase_volts=vbase; report.generator_vlf_pu=vlf;
report.voltage_base_map_complete=(numel(vbase)==10 && all(isfinite(vbase)) && all(vbase>0));
report.source_opcomm_inventory=mig.source_opcomm_inventory;
report.source_native_limits=source_native_provenance(fileread(sourceMdl));

% Add passive taps once. Exact t=0 logging artifacts are excluded from the
% sanity statistics; all samples at t>0 remain part of the audit.
taps=install_logging_taps(mdl); oldStop=get_param(mdl,'StopTime');
baseRuns=cell(1,numel(report.diagnostic_horizons_s)); firstBad=[];
for i=1:numel(report.diagnostic_horizons_s)
    h=report.diagnostic_horizons_s(i); baseRuns{i}=run_clean_horizon(mdl,h,vbase,report.sanity_bounds);
    if ~baseRuns{i}.passed && isempty(firstBad), firstBad=i; end
end
report.clean_baseline=baseRuns; report.clean_baseline_passed=all(cellfun(@(x)x.passed,baseRuns));
if isempty(firstBad), report.first_unstable_horizon_s=[]; else, report.first_unstable_horizon_s=report.diagnostic_horizons_s(firstBad); end
try, set_param(mdl,'StopTime',oldStop); catch, end

smoke={};
if report.clean_baseline_passed && report.control_map_complete
    smokeGenerators=[1 4 9]; smoke=cell(1,numel(smokeGenerators));
    for i=1:numel(smokeGenerators)
        g=smokeGenerators(i); smoke{i}=action_response_smoke(mdl,controls(g),g,0.01,0.005);
    end
end
report.action_response_smoke=smoke;
report.action_response_complete=~isempty(smoke) && all(cellfun(@(x)x.passed,smoke));
remove_logging_taps(taps);
report.bridge_contract_ready = report.control_map_complete && report.observation_map_complete && ...
    report.machine_map_complete && report.voltage_base_map_complete && report.clean_baseline_passed && report.action_response_complete;

write_report(fullfile(outRoot,'bridge_audit.json'),report);
fprintf('IEEE39 strict audit: mode=%s machines=%d controls=%d obs=%d vbase=%d clean=%d action=%d ready=%d\n', ...
    report.opcomm_surrogate,report.machine_count,report.control_map_complete,report.observation_map_complete, ...
    report.voltage_base_map_complete,report.clean_baseline_passed,report.action_response_complete,report.bridge_contract_ready);
for i=1:numel(baseRuns)
    r=baseRuns{i}; fprintf('  clean %.4fs: passed=%d wm=[%.6g, %.6g] vpu=[%.6g, %.6g] finite=%d\n', ...
        r.horizon_s,r.passed,r.global_wm_min,r.global_wm_max,r.global_vpu_min,r.global_vpu_max,r.all_finite);
    if ~r.passed && isfield(r,'worst_generators'), fprintf('    worst generators: %s\n',mat2str(r.worst_generators)); end
end
if ~isempty(report.first_unstable_horizon_s), fprintf('  first unstable horizon: %.6g s\n',report.first_unstable_horizon_s); end
if ~report.bridge_contract_ready
    error('AEFC:IEEE39BridgeAuditFailed','Strict IEEE39 bridge audit failed; inspect bridge_audit.json before enabling the real bridge.');
end
end

function controls=discover_generator_controls(mdl)
allc=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block','BlockType','Constant');
template=struct('generator',0,'candidates',{{}},'pref_path','','pref_name','','pref_value','', ...
    'pref_numeric',false,'vref_path','','vref_name','','vref_value','','vref_numeric',false);
controls=repmat(template,1,10);
for g=1:10
    controls(g).generator=g; candidates={};
    for i=1:numel(allc)
        b=allc{i}; if generator_from_path(b)~=g, continue; end
        n=get_param(b,'Name');
        if contains(lower(n),'pref') || contains(lower(n),'vref') || contains(lower(n),'wref')
            v=get_param(b,'Value'); candidates{end+1}=struct('path',b,'name',n,'value',v,'numeric',isfinite(str2double(v))); %#ok<AGROW>
        end
    end
    controls(g).candidates=candidates;
    [pp,pn,pv]=choose_control(candidates,{'Pref','Pref1'},g,'pref'); [vp,vn,vv]=choose_control(candidates,{'Vref','Vref1'},g,'vref');
    controls(g).pref_path=pp; controls(g).pref_name=pn; controls(g).pref_value=pv; controls(g).pref_numeric=isfinite(str2double(pv));
    controls(g).vref_path=vp; controls(g).vref_name=vn; controls(g).vref_value=vv; controls(g).vref_numeric=isfinite(str2double(vv));
end
end
function [path,name,value]=choose_control(candidates,names,g,kind)
if g==9, if strcmp(kind,'pref'), names=[names {'wref1'}]; else, names=[names {'wref2'}]; end, end
path=''; name=''; value='';
for q=1:numel(names), for i=1:numel(candidates), c=candidates{i}; if strcmpi(c.name,names{q}) && c.numeric, path=c.path; name=c.name; value=c.value; return; end, end, end
end

function p=source_native_provenance(txt)
p=struct('nominal_frequency_hz',50,'load_shedding_breakpoints_hz',[],'load_shedding_scale',[],'dynamic_load_minimum_voltage_pu',[]);
t=regexp(txt,'BreakpointsForDimension1\s+"\[\s*48\.00[^\"]*\]"','match','once'); if ~isempty(t), nums=regexp(t,'[-+]?\d*\.?\d+','match'); p.load_shedding_breakpoints_hz=str2double(nums); end
t=regexp(txt,'Table\s+"\[\s*0\.5\s+0\.55\s+0\.65\s+0\.75\s+0\.85\s+0\.95\s+1\s*\]"','match','once'); if ~isempty(t), nums=regexp(t,'[-+]?\d*\.?\d+','match'); p.load_shedding_scale=str2double(nums); end
t=regexp(txt,'MinimumVoltage\s+"([^"]+)"','tokens','once'); if ~isempty(t), p.dynamic_load_minimum_voltage_pu=str2double(t{1}); end
end

function taps=install_logging_taps(mdl)
taps=cell(1,20); k=0;
for g=1:10
    k=k+1; taps{k}=make_tap(mdl,sprintf('Wm_G%d',g),sprintf('aefc_wm_g%d',g),k);
    k=k+1; taps{k}=make_tap(mdl,sprintf('V_bus_G%d',g),sprintf('aefc_vabc_g%d',g),k);
end
end
function t=make_tap(mdl,tag,varName,k)
fn=sprintf('AEFC_Audit_From_%02d',k); ln=sprintf('AEFC_Audit_Log_%02d',k); fp=[mdl '/' fn]; lp=[mdl '/' ln]; safe_delete(lp); safe_delete(fp); y=30+32*k;
add_block('simulink/Signal Routing/From',fp,'GotoTag',tag,'Position',[40 y 120 y+14]);
add_block('simulink/Sinks/To Workspace',lp,'VariableName',varName,'SaveFormat','Timeseries','Position',[180 y-2 295 y+16]);
add_line(mdl,[fn '/1'],[ln '/1'],'autorouting','on'); t=struct('from_path',fp,'log_path',lp);
end
function remove_logging_taps(taps), for i=numel(taps):-1:1, safe_delete(taps{i}.log_path); safe_delete(taps{i}.from_path); end, end

function r=run_clean_horizon(mdl,h,vbase,bounds)
r=struct('horizon_s',h,'passed',false,'all_finite',false,'global_wm_min',NaN,'global_wm_max',NaN,'global_vpu_min',NaN,'global_vpu_max',NaN,'generators',{{}},'worst_generators',[]);
try
    set_param(mdl,'StopTime',sprintf('%.17g',h)); out=sim(mdl,'ReturnWorkspaceOutputs','on');
    gens=cell(1,10); allFinite=true; wmins=nan(1,10); wmaxs=nan(1,10); vmins=nan(1,10); vmaxs=nan(1,10); score=zeros(1,10);
    for g=1:10
        wm=data_after_t0(out.get(sprintf('aefc_wm_g%d',g))); vabc=reshape_vabc(data_after_t0(out.get(sprintf('aefc_vabc_g%d',g)))); wm=wm(:);
        vpu=sqrt(mean(double(vabc).^2,2))/(vbase(g)/sqrt(3)); finite=all(isfinite(wm)) && all(isfinite(vpu)); allFinite=allFinite && finite;
        wmins(g)=min(wm); wmaxs(g)=max(wm); vmins(g)=min(vpu); vmaxs(g)=max(vpu);
        sane=finite && wmins(g)>=bounds.wm_min_pu && wmaxs(g)<=bounds.wm_max_pu && vmins(g)>=bounds.voltage_min_pu && vmaxs(g)<=bounds.voltage_max_pu;
        score(g)=max([abs(wmins(g)-1),abs(wmaxs(g)-1),abs(vmins(g)-1),abs(vmaxs(g)-1)]);
        gens{g}=struct('generator',g,'wm_initial',wm(1),'wm_final',wm(end),'wm_min',wmins(g),'wm_max',wmaxs(g), ...
            'vpu_initial',vpu(1),'vpu_final',vpu(end),'vpu_min',vmins(g),'vpu_max',vmaxs(g),'passed',sane);
    end
    [~,ord]=sort(score,'descend'); r.worst_generators=ord(1:min(3,numel(ord)));
    r.generators=gens; r.all_finite=allFinite; r.global_wm_min=min(wmins); r.global_wm_max=max(wmaxs); r.global_vpu_min=min(vmins); r.global_vpu_max=max(vmaxs);
    r.passed=allFinite && all(cellfun(@(x)x.passed,gens));
catch ME, r.error=full_error(ME); end
end

function r=action_response_smoke(mdl,c,g,deltaPref,h)
r=struct('generator',g,'control_block',c.pref_path,'baseline_pref',c.pref_value,'delta_pref_pu',deltaPref,'horizon_s',h,'baseline_final_wm',NaN,'perturbed_final_wm',NaN,'absolute_delta_wm',NaN,'passed',false);
old=get_param(c.pref_path,'Value'); oldStop=get_param(mdl,'StopTime');
try
    set_param(mdl,'StopTime',sprintf('%.17g',h)); o1=sim(mdl,'ReturnWorkspaceOutputs','on'); b=data_after_t0(o1.get(sprintf('aefc_wm_g%d',g))); b=b(end);
    set_param(c.pref_path,'Value',sprintf('%.17g',str2double(old)+deltaPref)); o2=sim(mdl,'ReturnWorkspaceOutputs','on'); p=data_after_t0(o2.get(sprintf('aefc_wm_g%d',g))); p=p(end);
    d=abs(p-b); r.baseline_final_wm=b; r.perturbed_final_wm=p; r.absolute_delta_wm=d; r.passed=isfinite(d) && d>1e-10 && d<0.25;
catch ME, r.error=full_error(ME); end
try, set_param(c.pref_path,'Value',old); catch, end; try, set_param(mdl,'StopTime',oldStop); catch, end
end

function d=data_after_t0(x)
if isa(x,'timeseries')
    d=double(x.Data); t=double(x.Time(:)); idx=find(t>0); if ~isempty(idx), d=d(idx,:,:,:,:); end
elseif isstruct(x) && isfield(x,'signals'), d=double(x.signals.values); if size(d,1)>1, d=d(2:end,:,:,:,:); end
else, try, d=double(x.Data); catch, d=double(x); end; if size(d,1)>1, d=d(2:end,:,:,:,:); end, end
assert(~isempty(d),'Logged signal has no samples after t=0.');
end
function v=reshape_vabc(v), v=squeeze(double(v)); if isvector(v) && numel(v)==3, v=reshape(v,1,3); end; if size(v,2)~=3 && size(v,1)==3, v=v.'; end; assert(size(v,2)==3,'Vabc log must have three phase columns.'); end
function g=generator_from_path(p), t=regexp(p,'/GT\s*(\d+)','tokens','once'); if isempty(t), g=NaN; else, g=str2double(t{1}); end, end
function p=find_tag_producers(mdl,tag), b=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block','BlockType','Goto','GotoTag',tag); p=sort(b(:)'); end
function safe_delete(p), h=getSimulinkBlockHandle(p); if h>0, try, ph=get_param(p,'PortHandles'); if isfield(ph,'Inport'), for q=ph.Inport(:)', ln=get_param(q,'Line'); if ln~=-1, delete_line(ln); end, end, end; catch, end; delete_block(p); end, end
function safe_close_model(mdl), try, if bdIsLoaded(mdl), close_system(mdl,0); end, catch, end, end
function s=full_error(ME), s=ME.message; try, for i=1:numel(ME.cause), s=[s ' | cause: ' full_error(ME.cause{i})]; end, catch, end, end %#ok<AGROW>
function write_report(p,r), fid=fopen(p,'w'); assert(fid>0); fwrite(fid,jsonencode(r,'PrettyPrint',true)); fclose(fid); end
