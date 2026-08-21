function ieee39_bridge_audit
% Audit the migrated IEEE39 model for a reproducible RL bridge.
% This script does not train a policy. It establishes real observation and
% actuation anchors and performs an action-response smoke test on R2024a.

repoRoot=pwd;
outRoot=fullfile(repoRoot,'build','ieee39_r2024a');
modelPath=fullfile(outRoot,'migrated','IEEE39bus_R2024a.slx');
assert(exist(modelPath,'file')==2,'Migrated IEEE39 model not found. Run ieee39_migrate_r2024a first.');
[~,mdl,~]=fileparts(modelPath);
load_system(modelPath);

report=struct;
report.release=version('-release');
report.matlab_version=version;
report.model='build/ieee39_r2024a/migrated/IEEE39bus_R2024a.slx';
report.bridge_design='10 generator-side agents; local actions are delta Pref and delta Vref; observations are local/neighbor Wm and generator-bus voltage signals.';
report.timestamp_utc=char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));

% Exact physical control anchors already present in the source model.
prefs=sort(find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block','BlockType','Constant','Name','Pref'));
vrefs=sort(find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block','BlockType','Constant','Name','Vref'));
report.pref_count=numel(prefs); report.vref_count=numel(vrefs);
assert(numel(prefs)==10,'Expected exactly 10 generator Pref Constant blocks; found %d.',numel(prefs));
assert(numel(vrefs)==10,'Expected exactly 10 generator Vref Constant blocks; found %d.',numel(vrefs));
report.pref_controls=describe_constant_controls(prefs);
report.vref_controls=describe_constant_controls(vrefs);

% Exact global observation anchors. Each generator must expose one speed
% (per-unit mechanical speed Wm) and one three-phase generator-bus voltage.
obs=cell(1,10);
for g=1:10
    wtag=sprintf('Wm_G%d',g); vtag=sprintf('V_bus_G%d',g);
    wg=find_tag_producers(mdl,wtag); vg=find_tag_producers(mdl,vtag);
    assert(~isempty(wg),'Missing global speed producer %s.',wtag);
    assert(~isempty(vg),'Missing global voltage producer %s.',vtag);
    obs{g}=struct('generator',g,'speed_tag',wtag,'speed_producers',{wg}, ...
        'voltage_tag',vtag,'voltage_producers',{vg});
end
report.observation_anchors=obs;

% Inventory real machine/load/breaker assets for later safety and recovery
% definitions. Do not infer safety limits here; this is only provenance.
blocks=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block');
machines={}; loads={}; breakers={};
for k=1:numel(blocks)
    b=blocks{k};
    try, st=get_param(b,'SourceType'); catch, st=''; end
    if contains(lower(st),'synchronous machine')
        machines{end+1}=describe_machine(b); %#ok<AGROW>
    elseif contains(lower(st),'dynamic load')
        loads{end+1}=describe_block(b,st); %#ok<AGROW>
    elseif contains(lower(st),'breaker')
        breakers{end+1}=describe_block(b,st); %#ok<AGROW>
    end
end
report.machine_count=numel(machines); report.machines=machines;
report.dynamic_load_count=numel(loads); report.dynamic_loads=loads;
report.breaker_count=numel(breakers); report.breakers=breakers;
assert(report.machine_count>=10,'Expected at least 10 synchronous-machine assets.');

% The bridge uses one coupled plant, not one independent environment/client.
report.coupled_plant=true;
report.agent_count=10;
report.agent_partition='generator-side physical controllers acting synchronously on one IEEE39 plant instance';

% Add temporary logging taps through the model's existing global tags. These
% are deleted before exit and do not alter the stored migrated SLX artifact.
fromPath=[mdl '/AEFC_Audit_Wm_From']; logPath=[mdl '/AEFC_Audit_Wm_Log'];
safe_delete(logPath); safe_delete(fromPath);
add_block('simulink/Signal Routing/From',fromPath,'GotoTag','Wm_G1','Position',[80 80 160 100]);
add_block('simulink/Sinks/To Workspace',logPath,'VariableName','audit_wm_g1', ...
    'SaveFormat','Timeseries','Position',[230 78 330 102]);
add_line(mdl,'AEFC_Audit_Wm_From/1','AEFC_Audit_Wm_Log/1','autorouting','on');

% Action-response smoke test. We change only the existing G1 Pref input and
% require a finite, nonzero difference in the observed Wm_G1 trajectory.
g1pref=find_generator_control(prefs,1);
oldPref=get_param(g1pref,'Value'); oldStop=get_param(mdl,'StopTime');
report.action_response_smoke=struct('control_block',g1pref,'baseline_pref',oldPref, ...
    'delta_pref_pu',0.05,'stop_time_s',0.02,'baseline_final_wm',NaN, ...
    'perturbed_final_wm',NaN,'absolute_delta_wm',NaN,'passed',false);
try
    set_param(mdl,'StopTime','0.02');
    baseOut=sim(mdl,'ReturnWorkspaceOutputs','on');
    bts=baseOut.get('audit_wm_g1'); bfinal=last_numeric_value(bts);
    p0=str2double(oldPref); assert(isfinite(p0),'G1 Pref must be a numeric constant for the bridge smoke test.');
    set_param(g1pref,'Value',sprintf('%.17g',p0+0.05));
    pertOut=sim(mdl,'ReturnWorkspaceOutputs','on');
    pts=pertOut.get('audit_wm_g1'); pfinal=last_numeric_value(pts);
    d=abs(pfinal-bfinal);
    report.action_response_smoke.baseline_final_wm=bfinal;
    report.action_response_smoke.perturbed_final_wm=pfinal;
    report.action_response_smoke.absolute_delta_wm=d;
    report.action_response_smoke.passed=isfinite(d) && d>1e-10;
catch ME
    report.action_response_smoke.error=full_error(ME);
end
set_param(g1pref,'Value',oldPref); set_param(mdl,'StopTime',oldStop);
safe_delete(logPath); safe_delete(fromPath);

% Hard bridge contract: all control/measurement anchors exist and a genuine
% plant response to a physical control input has been observed.
report.bridge_contract_ready = report.pref_count==10 && report.vref_count==10 && ...
    numel(report.observation_anchors)==10 && report.action_response_smoke.passed;

jsonPath=fullfile(outRoot,'bridge_audit.json'); write_report(jsonPath,report);
close_system(mdl,0);
fprintf('IEEE39 bridge audit: machines=%d Pref=%d Vref=%d response_delta=%.12g ready=%d\n', ...
    report.machine_count,report.pref_count,report.vref_count, ...
    report.action_response_smoke.absolute_delta_wm,report.bridge_contract_ready);
if ~report.bridge_contract_ready
    error('AEFC:IEEE39BridgeAuditFailed','Real IEEE39 control/observation bridge audit failed; inspect bridge_audit.json.');
end
end

function x=describe_constant_controls(blocks)
x=cell(1,numel(blocks));
for i=1:numel(blocks)
    b=blocks{i}; x{i}=struct('path',b,'value',get_param(b,'Value'),'generator',generator_from_path(b));
end
end

function g=generator_from_path(p)
t=regexp(p,'/GT\s*(\d+)','tokens','once');
if isempty(t), g=NaN; else, g=str2double(t{1}); end
end

function b=find_generator_control(blocks,g)
idx=[];
for i=1:numel(blocks), if generator_from_path(blocks{i})==g, idx(end+1)=i; end, end %#ok<AGROW>
assert(numel(idx)==1,'Expected exactly one control block for generator %d.',g); b=blocks{idx};
end

function p=find_tag_producers(mdl,tag)
b=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block','BlockType','Goto','GotoTag',tag);
p=sort(b(:)');
end

function s=describe_machine(b)
s=struct('path',b);
props={'SourceType','NominalParameters','MechanicalLoad','RotorType','Mechanical','InitialConditions','LoadFlowParameters','Pref','Qref','IterativeModel'};
for i=1:numel(props), try, s.(props{i})=get_param(b,props{i}); catch, end, end
end

function s=describe_block(b,sourceType)
s=struct('path',b,'source_type',sourceType);
end

function v=last_numeric_value(x)
if isa(x,'timeseries')
    d=x.Data;
elseif isstruct(x) && isfield(x,'signals')
    d=x.signals.values;
else
    try, d=x.Data; catch, d=x; end
end
d=double(d); assert(~isempty(d),'Logged signal is empty.'); v=d(end);
assert(isfinite(v),'Logged final value is not finite.');
end

function safe_delete(p), if getSimulinkBlockHandle(p)>0, delete_block(p); end, end
function s=full_error(ME), s=ME.message; try, for i=1:numel(ME.cause), s=[s ' | cause: ' full_error(ME.cause{i})]; end, catch, end, end %#ok<AGROW>
function write_report(p,r), fid=fopen(p,'w'); assert(fid>0); fwrite(fid,jsonencode(r,'PrettyPrint',true)); fclose(fid); end
