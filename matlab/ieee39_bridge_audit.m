function ieee39_bridge_audit
% Audit the migrated IEEE39 model for a reproducible RL bridge.
% The audit discovers the real heterogeneous generator reference inputs
% rather than assuming that every machine uses an identically named block.

repoRoot=pwd;
outRoot=fullfile(repoRoot,'build','ieee39_r2024a');
sourceModelDir=fullfile(outRoot,'source','model');
modelPath=fullfile(outRoot,'migrated','IEEE39bus_R2024a.slx');
assert(exist(modelPath,'file')==2,'Migrated IEEE39 model not found. Run ieee39_migrate_r2024a first.');
assert(exist(sourceModelDir,'dir')==7,'Migrated-model source directory is missing.');
addpath(sourceModelDir);
[~,mdl,~]=fileparts(modelPath);
load_system(modelPath);

report=struct;
report.release=version('-release');
report.matlab_version=version;
report.model='build/ieee39_r2024a/migrated/IEEE39bus_R2024a.slx';
report.source_model_helper_path='build/ieee39_r2024a/source/model';
report.timestamp_utc=char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
report.coupled_plant=true;
report.agent_count=10;
report.agent_partition='generator-side physical controllers acting synchronously on one IEEE39 plant instance';
report.bridge_design=['Ten generator-side agents share one coupled plant. The audit discovers each ' ...
    'generator''s actual governor/power and excitation/voltage reference constants; names are not assumed homogeneous.'];

controls=discover_generator_controls(mdl);
report.generator_controls=controls;
report.control_map_complete=all(arrayfun(@(x)~isempty(x.pref_path) && ~isempty(x.vref_path) && ...
    x.pref_numeric && x.vref_numeric,controls));
report.uniform_control_names=all(strcmp({controls.pref_name},'Pref')) && all(strcmp({controls.vref_name},'Vref'));

obs=cell(1,10); obsComplete=true;
for g=1:10
    wtag=sprintf('Wm_G%d',g); vtag=sprintf('V_bus_G%d',g);
    wg=find_tag_producers(mdl,wtag); vg=find_tag_producers(mdl,vtag);
    if isempty(wg) || isempty(vg), obsComplete=false; end
    obs{g}=struct('generator',g,'speed_tag',wtag,'speed_producers',{wg}, ...
        'voltage_tag',vtag,'voltage_producers',{vg});
end
report.observation_anchors=obs;
report.observation_map_complete=obsComplete;

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

% Create one reusable measurement tap. Repeatedly adding/deleting separate
% From->To Workspace branches can leave compiled port connection state in SPS
% models. Reusing one connected pair avoids topology churn between smoke runs.
fromName='AEFC_Audit_Wm_From'; logName='AEFC_Audit_Wm_Log';
fromPath=[mdl '/' fromName]; logPath=[mdl '/' logName];
safe_delete(logPath); safe_delete(fromPath);
add_block('simulink/Signal Routing/From',fromPath,'GotoTag','Wm_G1','Position',[80 80 175 100]);
add_block('simulink/Sinks/To Workspace',logPath,'VariableName','audit_wm','SaveFormat','Timeseries','Position',[230 78 340 102]);
phFrom=get_param(fromPath,'PortHandles'); phLog=get_param(logPath,'PortHandles');
ln=get_param(phLog.Inport(1),'Line'); if ln~=-1, delete_line(ln); end
add_line(mdl,phFrom.Outport(1),phLog.Inport(1),'autorouting','on');

% Cover both naming outliers (G1/G9) and one conventional mapping (G4).
smokeGenerators=[1 4 9]; smoke=cell(1,numel(smokeGenerators));
for i=1:numel(smokeGenerators)
    g=smokeGenerators(i);
    smoke{i}=action_response_smoke(mdl,controls(g),g,0.05,0.05,fromPath,logPath);
end
safe_delete(logPath); safe_delete(fromPath);
report.action_response_smoke=smoke;
report.action_response_complete=all(cellfun(@(x)x.passed,smoke));

report.bridge_contract_ready = report.control_map_complete && report.observation_map_complete && ...
    report.machine_count>=10 && report.action_response_complete;

jsonPath=fullfile(outRoot,'bridge_audit.json'); write_report(jsonPath,report);
fprintf('IEEE39 bridge audit: machines=%d control_map=%d obs_map=%d uniform_names=%d action_response=%d ready=%d\n', ...
    report.machine_count,report.control_map_complete,report.observation_map_complete, ...
    report.uniform_control_names,report.action_response_complete,report.bridge_contract_ready);
for g=1:10
    fprintf('  G%d Pref-like=%s (%s) Vref-like=%s (%s)\n',g,controls(g).pref_name, ...
        controls(g).pref_value,controls(g).vref_name,controls(g).vref_value);
end
for i=1:numel(smoke)
    fprintf('  smoke G%d deltaWm=%.12g passed=%d\n',smoke{i}.generator,smoke{i}.absolute_delta_wm,smoke{i}.passed);
end
close_system(mdl,0);
if ~report.bridge_contract_ready
    error('AEFC:IEEE39BridgeAuditFailed','Real IEEE39 control/observation bridge audit failed; inspect bridge_audit.json.');
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
            v=get_param(b,'Value');
            candidates{end+1}=struct('path',b,'name',n,'value',v,'numeric',isfinite(str2double(v))); %#ok<AGROW>
        end
    end
    controls(g).candidates=candidates;
    [pp,pn,pv]=choose_control(candidates,{'Pref','Pref1'},g,'pref');
    [vp,vn,vv]=choose_control(candidates,{'Vref','Vref1'},g,'vref');
    controls(g).pref_path=pp; controls(g).pref_name=pn; controls(g).pref_value=pv; controls(g).pref_numeric=isfinite(str2double(pv));
    controls(g).vref_path=vp; controls(g).vref_name=vn; controls(g).vref_value=vv; controls(g).vref_numeric=isfinite(str2double(vv));
end
end

function [path,name,value]=choose_control(candidates,names,g,kind)
% G9 uses wref1 as the hydraulic-governor power reference and wref2 as the
% excitation voltage reference in the source MDL.
if g==9
    if strcmp(kind,'pref'), names=[names {'wref1'}]; else, names=[names {'wref2'}]; end
end
path=''; name=''; value='';
for q=1:numel(names)
    for i=1:numel(candidates)
        c=candidates{i};
        if strcmpi(c.name,names{q}) && c.numeric
            path=c.path; name=c.name; value=c.value; return;
        end
    end
end
end

function r=action_response_smoke(mdl,control,g,deltaPref,stopTime,fromPath,logPath)
r=struct('generator',g,'control_block',control.pref_path,'control_name',control.pref_name, ...
    'baseline_pref',control.pref_value,'delta_pref_pu',deltaPref,'stop_time_s',stopTime, ...
    'baseline_final_wm',NaN,'perturbed_final_wm',NaN,'absolute_delta_wm',NaN,'passed',false);
if isempty(control.pref_path) || ~control.pref_numeric
    r.error='No numeric Pref-like control was discovered.'; return;
end
oldPref=get_param(control.pref_path,'Value'); oldStop=get_param(mdl,'StopTime');
try
    set_param(fromPath,'GotoTag',sprintf('Wm_G%d',g));
    set_param(logPath,'VariableName','audit_wm');
    set_param(mdl,'StopTime',sprintf('%.17g',stopTime));
    baseOut=sim(mdl,'ReturnWorkspaceOutputs','on'); bfinal=last_numeric_value(baseOut.get('audit_wm'));
    p0=str2double(oldPref); set_param(control.pref_path,'Value',sprintf('%.17g',p0+deltaPref));
    pertOut=sim(mdl,'ReturnWorkspaceOutputs','on'); pfinal=last_numeric_value(pertOut.get('audit_wm'));
    d=abs(pfinal-bfinal);
    r.baseline_final_wm=bfinal; r.perturbed_final_wm=pfinal; r.absolute_delta_wm=d;
    r.passed=isfinite(d) && d>1e-10;
catch ME
    r.error=full_error(ME);
end
try, set_param(control.pref_path,'Value',oldPref); catch, end
try, set_param(mdl,'StopTime',oldStop); catch, end
end

function g=generator_from_path(p)
t=regexp(p,'/GT\s*(\d+)','tokens','once'); if isempty(t), g=NaN; else, g=str2double(t{1}); end
end
function p=find_tag_producers(mdl,tag)
b=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block','BlockType','Goto','GotoTag',tag); p=sort(b(:)');
end
function s=describe_machine(b)
s=struct('path',b); props={'SourceType','NominalParameters','MechanicalLoad','RotorType','Mechanical','InitialConditions','LoadFlowParameters','Pref','Qref','IterativeModel'};
for i=1:numel(props), try, s.(props{i})=get_param(b,props{i}); catch, end, end
end
function s=describe_block(b,sourceType), s=struct('path',b,'source_type',sourceType); end
function v=last_numeric_value(x)
if isa(x,'timeseries'), d=x.Data; elseif isstruct(x) && isfield(x,'signals'), d=x.signals.values; else, try, d=x.Data; catch, d=x; end, end
d=double(d); assert(~isempty(d),'Logged signal is empty.'); v=d(end); assert(isfinite(v),'Logged final value is not finite.');
end
function safe_delete(p), if getSimulinkBlockHandle(p)>0, delete_block(p); end, end
function s=full_error(ME), s=ME.message; try, for i=1:numel(ME.cause), s=[s ' | cause: ' full_error(ME.cause{i})]; end, catch, end, end %#ok<AGROW>
function write_report(p,r), fid=fopen(p,'w'); assert(fid>0); fwrite(fid,jsonencode(r,'PrettyPrint',true)); fclose(fid); end
