function ieee39_dpl_dynamic_state_audit
% Read-only dynamic-state audit for the R2024a ARTEMIS->SPS DPL migration.
% No model parameter is changed. The audit separates three questions:
%   (1) what initial SPS electrical states R2024a actually compiles,
%   (2) whether the assembled SPS state-space is finite/well conditioned,
%   (3) whether the first physical violation occurs before or after the
%       minimum migrated-line propagation delay (~1.9 native steps).

repoRoot=pwd;
outRoot=fullfile(repoRoot,'build','ieee39_r2024a');
sourceDir=fullfile(outRoot,'source','model');
modelPath=fullfile(outRoot,'migrated','IEEE39bus_R2024a.slx');
provPath=fullfile(repoRoot,'build','ieee39_source_provenance.json');
dplPath=fullfile(outRoot,'dpl_equivalence_audit.json');
assert(exist(modelPath,'file')==2 && exist(provPath,'file')==2 && exist(dplPath,'file')==2, ...
    'Required migration artifacts are missing.');
prov=jsondecode(fileread(provPath)); dpl=jsondecode(fileread(dplPath));
assert(isfield(dpl,'source_line_count') && isfield(dpl,'migrated_line_count') && ...
    dpl.source_line_count==34 && dpl.migrated_line_count==34 && ...
    dpl.parameter_equivalence_passed && dpl.propagation_plausibility_passed && ...
    dpl.topology_connectivity_passed && isfield(dpl,'lines') && numel(dpl.lines)==34, ...
    'DPL parameter/propagation/topology contract must pass before dynamic-state audit.');

addpath(sourceDir); oldDir=pwd; dirCleanup=onCleanup(@()restore_dir(oldDir)); %#ok<NASGU>
cd(sourceDir); [~,mdl,~]=fileparts(modelPath); load_system(modelPath);
modelCleanup=onCleanup(@()safe_close(mdl)); %#ok<NASGU>

delays=arrayfun(@(x)double(x.positive_sequence_delay_samples),dpl.lines);
assert(all(isfinite(delays)) && all(delays>0),'DPL propagation delays are invalid.');

report=struct;
report.release=version('-release');
report.read_only=true;
report.model_parameter_changes=0;
report.source_git_blob=prov.source_git_blob_expected;
report.native_step_s=25e-6;
report.minimum_positive_sequence_delay_steps=min(delays);
report.maximum_positive_sequence_delay_steps=max(delays);
report.dpl_parameter_equivalence_passed=dpl.parameter_equivalence_passed;
report.dpl_propagation_plausibility_passed=dpl.propagation_plausibility_passed;
report.dpl_topology_connectivity_passed=dpl.topology_connectivity_passed;
report.dpl_algorithm_equivalence_previously_demonstrated=dpl.algorithm_equivalence_demonstrated;
report.dpl_initialization_equivalence_previously_demonstrated=dpl.initialization_equivalence_demonstrated;

% Record native SPS DPL mask-level initialization fields. These are provenance
% only; they are NOT interpreted as the compiled electrical state.
lineInit=cell(1,numel(dpl.lines));
for i=1:numel(dpl.lines)
    b=dpl.lines(i).migrated_block;
    vals=struct;
    fields={'x1','x2','x3','x4','x5','V1','V2','I1','I2','nHarmo', ...
        'Decoupling','VsMag0','VrMag0','VsAngle0','VrAngle0','IsMag0','IrMag0','IsAngle0','IrAngle0'};
    for j=1:numel(fields)
        f=fields{j}; try, vals.(f)=get_param(b,f); catch, vals.(f)=''; end
    end
    lineInit{i}=struct('block',b,'positive_sequence_delay_samples',delays(i),'fields',vals);
end
report.native_dpl_mask_initialization=lineInit;

% Compile-time initial block states from Simulink itself. Failure of this API
% is recorded, not confused with a plant-fidelity failure.
try
    x0=Simulink.BlockDiagram.getInitialState(mdl);
    [initSummary,initRecords]=summarize_initial_states(x0);
    report.simulink_initial_state=initSummary;
    report.simulink_initial_state_records=initRecords;
catch ME
    report.simulink_initial_state=struct('available',false,'error',full_error(ME));
    report.simulink_initial_state_records={};
end

% Specialized Power Systems state-space diagnostic. API/release differences
% are recorded rather than allowed to abort the decisive micro-trajectory.
try
    sps=power_analyze(mdl,'structure');
    pa=struct('available',true);
    if isfield(sps,'A')
        pa.state_count=size(sps.A,1); pa.A_all_finite=all(isfinite(sps.A(:))); pa.rcond_A=safe_rcond(sps.A);
    end
    if isfield(sps,'B'), pa.input_count=size(sps.B,2); end
    if isfield(sps,'C'), pa.output_count=size(sps.C,1); end
    if isfield(sps,'x0') && ~isempty(sps.x0)
        pa.x0_all_finite=all(isfinite(sps.x0(:))); pa.x0_max_abs=maxabs(sps.x0);
    end
    if isfield(sps,'frequencies'), pa.frequencies=double(sps.frequencies(:).'); end
    if isfield(sps,'xss') && ~isempty(sps.xss)
        pa.xss_all_finite=all(isfinite(sps.xss(:))); pa.xss_max_abs=maxabs(sps.xss);
        if isfield(sps,'frequencies') && ~isempty(sps.frequencies) && isfield(sps,'x0') && ~isempty(sps.x0)
            [~,ix]=min(abs(double(sps.frequencies(:))-50));
            xss50=double(sps.xss(:,ix)); x0v=double(sps.x0(:));
            if numel(xss50)==numel(x0v)
                pa.nearest_50hz_frequency=double(sps.frequencies(ix));
                pa.x0_vs_xss50_l2=norm(x0v-xss50);
                pa.x0_vs_xss50_relative=norm(x0v-xss50)/max(1,norm(xss50));
            end
        end
    end
    report.power_analyze=pa;
catch ME
    report.power_analyze=struct('available',false,'error',full_error(ME));
end

% One clean 5-ms trajectory; no FRL action and no model parameter change.
measBases=arrayfun(@(x)x.vbase_volts,prov.voltage_observations);
taps=install_logging_taps(mdl); tapCleanup=onCleanup(@()remove_logging_taps(taps)); %#ok<NASGU>
trajectory=run_micro_trajectory(mdl,0.005,measBases,report.native_step_s);
report.micro_trajectory=trajectory;
report.first_physical_violation_s=trajectory.first_violation_s;
report.first_positive_time_violation_s=trajectory.first_positive_time_violation_s;
if isempty(trajectory.first_positive_time_violation_s)
    report.first_positive_time_violation_delay_steps=[];
    report.violation_before_minimum_dpl_propagation=false;
else
    report.first_positive_time_violation_delay_steps=trajectory.first_positive_time_violation_s/report.native_step_s;
    report.violation_before_minimum_dpl_propagation= ...
        report.first_positive_time_violation_delay_steps < report.minimum_positive_sequence_delay_steps;
end
report.dynamic_equivalence_demonstrated=false;
report.full_stack_frl_admitted=false;
report.timestamp_utc=char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
write_report(fullfile(outRoot,'dpl_dynamic_state_audit.json'),report);

fprintf('IEEE39 DPL dynamic-state audit: minDelay=%.6g Ts firstPositiveViolation=',report.minimum_positive_sequence_delay_steps);
if isempty(report.first_positive_time_violation_delay_steps), fprintf('none'); else, fprintf('%.6g Ts',report.first_positive_time_violation_delay_steps); end
fprintf(' beforeMinDelay=%d\n',logical(report.violation_before_minimum_dpl_propagation));
if isfield(report.power_analyze,'available') && report.power_analyze.available
    if isfield(report.power_analyze,'state_count'), fprintf('  SPS states=%d',report.power_analyze.state_count); end
    if isfield(report.power_analyze,'rcond_A'), fprintf(' rcond(A)=%.6g',report.power_analyze.rcond_A); end
    if isfield(report.power_analyze,'x0_max_abs'), fprintf(' x0max=%.6g',report.power_analyze.x0_max_abs); end
    fprintf('\n');
else
    fprintf('  power_analyze unavailable in this diagnostic; micro-trajectory retained as decisive evidence.\n');
end
end

function [s,records]=summarize_initial_states(x0)
s=struct('available',true,'count',0,'all_finite',true,'max_abs',0); records={};
if isstruct(x0) && isfield(x0,'signals')
    sig=x0.signals; s.count=numel(sig); records=cell(1,numel(sig));
    for i=1:numel(sig)
        v=double(sig(i).values); fin=all(isfinite(v(:))); m=maxabs(v);
        s.all_finite=s.all_finite && fin; s.max_abs=max(s.max_abs,m);
        block=''; label=''; sampleTime=[]; dimensions=[];
        try, block=sig(i).blockName; catch, end
        try, label=sig(i).label; catch, end
        try, sampleTime=sig(i).sampleTime; catch, end
        try, dimensions=sig(i).dimensions; catch, end
        records{i}=struct('block',block,'label',label,'sample_time',sampleTime, ...
            'finite',fin,'max_abs',m,'dimensions',dimensions);
    end
else
    s.available=false; s.class=class(x0);
end
end

function tr=run_micro_trajectory(mdl,stopTime,vbase,Ts)
tr=struct('stop_time_s',stopTime,'bounds',struct('wm_min',0.5,'wm_max',1.5,'vpu_min',0.2,'vpu_max',2.0));
set_param(mdl,'StopTime',sprintf('%.17g',stopTime)); out=sim(mdl,'ReturnWorkspaceOutputs','on');
wm=cell(1,10); vv=cell(1,10);
for g=1:10
    [tw,dw]=timeseries_data(out.get(sprintf('aefc_dyn_wm_g%d',g)));
    [tv,dv]=timeseries_data(out.get(sprintf('aefc_dyn_vabc_g%d',g)));
    dv=reshape_vabc(dv); vpu=sqrt(mean(double(dv).^2,2))/(vbase(g)/sqrt(3));
    wm{g}=struct('t',tw,'v',double(dw(:))); vv{g}=struct('t',tv,'v',double(vpu(:)));
end
checkSteps=[0 1 2 4 10 20 40 80 200]; points=cell(1,numel(checkSteps));
for k=1:numel(checkSteps)
    target=checkSteps(k)*Ts; wvals=nan(1,10); vvals=nan(1,10);
    for g=1:10
        wvals(g)=sample_at(wm{g}.t,wm{g}.v,target);
        vvals(g)=sample_at(vv{g}.t,vv{g}.v,target);
    end
    finite=all(isfinite(wvals)) && all(isfinite(vvals));
    sane=finite && all(wvals>=0.5 & wvals<=1.5) && all(vvals>=0.2 & vvals<=2.0);
    points{k}=struct('target_s',target,'delay_steps',checkSteps(k),'wm_min',min(wvals),'wm_max',max(wvals), ...
        'vpu_min',min(vvals),'vpu_max',max(vvals),'all_finite',finite,'passed',sane, ...
        'wm_by_generator',wvals,'vpu_by_generator',vvals);
end
% Search the union of actual logging times, not interpolated times.
allTimes=[];
for g=1:10, allTimes=[allTimes; wm{g}.t(:); vv{g}.t(:)]; end %#ok<AGROW>
allTimes=unique(double(allTimes)); allTimes=allTimes(allTimes>=0 & allTimes<=stopTime);
first=[]; firstPositive=[];
for q=1:numel(allTimes)
    t=allTimes(q); sane=true;
    for g=1:10
        a=sample_at(wm{g}.t,wm{g}.v,t); b=sample_at(vv{g}.t,vv{g}.v,t);
        if ~isfinite(a) || ~isfinite(b) || a<0.5 || a>1.5 || b<0.2 || b>2.0, sane=false; break; end
    end
    if ~sane && isempty(first), first=t; end
    if ~sane && t>0, firstPositive=t; break; end
end
tr.checkpoints=points; tr.first_violation_s=first; tr.first_positive_time_violation_s=firstPositive; tr.logged_time_count=numel(allTimes);
end

function x=sample_at(t,v,target)
[~,i]=min(abs(double(t(:))-target)); x=double(v(i));
end
function [t,d]=timeseries_data(x)
if isa(x,'timeseries'), t=double(x.Time(:)); d=double(x.Data); else, error('Expected timeseries logging.'); end
end
function v=reshape_vabc(v)
v=squeeze(double(v)); if isvector(v)&&numel(v)==3, v=reshape(v,1,3); end
if size(v,2)~=3&&size(v,1)==3, v=v.'; end
assert(size(v,2)==3,'V_bus_G* must contain three phase columns.');
end

function taps=install_logging_taps(mdl)
taps=cell(1,20); k=0;
for g=1:10
    k=k+1; taps{k}=make_tap(mdl,sprintf('Wm_G%d',g),sprintf('aefc_dyn_wm_g%d',g),k);
    k=k+1; taps{k}=make_tap(mdl,sprintf('V_bus_G%d',g),sprintf('aefc_dyn_vabc_g%d',g),k);
end
end
function t=make_tap(mdl,tag,varName,k)
fn=sprintf('AEFC_Dyn_From_%02d',k); ln=sprintf('AEFC_Dyn_Log_%02d',k); fp=[mdl '/' fn]; lp=[mdl '/' ln]; y=30+30*k;
safe_delete(lp); safe_delete(fp); add_block('simulink/Signal Routing/From',fp,'GotoTag',tag,'Position',[40 y 120 y+14]);
add_block('simulink/Sinks/To Workspace',lp,'VariableName',varName,'SaveFormat','Timeseries','Position',[180 y-2 300 y+16]);
add_line(mdl,[fn '/1'],[ln '/1'],'autorouting','on'); t=struct('from_path',fp,'log_path',lp);
end
function remove_logging_taps(taps), for i=numel(taps):-1:1, try, safe_delete(taps{i}.log_path); safe_delete(taps{i}.from_path); catch, end, end, end
function m=maxabs(x), if isempty(x), m=0; else, m=max(abs(double(x(:)))); if isempty(m), m=0; end, end, end
function r=safe_rcond(A), try, if isempty(A), r=NaN; else, r=rcond(full(A)); end, catch, r=NaN; end, end
function safe_delete(p), try, h=getSimulinkBlockHandle(p); if h>0, delete_block(p); end, catch, end, end
function restore_dir(p), try, cd(p); catch, end, end
function safe_close(mdl), try, if bdIsLoaded(mdl), close_system(mdl,0); end, catch, end, end
function s=full_error(ME), s=ME.message; try, for i=1:numel(ME.cause), s=[s ' | cause: ' full_error(ME.cause{i})]; end, catch, end, end %#ok<AGROW>
function write_report(p,r), fid=fopen(p,'w'); assert(fid>0); fwrite(fid,jsonencode(r,'PrettyPrint',true)); fclose(fid); end
