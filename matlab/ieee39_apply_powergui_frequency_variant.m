function ieee39_apply_powergui_frequency_variant
% Exact single-variable A/B for R2024a migrated IEEE39 initialization.
% Change only powergui.frequency from the released 60-Hz value to the
% physical/network nominal frequency 50 Hz.  Do NOT modify FFT fundamental,
% x0status, line parameters, OpComm surrogate, machine solvers, controllers,
% or any physical-gate thresholds.

repoRoot=pwd;
outRoot=fullfile(repoRoot,'build','ieee39_r2024a');
sourceDir=fullfile(outRoot,'source','model');
modelPath=fullfile(outRoot,'migrated','IEEE39bus_R2024a.slx');
assert(exist(modelPath,'file')==2,'Migrated IEEE39 model not found.');
assert(exist(sourceDir,'dir')==7,'Source-model support directory not found.');

addpath(sourceDir);
oldDir=pwd; dirCleanup=onCleanup(@()restore_dir(oldDir)); %#ok<NASGU>
cd(sourceDir);
[~,mdl,~]=fileparts(modelPath);
load_system(modelPath);
modelCleanup=onCleanup(@()safe_close(mdl)); %#ok<NASGU>

pg=[mdl '/powergui'];
assert(getSimulinkBlockHandle(pg)>0,'powergui block not found.');

beforeFreq=get_param(pg,'frequency');
beforeFund=''; beforeX0=''; beforeMode=''; beforeTs=''; beforeSolver='';
try, beforeFund=get_param(pg,'fundamental'); catch, end
try, beforeX0=get_param(pg,'x0status'); catch, end
try, beforeMode=get_param(pg,'SimulationMode'); catch, end
try, beforeTs=get_param(pg,'SampleTime'); catch, end
try, beforeSolver=get_param(pg,'SolverType'); catch, end

% Precondition is itself part of the A/B contract: if a future model no
% longer carries the released 60-Hz value, this experiment must not silently
% mutate some other baseline.
beforeNumeric=resolve_numeric(beforeFreq,pg);
assert(isscalar(beforeNumeric) && abs(beforeNumeric-60)<1e-12, ...
    'Expected released powergui.frequency=60 before A/B; got %s.',beforeFreq);

set_param(pg,'frequency','50');
afterFreq=get_param(pg,'frequency');
afterNumeric=resolve_numeric(afterFreq,pg);
assert(isscalar(afterNumeric) && abs(afterNumeric-50)<1e-12, ...
    'Failed to set powergui.frequency to 50.');

% Verify every protected field is unchanged by this single-variable variant.
unchangedFund=read_same(pg,'fundamental',beforeFund);
unchangedX0=read_same(pg,'x0status',beforeX0);
unchangedMode=read_same(pg,'SimulationMode',beforeMode);
unchangedTs=read_same(pg,'SampleTime',beforeTs);
unchangedSolver=read_same(pg,'SolverType',beforeSolver);
assert(unchangedFund && unchangedX0 && unchangedMode && unchangedTs && unchangedSolver, ...
    'Powergui frequency A/B modified a protected powergui field.');

save_system(mdl,modelPath);

report=struct;
report.release=version('-release');
report.variant='powergui_initialization_frequency_60_to_50';
report.changed_field='powergui.frequency';
report.before=beforeFreq;
report.after=afterFreq;
report.before_hz=beforeNumeric;
report.after_hz=afterNumeric;
report.protected_fields=struct('fundamental',beforeFund,'x0status',beforeX0, ...
    'SimulationMode',beforeMode,'SampleTime',beforeTs,'SolverType',beforeSolver);
report.changed_field_count=1;
report.line_parameters_changed=false;
report.opcomm_surrogate_changed=false;
report.machine_solver_changed=false;
report.controller_parameters_changed=false;
report.audit_thresholds_changed=false;
report.timestamp_utc=char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
write_report(fullfile(outRoot,'powergui_frequency_variant.json'),report);

fprintf('IEEE39 powergui frequency A/B: frequency %.12g -> %.12g Hz; changed=1\n', ...
    beforeNumeric,afterNumeric);
end

function tf=read_same(block,param,before)
try
    now=get_param(block,param);
    tf=strcmp(now,before);
catch
    tf=isempty(before);
end
end

function v=resolve_numeric(expr,b)
try, v=slResolve(expr,b); catch, v=evalin('base',expr); end
v=double(v(:).');
end
function restore_dir(p), try, cd(p); catch, end, end
function safe_close(mdl), try, if bdIsLoaded(mdl), close_system(mdl,0); end, catch, end, end
function write_report(p,r), fid=fopen(p,'w'); assert(fid>0); fwrite(fid,jsonencode(r,'PrettyPrint',true)); fclose(fid); end
