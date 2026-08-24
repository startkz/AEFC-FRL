function ieee39_apply_machine_solver_variant
% Exact single-variable A/B for the R2024a IEEE39 migration.
% Inventory evidence shows that only G1 uses IterativeModel='Forward Euler';
% G2--G10 already use 'Trapezoidal non iterative'.  Keep every other solver
% parameter unchanged and modify only G1.IterativeModel.  The strict bridge
% audit decides whether this single change improves the clean physical baseline.

repoRoot=pwd;
outRoot=fullfile(repoRoot,'build','ieee39_r2024a');
sourceModelDir=fullfile(outRoot,'source','model');
modelPath=fullfile(outRoot,'migrated','IEEE39bus_R2024a.slx');
provPath=fullfile(repoRoot,'build','ieee39_source_provenance.json');
assert(exist(modelPath,'file')==2,'Migrated IEEE39 model not found.');
assert(exist(sourceModelDir,'dir')==7,'Migrated source-model support directory not found.');
assert(exist(provPath,'file')==2,'IEEE39 source provenance JSON not found.');

addpath(sourceModelDir);
oldDir=pwd;
cdCleanup=onCleanup(@()restore_dir(oldDir)); %#ok<NASGU>
cd(sourceModelDir);
[~,mdl,~]=fileparts(modelPath);
load_system(modelPath);
modelCleanup=onCleanup(@()safe_close(mdl)); %#ok<NASGU>

prov=jsondecode(fileread(provPath));
assert(numel(prov.generators)==10,'Expected 10 generators in source provenance.');
machines=cell(1,10);
for g=1:10
    sourcePath=prov.generators(g).machine_path;
    prefix='IEEE39bus/';
    assert(startsWith(sourcePath,prefix),'Unexpected source machine path: %s',sourcePath);
    machines{g}=[mdl '/' extractAfter(sourcePath,strlength(prefix))];
    assert(getSimulinkBlockHandle(machines{g})>0,'Mapped synchronous machine does not exist: %s',machines{g});
end

% Record the complete solver-related state before modification so the A/B
% remains auditable and proves that exactly one mask value changed.
before=cell(1,10);
for g=1:10
    before{g}=struct('generator',g, ...
        'IterativeModel',get_param(machines{g},'IterativeModel'), ...
        'IterativeDiscreteModel',get_param(machines{g},'IterativeDiscreteModel'));
end
assert(strcmpi(strtrim(before{1}.IterativeModel),'Forward Euler'), ...
    'Inventory contract changed: G1 IterativeModel is %s, expected Forward Euler.',before{1}.IterativeModel);
for g=2:10
    assert(strcmpi(strtrim(before{g}.IterativeModel),'Trapezoidal non iterative'), ...
        'Inventory contract changed: G%d IterativeModel is %s.',g,before{g}.IterativeModel);
end
for g=1:10
    assert(strcmpi(strtrim(before{g}.IterativeDiscreteModel),'Trapezoidal non iterative'), ...
        'Inventory contract changed: G%d IterativeDiscreteModel is %s.',g,before{g}.IterativeDiscreteModel);
end

% Exact single variable under test.
set_param(machines{1},'IterativeModel','Trapezoidal non iterative');
assert(strcmpi(strtrim(get_param(machines{1},'IterativeModel')),'Trapezoidal non iterative'), ...
    'Failed to apply the G1 IterativeModel A/B variant.');

after=cell(1,10);
changed=0;
for g=1:10
    after{g}=struct('generator',g, ...
        'IterativeModel',get_param(machines{g},'IterativeModel'), ...
        'IterativeDiscreteModel',get_param(machines{g},'IterativeDiscreteModel'));
    changed=changed + ~strcmp(before{g}.IterativeModel,after{g}.IterativeModel) ...
        + ~strcmp(before{g}.IterativeDiscreteModel,after{g}.IterativeDiscreteModel);
end
assert(changed==1,'Single-variable contract violated: %d solver-related values changed.',changed);

save_system(mdl,modelPath);
report=struct('release',version('-release'), ...
    'variant','G1_IterativeModel_forward_euler_to_trapezoidal_non_iterative', ...
    'changed_generator',1, ...
    'changed_parameter','IterativeModel', ...
    'before_value','Forward Euler', ...
    'after_value','Trapezoidal non iterative', ...
    'solver_related_values_changed',changed, ...
    'before',{before}, ...
    'after',{after}, ...
    'machine_paths_from_raw_mdl_provenance',true, ...
    'network_parameters_changed',false, ...
    'controller_parameters_changed',false, ...
    'audit_thresholds_changed',false, ...
    'opcomm_surrogate_changed',false, ...
    'timestamp_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z''')));
write_report(fullfile(outRoot,'machine_solver_variant.json'),report);
fprintf('IEEE39 exact solver A/B: G1 IterativeModel Forward Euler -> Trapezoidal non iterative; changed=%d\n',changed);
end

function restore_dir(p)
try, cd(p); catch, end
end
function safe_close(mdl)
try, if bdIsLoaded(mdl), close_system(mdl,0); end, catch, end
end
function write_report(p,r)
fid=fopen(p,'w'); assert(fid>0); fwrite(fid,jsonencode(r,'PrettyPrint',true)); fclose(fid);
end
