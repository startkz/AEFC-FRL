function ieee39_apply_machine_solver_variant
% Diagnostic A/B variant for R2024a numerical fidelity.
% Change only the discrete solver used by the ten SPS synchronous machines.
% Do not change network parameters, controller gains, audit thresholds, or
% the OpComm surrogate. The strict bridge audit decides whether this variant
% improves the clean physical baseline.

repoRoot=pwd;
outRoot=fullfile(repoRoot,'build','ieee39_r2024a');
sourceModelDir=fullfile(outRoot,'source','model');
modelPath=fullfile(outRoot,'migrated','IEEE39bus_R2024a.slx');
provPath=fullfile(repoRoot,'build','ieee39_source_provenance.json');
assert(exist(modelPath,'file')==2,'Migrated IEEE39 model not found.');
assert(exist(sourceModelDir,'dir')==7,'Migrated source-model support directory not found.');
assert(exist(provPath,'file')==2,'IEEE39 source provenance JSON not found.');

% The migrated model retains the original callbacks. Resolve their support
% files before loading so PostLoadFcn can execute exactly as in migration/audit.
addpath(sourceModelDir);
oldDir=pwd;
cdCleanup=onCleanup(@()restore_dir(oldDir)); %#ok<NASGU>
cd(sourceModelDir);
[~,mdl,~]=fileparts(modelPath);
load_system(modelPath);
modelCleanup=onCleanup(@()safe_close(mdl)); %#ok<NASGU>

% Do not rediscover legacy-linked machine metadata through unavailable
% libraries. Reuse the already audited raw-MDL provenance and map only the
% top-level model name IEEE39bus -> IEEE39bus_R2024a.
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

records=cell(1,numel(machines));
for i=1:numel(machines)
    b=machines{i};
    dp=get_param(b,'DialogParameters'); names=fieldnames(dp);
    solverField=''; before='';
    % Prefer a parameter whose prompt explicitly names the discrete solver.
    for j=1:numel(names)
        f=names{j}; prompt='';
        try, prompt=dp.(f).Prompt; catch, end
        if contains(lower(prompt),'discrete solver')
            solverField=f;
            try, before=get_param(b,f); catch, before=''; end
            break;
        end
    end
    % Release-specific fallback: locate the current integration method value.
    if isempty(solverField)
        for j=1:numel(names)
            f=names{j}; v='';
            try, v=get_param(b,f); catch, continue; end
            if ischar(v) && (strcmpi(strtrim(v),'Forward Euler') || ...
                    contains(lower(v),'trapezoidal') || contains(lower(v),'backward euler'))
                solverField=f; before=v; break;
            end
        end
    end
    assert(~isempty(solverField),'Could not identify discrete solver parameter for %s.',b);
    set_param(b,solverField,'Trapezoidal robust');
    after=get_param(b,solverField);
    assert(strcmpi(strtrim(after),'Trapezoidal robust'),'Robust solver was not applied to %s.',b);
    records{i}=struct('generator',i,'block',b,'parameter',solverField,'before',before,'after',after);
end

save_system(mdl,modelPath);
report=struct('release',version('-release'), ...
    'variant','synchronous_machine_trapezoidal_robust', ...
    'machine_count',numel(machines), ...
    'records',{records}, ...
    'machine_paths_from_raw_mdl_provenance',true, ...
    'network_parameters_changed',false, ...
    'controller_parameters_changed',false, ...
    'audit_thresholds_changed',false, ...
    'timestamp_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z''')));
write_report(fullfile(outRoot,'machine_solver_variant.json'),report);
fprintf('IEEE39 machine solver variant: machines=%d solver=Trapezoidal robust\n',numel(machines));
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
