function ieee39_apply_machine_solver_variant
% Diagnostic A/B variant for R2024a numerical fidelity.
% Change only the discrete solver used by the ten SPS synchronous machines.
% Do not change network parameters, controller gains, audit thresholds, or
% the OpComm surrogate. The strict bridge audit decides whether this variant
% improves the clean physical baseline.

repoRoot=pwd;
outRoot=fullfile(repoRoot,'build','ieee39_r2024a');
modelPath=fullfile(outRoot,'migrated','IEEE39bus_R2024a.slx');
assert(exist(modelPath,'file')==2,'Migrated IEEE39 model not found.');
[~,mdl,~]=fileparts(modelPath);
load_system(modelPath);
cleanupObj=onCleanup(@()safe_close(mdl)); %#ok<NASGU>

blocks=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block');
machines={};
for i=1:numel(blocks)
    b=blocks{i}; st='';
    try, st=get_param(b,'SourceType'); catch, end
    if strcmpi(strtrim(st),'Synchronous Machine')
        machines{end+1}=b; %#ok<AGROW>
    end
end
machines=unique(machines,'stable');
assert(numel(machines)==10,'Expected exactly 10 synchronous machines, found %d.',numel(machines));

records=cell(1,numel(machines));
for i=1:numel(machines)
    b=machines{i}; dp=get_param(b,'DialogParameters'); names=fieldnames(dp);
    solverField=''; before='';
    for j=1:numel(names)
        f=names{j}; v='';
        try, v=get_param(b,f); catch, continue; end
        if ischar(v) && (strcmpi(strtrim(v),'Forward Euler') || contains(lower(v),'trapezoidal') || contains(lower(v),'backward euler'))
            prompt='';
            try, prompt=dp.(f).Prompt; catch, end
            if strcmpi(strtrim(v),'Forward Euler') || contains(lower(prompt),'discrete solver')
                solverField=f; before=v; break;
            end
        end
    end
    if isempty(solverField)
        % Fallback: inspect prompts even if the current value is release-specific.
        for j=1:numel(names)
            f=names{j}; prompt='';
            try, prompt=dp.(f).Prompt; catch, end
            if contains(lower(prompt),'discrete solver')
                solverField=f; try, before=get_param(b,f); catch, before=''; end; break;
            end
        end
    end
    assert(~isempty(solverField),'Could not identify discrete solver parameter for %s.',b);
    set_param(b,solverField,'Trapezoidal robust');
    after=get_param(b,solverField);
    assert(strcmpi(strtrim(after),'Trapezoidal robust'),'Robust solver was not applied to %s.',b);
    records{i}=struct('block',b,'parameter',solverField,'before',before,'after',after);
end

save_system(mdl,modelPath);
report=struct('release',version('-release'), ...
    'variant','synchronous_machine_trapezoidal_robust', ...
    'machine_count',numel(machines), ...
    'records',{records}, ...
    'network_parameters_changed',false, ...
    'controller_parameters_changed',false, ...
    'audit_thresholds_changed',false, ...
    'timestamp_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z''')));
write_report(fullfile(outRoot,'machine_solver_variant.json'),report);
fprintf('IEEE39 machine solver variant: machines=%d solver=Trapezoidal robust\n',numel(machines));
end

function safe_close(mdl)
try, if bdIsLoaded(mdl), close_system(mdl,0); end, catch, end
end
function write_report(p,r)
fid=fopen(p,'w'); assert(fid>0); fwrite(fid,jsonencode(r,'PrettyPrint',true)); fclose(fid);
end
