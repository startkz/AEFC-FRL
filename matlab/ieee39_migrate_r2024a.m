function ieee39_migrate_r2024a
% Conservative migration/audit for the DESL-EPFL full-replica IEEE39 model.
% Declares success only after the migrated model updates and advances in a
% genuine Simulink simulation on a GitHub-hosted MATLAB runner.

repoRoot = pwd;
srcZip = fullfile(repoRoot,'Figures','IEEE-39-bus-power.zip');
outRoot = fullfile(repoRoot,'build','ieee39_r2024a');
srcRoot = fullfile(outRoot,'source');
migRoot = fullfile(outRoot,'migrated');
if exist(outRoot,'dir'), rmdir(outRoot,'s'); end
mkdir(srcRoot); mkdir(migRoot);
assert(exist(srcZip,'file')==2,'IEEE39 archive not found: %s',srcZip);
unzip(srcZip,srcRoot);
addpath(genpath(srcRoot));

report = struct;
report.matlab_version = version;
report.release = version('-release');
report.source_zip = 'Figures/IEEE-39-bus-power.zip';
report.source_git_blob = '41db586d592851c4a81205a4cc5c7c770b7a0c48';
report.upstream = 'DESL-EPFL/IEEE-39-bus-power-system:model.zip';
report.timestamp_utc = char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
report.models = {};
report.runnable = false;
report.real_simulation_completed = false;

files = [dir(fullfile(srcRoot,'**','*.slx')); dir(fullfile(srcRoot,'**','*.mdl'))];
assert(~isempty(files),'No .slx/.mdl model found after unzip.');
[~,order] = sort([files.bytes],'descend');
files = files(order);
modelRecords = cell(1,numel(files));
primaryName = '';
primaryPath = '';

for k = 1:numel(files)
    f = files(k);
    fullp = fullfile(f.folder,f.name);
    [~,mdl,~] = fileparts(fullp);
    rec = struct('file',strrep(fullp,[repoRoot filesep],''),'name',mdl, ...
        'bytes',f.bytes,'load_ok',false,'update_ok',false,'third_party_blocks',{{}}, ...
        'broken_links',{{}},'error','');
    try
        load_system(fullp);
        rec.load_ok = true;
        if isempty(primaryName)
            primaryName = mdl;
            primaryPath = fullp;
        end
        blocks = find_system(mdl,'FollowLinks','on','LookUnderMasks','all','Type','Block');
        third = {};
        broken = {};
        for b = 1:numel(blocks)
            blk = blocks{b};
            vals = {};
            props = {'ReferenceBlock','MaskType','BlockType','LinkStatus'};
            for p = 1:numel(props)
                try, vals{end+1} = get_param(blk,props{p}); catch, end %#ok<AGROW>
            end
            joined = lower(strjoin(cellfun(@char,vals,'UniformOutput',false),'|'));
            if contains(joined,'artemis') || contains(joined,'opal') || ...
                    contains(joined,'rt-lab') || contains(joined,'rtlab') || ...
                    contains(joined,'emegasim') || contains(joined,'hypersim')
                third{end+1} = blk; %#ok<AGROW>
            end
            try
                ls = get_param(blk,'LinkStatus');
                if strcmpi(ls,'unresolved') || strcmpi(ls,'inactive')
                    broken{end+1} = blk; %#ok<AGROW>
                end
            catch
            end
        end
        rec.third_party_blocks = third;
        rec.broken_links = broken;
        try
            set_param(mdl,'SimulationCommand','update');
            rec.update_ok = true;
        catch ME
            rec.error = ['update: ' ME.message];
        end
        close_system(mdl,0);
    catch ME
        rec.error = ['load: ' ME.message];
        try, close_system(mdl,0); catch, end
    end
    modelRecords{k} = rec;
end
report.models = modelRecords;
report.primary_source_model = strrep(primaryPath,[repoRoot filesep],'');

assert(~isempty(primaryName),'No model could be loaded in current MATLAB release.');
load_system(primaryPath);
migratedPath = fullfile(migRoot,[primaryName '_R2024a.slx']);
save_system(primaryName,migratedPath);
report.migrated_model = strrep(migratedPath,[repoRoot filesep],'');
close_system(primaryName,0);

[~,migratedName,~] = fileparts(migratedPath);
load_system(migratedPath);
try
    set_param(migratedName,'SimulationCommand','update');
    report.primary_update_ok = true;
catch ME
    report.primary_update_ok = false;
    report.primary_update_error = ME.message;
end

if report.primary_update_ok
    try
        oldStop = get_param(migratedName,'StopTime');
        cleanupStop = onCleanup(@() set_param(migratedName,'StopTime',oldStop)); %#ok<NASGU>
        set_param(migratedName,'StopTime','0.02');
        simOut = sim(migratedName,'ReturnWorkspaceOutputs','on'); %#ok<NASGU>
        report.real_simulation_completed = true;
        report.runnable = true;
    catch ME
        report.short_sim_error = ME.message;
    end
end
close_system(migratedName,0);

jsonPath = fullfile(outRoot,'migration_report.json');
fid = fopen(jsonPath,'w');
assert(fid>0,'Cannot create migration report.');
fwrite(fid,jsonencode(report,'PrettyPrint',true));
fclose(fid);

fprintf('IEEE39 migration report: %s\n',jsonPath);
fprintf('runnable=%d real_simulation_completed=%d\n',report.runnable,report.real_simulation_completed);
if ~report.runnable
    error('AEFC:IEEE39MigrationBlocked','Migrated model did not complete a real short Simulink simulation. See migration_report.json.');
end
end
