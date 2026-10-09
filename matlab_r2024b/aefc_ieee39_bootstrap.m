function aefc_ieee39_bootstrap
% Bootstrap the native MathWorks IEEE39BusSystem on a headless CI runner.
% Public GitHub runners can execute MATLAB/Simulink, but openExample rejects
% headless configurations before it reaches the example resource backend.
% We therefore reuse MathWorks' own findExample/setupExample backend directly
% and never substitute a third-party IEEE39 model.

assert(~verLessThan('matlab','24.2'), ...
    'AEFC:ReleaseTooOld', 'R2024b or later is required for IEEE39BusSystem.');

repo = pwd;
outDir = fullfile(repo,'Figures','IEEE39-R2024b');
resDir = fullfile(repo,'results','ieee39_r2024b');
if ~exist(outDir,'dir'), mkdir(outDir); end
if ~exist(resDir,'dir'), mkdir(resDir); end

modelName = 'IEEE39BusSystem';
resolvedModel = '';
resolutionMethod = '';
headless = struct;
headless.openExample_path = which('openExample');
headless.findExample_path = which('findExample');
headless.setupExample_path = which('setupExample');
headless.attempts = struct('id',{},'success',{},'resolved_model',{},'error_identifier',{},'error_message',{});

% 1) Already on path.
hit = which([modelName '.slx']);
if ~isempty(hit)
    resolvedModel = hit;
    resolutionMethod = 'which';
end

% 2) Shell-side asset inventory produced by the workflow.
assetFile = fullfile(resDir,'install_asset_search.txt');
if isempty(resolvedModel) && exist(assetFile,'file')==2
    lines = splitlines(string(fileread(assetFile)));
    for i=1:numel(lines)
        p = strtrim(lines(i));
        if strlength(p)==0, continue; end
        [~,n,e] = fileparts(char(p));
        if strcmpi(n,modelName) && strcmpi(e,'.slx') && exist(char(p),'file')==2
            resolvedModel = char(p);
            resolutionMethod = 'workflow-file-search';
            break;
        end
    end
end

% 3) Installed product tree.
if isempty(resolvedModel)
    matches = dir(fullfile(matlabroot,'**',[modelName '.slx']));
    if ~isempty(matches)
        resolvedModel = fullfile(matches(1).folder,matches(1).name);
        resolutionMethod = 'matlabroot-recursive-search';
    end
end

% 4) Headless official-example setup. openExample.m performs a display check
% before calling findExample/setupExample. These resource functions themselves
% are the backend used by MathWorks to resolve and materialize the example.
if isempty(resolvedModel)
    oe = which('openExample');
    if ~isempty(oe), addpath(fileparts(oe)); end
    headless.findExample_path = which('findExample');
    headless.setupExample_path = which('setupExample');
    ids = {'sps/IEEE39BusSystemExample', ...
           'simscapeelectrical/IEEE39BusSystemExample', ...
           'sps/IEEE39BusSystem', ...
           'simscapeelectrical/IEEE39BusSystem'};
    workRoot = fullfile(resDir,'official_mathworks_example');
    if ~exist(workRoot,'dir'), mkdir(workRoot); end
    for k=1:numel(ids)
        a = struct('id',ids{k},'success',false,'resolved_model','', ...
                   'error_identifier','','error_message','');
        try
            metadata = findExample(ids{k});
            workDir = fullfile(workRoot,sprintf('candidate_%02d',k));
            if ~exist(workDir,'dir'), mkdir(workDir); end
            [materializedDir,~] = setupExample(metadata,workDir);
            m = dir(fullfile(materializedDir,'**',[modelName '.slx']));
            if isempty(m), m = dir(fullfile(workDir,'**',[modelName '.slx'])); end
            if ~isempty(m)
                resolvedModel = fullfile(m(1).folder,m(1).name);
                resolutionMethod = ['headless-findExample-setupExample:' ids{k}];
                a.success = true;
                a.resolved_model = resolvedModel;
                headless.attempts(end+1) = a; %#ok<AGROW>
                break;
            else
                error('AEFC:HeadlessExampleNoModel', ...
                    'Official example materialized but IEEE39BusSystem.slx was not found for %s.',ids{k});
            end
        catch ME
            a.error_identifier = ME.identifier;
            a.error_message = ME.message;
            headless.attempts(end+1) = a; %#ok<AGROW>
        end
    end
end
writejson(fullfile(resDir,'headless_example_setup.json'),headless);

resolution = struct;
resolution.model = modelName;
resolution.matlab_release = version('-release');
resolution.matlabroot = matlabroot;
resolution.resolved_model = resolvedModel;
resolution.resolution_method = resolutionMethod;
resolution.openExample_used = false;
resolution.headless_backend_attempted = true;
writejson(fullfile(resDir,'model_resolution.json'),resolution);

assert(~isempty(resolvedModel),'AEFC:ModelNotFound', ...
    ['The official IEEE39BusSystem example could not be materialized on the headless runner. ' ...
     'Inspect headless_example_setup.json and MathWorks example-backend diagnostics.']);

% Preserve an immutable copy without renaming the currently loaded block
% diagram. save_system(model,newPath) changes the loaded diagram identity;
% copyfile avoids that and keeps all subsequent calls addressed to
% IEEE39BusSystem.
copyPath = fullfile(outDir,'IEEE39BusSystem_AEFC.slx');
copyfile(resolvedModel,copyPath,'f');
load_system(resolvedModel);
assert(bdIsLoaded(modelName),'AEFC:ModelNotLoaded','IEEE39BusSystem is not loaded.');

try, set_param(modelName,'SignalLogging','on'); catch, end
try, set_param(modelName,'SaveOutput','on'); catch, end
try, set_param(modelName,'SaveTime','on'); catch, end

blocks = find_system(modelName,'LookUnderMasks','all','FollowLinks','on','Type','Block');
records = repmat(struct('path','','name','','block_type','','mask_type','','reference_block',''),numel(blocks),1);
for i = 1:numel(blocks)
    b = blocks{i};
    records(i).path = b;
    records(i).name = get_param(b,'Name');
    records(i).block_type = safeget(b,'BlockType');
    records(i).mask_type = safeget(b,'MaskType');
    records(i).reference_block = safeget(b,'ReferenceBlock');
end

patterns = {'Generators','Measurements','AVR','Exciter','Governor','PSS','Speed','Rotor','Voltage','Vref','Pref','Load','Fault','Bus'};
interesting = false(numel(records),1);
for i = 1:numel(records)
    txt = lower([records(i).path ' ' records(i).name ' ' records(i).mask_type ' ' records(i).reference_block]);
    for j = 1:numel(patterns)
        if contains(txt,lower(patterns{j}))
            interesting(i) = true;
            break;
        end
    end
end

inv = struct;
inv.evidence_source = 'real_mathworks_ieee39_simscape';
inv.migration_target = 'R2024b';
inv.matlab_release = version('-release');
inv.matlab_version = version;
inv.model = modelName;
inv.resolved_model = resolvedModel;
inv.resolution_method = resolutionMethod;
inv.saved_model = strrep(copyPath,[repo filesep],'');
inv.legacy_archive = 'Figures/IEEE-39-bus-power.zip';
inv.legacy_git_blob_sha = '41db586d592851c4a81205a4cc5c7c770b7a0c48';
inv.total_blocks = numel(blocks);
inv.interesting_blocks = records(interesting);
writejson(fullfile(resDir,'bootstrap_inventory.json'),inv);

smoke = struct;
smoke.evidence_source = inv.evidence_source;
smoke.matlab_release = inv.matlab_release;
smoke.model = modelName;
smoke.stop_time_s = 0.20;
smoke.success = false;
smoke.error_identifier = '';
smoke.error_message = '';
smoke.output_fields = {};
smoke.final_time = NaN;

try
    simIn = Simulink.SimulationInput(modelName);
    simIn = simIn.setModelParameter('StopTime',num2str(smoke.stop_time_s));
    out = sim(simIn);
    smoke.output_fields = fieldnames(out);
    try
        t = out.tout;
        if ~isempty(t), smoke.final_time = t(end); end
    catch
    end
    smoke.success = true;
catch ME
    smoke.error_identifier = ME.identifier;
    smoke.error_message = ME.message;
    writejson(fullfile(resDir,'smoke_summary.json'),smoke);
    close_system(modelName,0);
    rethrow(ME);
end

writejson(fullfile(resDir,'smoke_summary.json'),smoke);
close_system(modelName,0);
end

function v = safeget(block,param)
try
    v = get_param(block,param);
    if isstring(v), v = char(v); end
catch
    v = '';
end
end

function writejson(path,obj)
fid = fopen(path,'w');
assert(fid>0,'AEFC:IO','Cannot open %s for writing.',path);
c = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(obj,'PrettyPrint',true),'char');
fwrite(fid,sprintf('\n'),'char');
end
