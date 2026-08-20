function aefc_ieee39_bootstrap
% Bootstrap a GitHub-Actions-compatible IEEE 39-bus plant using the
% MathWorks native Simscape Electrical model available in R2024b+.
% No interactive openExample call is used because that API is unavailable
% in the GitHub batch configuration.

assert(~verLessThan('matlab','24.2'), ...
    'AEFC:ReleaseTooOld', 'R2024b or later is required for IEEE39BusSystem.');

repo = pwd;
outDir = fullfile(repo,'generated','IEEE39-R2024b');
resDir = fullfile(repo,'results','ieee39_r2024b');
if ~exist(outDir,'dir'), mkdir(outDir); end
if ~exist(resDir,'dir'), mkdir(resDir); end

modelName = 'IEEE39BusSystem';
sourceModel = '';

% First try normal MATLAB path resolution.
hit = which([modelName '.slx']);
if ~isempty(hit)
    sourceModel = hit;
end

% GitHub-hosted MATLAB does not support openExample.  The workflow probes
% the installed product tree with the host `find` command and records all
% matching assets here; consume that list directly.
probeFile = fullfile(resDir,'installed_ieee39_files.txt');
if isempty(sourceModel) && exist(probeFile,'file')
    txt = fileread(probeFile);
    lines = regexp(strtrim(txt),'\r?\n','split');
    for k = 1:numel(lines)
        p = strtrim(lines{k});
        if isempty(p), continue; end
        [~,n,e] = fileparts(p);
        if strcmpi(e,'.slx') && strcmpi(n,modelName)
            sourceModel = p;
            break;
        end
    end
    if isempty(sourceModel)
        for k = 1:numel(lines)
            p = strtrim(lines{k});
            if endsWith(lower(p),'.slx') && contains(lower(p),'ieee39')
                sourceModel = p;
                break;
            end
        end
    end
end

% Last non-interactive fallback: search the installed MATLAB tree from
% inside MATLAB.  This is deliberately filesystem-based, not openExample.
if isempty(sourceModel)
    roots = {matlabroot, fullfile(matlabroot,'toolbox')};
    for r = 1:numel(roots)
        d = dir(fullfile(roots{r},'**','*IEEE39*.slx'));
        if ~isempty(d)
            sourceModel = fullfile(d(1).folder,d(1).name);
            break;
        end
    end
end

if isempty(sourceModel)
    diag = struct;
    diag.success = false;
    diag.stage = 'asset-discovery';
    diag.matlab_release = version('-release');
    diag.matlabroot = matlabroot;
    diag.probe_file_exists = exist(probeFile,'file') == 2;
    if diag.probe_file_exists
        diag.probe_contents = fileread(probeFile);
    else
        diag.probe_contents = '';
    end
    writejson(fullfile(resDir,'asset_discovery.json'),diag);
    error('AEFC:ModelAssetNotFound', ...
        'IEEE39BusSystem SLX asset was not found in the installed R2024b product tree.');
end

load_system(sourceModel);
loadedName = bdroot(modelName);
if ~bdIsLoaded(modelName)
    [~,loadedName] = fileparts(sourceModel);
end
assert(bdIsLoaded(loadedName),'AEFC:ModelNotLoaded','IEEE39 model is not loaded.');

try, set_param(loadedName,'SignalLogging','on'); catch, end
try, set_param(loadedName,'SaveOutput','on'); catch, end
try, set_param(loadedName,'SaveTime','on'); catch, end

copyPath = fullfile(outDir,'IEEE39BusSystem_AEFC.slx');
save_system(loadedName, copyPath);

blocks = find_system(loadedName,'LookUnderMasks','all','FollowLinks','on','Type','Block');
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
inv.runner_repository = 'startkz/AEFC-FRL';
inv.paper_repository = 'startkz/AEFC';
inv.migration_target = 'R2024b';
inv.matlab_release = version('-release');
inv.matlab_version = version;
inv.model = loadedName;
inv.source_model = sourceModel;
inv.saved_model = strrep(copyPath,[repo filesep],'');
inv.legacy_archive = 'AEFC/Figures/IEEE-39-bus-power.zip';
inv.legacy_git_blob_sha = '41db586d592851c4a81205a4cc5c7c770b7a0c48';
inv.total_blocks = numel(blocks);
inv.interesting_blocks = records(interesting);
writejson(fullfile(resDir,'bootstrap_inventory.json'),inv);

smoke = struct;
smoke.evidence_source = inv.evidence_source;
smoke.matlab_release = inv.matlab_release;
smoke.model = loadedName;
smoke.source_model = sourceModel;
smoke.stop_time_s = 0.20;
smoke.success = false;
smoke.error_identifier = '';
smoke.error_message = '';
smoke.output_fields = {};
smoke.final_time = NaN;

try
    simIn = Simulink.SimulationInput(loadedName);
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
    rethrow(ME);
end

writejson(fullfile(resDir,'smoke_summary.json'),smoke);
save_system(loadedName, copyPath);
close_system(loadedName,0);
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
c = onCleanup(@() fclose(fid));
fwrite(fid,jsonencode(obj,'PrettyPrint',true),'char');
fwrite(fid,sprintf('\n'),'char');
end
