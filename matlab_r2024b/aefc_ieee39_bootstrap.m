function aefc_ieee39_bootstrap
% Bootstrap a GitHub-Actions-compatible IEEE 39-bus plant using the
% MathWorks native Simscape Electrical example introduced in R2024b.
% This public runner repo is used only to obtain MathWorks public-project
% batch licensing; the legacy AEFC paper archive remains private.

assert(~verLessThan('matlab','24.2'), ...
    'AEFC:ReleaseTooOld', 'R2024b or later is required for IEEE39BusSystem.');

repo = pwd;
outDir = fullfile(repo,'generated','IEEE39-R2024b');
resDir = fullfile(repo,'results','ieee39_r2024b');
if ~exist(outDir,'dir'), mkdir(outDir); end
if ~exist(resDir,'dir'), mkdir(resDir); end

modelName = 'IEEE39BusSystem';
openedExample = '';

try
    load_system(modelName);
catch
    ids = { ...
        'simscapeelectrical/IEEE39BusSystemExample', ...
        'simscapeelectrical/IEEE39BusSystem', ...
        'sps/IEEE39BusSystemExample'};
    lastErr = [];
    for k = 1:numel(ids)
        try
            openExample(ids{k});
            openedExample = ids{k};
            break;
        catch ME
            lastErr = ME;
        end
    end
    if ~bdIsLoaded(modelName)
        hit = which([modelName '.slx']);
        if ~isempty(hit)
            load_system(hit);
        elseif ~isempty(lastErr)
            rethrow(lastErr);
        else
            error('AEFC:ModelNotFound','Could not locate IEEE39BusSystem.');
        end
    end
end

assert(bdIsLoaded(modelName),'AEFC:ModelNotLoaded','IEEE39BusSystem is not loaded.');

try, set_param(modelName,'SignalLogging','on'); catch, end
try, set_param(modelName,'SaveOutput','on'); catch, end
try, set_param(modelName,'SaveTime','on'); catch, end

copyPath = fullfile(outDir,'IEEE39BusSystem_AEFC.slx');
save_system(modelName, copyPath);

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
inv.runner_repository = 'startkz/AEFC-FRL';
inv.paper_repository = 'startkz/AEFC';
inv.migration_target = 'R2024b';
inv.matlab_release = version('-release');
inv.matlab_version = version;
inv.model = modelName;
inv.saved_model = strrep(copyPath,[repo filesep],'');
inv.opened_example = openedExample;
inv.legacy_archive = 'AEFC/Figures/IEEE-39-bus-power.zip';
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
    rethrow(ME);
end

writejson(fullfile(resDir,'smoke_summary.json'),smoke);
save_system(modelName, copyPath);
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
c = onCleanup(@() fclose(fid));
fwrite(fid,jsonencode(obj,'PrettyPrint',true),'char');
fwrite(fid,sprintf('\n'),'char');
end
