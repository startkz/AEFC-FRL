function ieee39_machine_solver_inventory
% Read-only inventory of R2024a synchronous-machine solver parameters.
% This function changes no model parameter. It maps machine paths from the
% audited raw-MDL provenance and records every dialog field whose name,
% prompt, or current value mentions solver/discrete/euler/trapezoidal/
% iteration/robust. The resulting JSON is used to select an exact A/B field.

repoRoot=pwd;
outRoot=fullfile(repoRoot,'build','ieee39_r2024a');
sourceModelDir=fullfile(outRoot,'source','model');
modelPath=fullfile(outRoot,'migrated','IEEE39bus_R2024a.slx');
provPath=fullfile(repoRoot,'build','ieee39_source_provenance.json');
assert(exist(modelPath,'file')==2,'Migrated IEEE39 model not found.');
assert(exist(sourceModelDir,'dir')==7,'Source-model support directory not found.');
assert(exist(provPath,'file')==2,'IEEE39 source provenance JSON not found.');

addpath(sourceModelDir);
oldDir=pwd; cdCleanup=onCleanup(@()restore_dir(oldDir)); %#ok<NASGU>
cd(sourceModelDir);
[~,mdl,~]=fileparts(modelPath);
load_system(modelPath);
modelCleanup=onCleanup(@()safe_close(mdl)); %#ok<NASGU>
prov=jsondecode(fileread(provPath));

records=cell(1,10);
for g=1:10
    sourcePath=prov.generators(g).machine_path;
    prefix='IEEE39bus/';
    assert(startsWith(sourcePath,prefix),'Unexpected source machine path: %s',sourcePath);
    b=[mdl '/' extractAfter(sourcePath,strlength(prefix))];
    assert(getSimulinkBlockHandle(b)>0,'Mapped synchronous machine does not exist: %s',b);
    dp=get_param(b,'DialogParameters'); names=fieldnames(dp);
    hits={}; allStringFields={};
    for j=1:numel(names)
        f=names{j}; prompt=''; value='';
        try, prompt=dp.(f).Prompt; catch, end
        try, value=get_param(b,f); catch, end
        if ~ischar(prompt), prompt=char(string(prompt)); end
        if isstring(value), value=char(value); end
        if ischar(value)
            allStringFields{end+1}=struct('name',f,'prompt',prompt,'value',value); %#ok<AGROW>
        else
            try, value=char(string(value)); catch, value=''; end
        end
        text=lower([f ' ' prompt ' ' value]);
        if contains(text,'solver') || contains(text,'discrete') || contains(text,'euler') || ...
                contains(text,'trapezoidal') || contains(text,'iterat') || contains(text,'robust')
            hits{end+1}=struct('name',f,'prompt',prompt,'value',value); %#ok<AGROW>
        end
    end
    records{g}=struct('generator',g,'block',b,'solver_related',{hits},'all_string_fields',{allStringFields});
end

report=struct('release',version('-release'),'read_only',true,'machine_count',10, ...
    'records',{records}, ...
    'timestamp_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z''')));
write_report(fullfile(outRoot,'machine_solver_inventory.json'),report);
fprintf('IEEE39 machine solver inventory: machines=10 read_only=1\n');
for g=1:10
    fprintf('  G%d:\n',g);
    for k=1:numel(records{g}.solver_related)
        x=records{g}.solver_related{k};
        fprintf('    %s | %s | %s\n',x.name,x.prompt,x.value);
    end
end
end

function restore_dir(p), try, cd(p); catch, end, end
function safe_close(mdl), try, if bdIsLoaded(mdl), close_system(mdl,0); end, catch, end, end
function write_report(p,r), fid=fopen(p,'w'); assert(fid>0); fwrite(fid,jsonencode(r,'PrettyPrint',true)); fclose(fid); end
