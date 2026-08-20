function report = aefc_r2022b_probe(archivePath,outDir)
%AEFC_R2022B_PROBE Audit the real EPFL IEEE39 archive under MATLAB R2022b.
% No physical-network block is changed in this first pass. The probe records
% actual model names, initialization scripts, block/library provenance and the
% exact update/simulation errors needed for a release migration.

if nargin<1, archivePath='external/IEEE39/model.zip'; end
if nargin<2, outDir='build/ieee39_r2022b_probe'; end
if ~isfile(archivePath), error('Archive not found: %s',archivePath); end
if exist(outDir,'dir'), rmdir(outDir,'s'); end
mkdir(outDir); srcDir=fullfile(outDir,'source'); repDir=fullfile(outDir,'report'); upDir=fullfile(outDir,'upgraded');
mkdir(srcDir); mkdir(repDir); mkdir(upDir);
unzip(archivePath,srcDir); addpath(genpath(srcDir));

report=struct(); report.release=version('-release'); report.version=version; report.products=ver;
report.archive=archivePath; report.modelFiles={}; report.initScripts={}; report.models=struct([]);
M=[dir(fullfile(srcDir,'**','*.mdl')); dir(fullfile(srcDir,'**','*.slx'))];
for i=1:numel(M), report.modelFiles{end+1}=fullfile(M(i).folder,M(i).name); end
S=dir(fullfile(srcDir,'**','*.m'));
for i=1:numel(S), report.initScripts{end+1}=fullfile(S(i).folder,S(i).name); end
if isempty(report.modelFiles), write_report(report,repDir); error('No Simulink model found'); end

% Run only likely initialization/data scripts in the base workspace.
report.initRun={}; report.initErrors={};
for i=1:numel(report.initScripts)
    f=report.initScripts{i}; [~,n,~]=fileparts(f); t=lower(n);
    if contains(t,'init') || contains(t,'param') || contains(t,'data') || contains(t,'load')
        report.initRun{end+1}=f;
        try
            evalin('base',sprintf('run(''%s'')',strrep(f,'''','''''')));
        catch ME
            report.initErrors(end+1,:)={f,flaterr(ME)};
        end
    end
end

for i=1:numel(report.modelFiles)
    f=report.modelFiles{i}; [~,name,~]=fileparts(f);
    r=struct('source',f,'name',name,'loadOk',false,'blocks',0,'powergui',0,'machines',0, ...
        'runtimeBlocks',0,'updateOk',false,'updateError','','saveUpgradeOk',false, ...
        'upgradedFile','','upgradedUpdateOk',false,'upgradedUpdateError','', ...
        'smokeOk',false,'smokeError','','score',0);
    try
        load_system(f); r.loadOk=true;
        B=find_system(name,'LookUnderMasks','all','FollowLinks','on','Type','Block'); r.blocks=numel(B);
        invFile=fullfile(repDir,sprintf('%02d_%s_blocks.tsv',i,safe(name)));
        fid=fopen(invFile,'w'); fprintf(fid,'path\tblock_type\tmask_type\treference_block\tlink_status\truntime_token\n');
        for j=1:numel(B)
            b=B{j}; bt=g(b,'BlockType'); mt=g(b,'MaskType'); rb=g(b,'ReferenceBlock'); ls=g(b,'LinkStatus');
            tok=runtime_token([b ' ' bt ' ' mt ' ' rb]);
            if contains(lower([b ' ' mt ' ' rb]),'powergui'), r.powergui=r.powergui+1; end
            if contains(lower([b ' ' mt ' ' rb]),'synchronous') || contains(lower([b ' ' mt ' ' rb]),'machine'), r.machines=r.machines+1; end
            if ~isempty(tok), r.runtimeBlocks=r.runtimeBlocks+1; end
            fprintf(fid,'%s\t%s\t%s\t%s\t%s\t%s\n',esc(b),esc(bt),esc(mt),esc(rb),esc(ls),esc(tok));
        end
        fclose(fid);
        r.score=20*r.powergui+2*r.machines+0.001*r.blocks;
        try set_param(name,'SimulationCommand','update'); r.updateOk=true; catch ME, r.updateError=flaterr(ME); end
        out=fullfile(upDir,[safe(name) '_R2022b.slx']);
        try save_system(name,out); r.saveUpgradeOk=true; r.upgradedFile=out; catch ME, r.upgradedUpdateError=flaterr(ME); end
        close_system(name,0);
        if r.saveUpgradeOk
            [~,nn,~]=fileparts(out);
            try
                load_system(out); try set_param(nn,'SimulationCommand','update'); r.upgradedUpdateOk=true; catch ME, r.upgradedUpdateError=flaterr(ME); end
                if r.upgradedUpdateOk
                    try sim(nn,'StopTime','0.02','ReturnWorkspaceOutputs','on'); r.smokeOk=true; catch ME, r.smokeError=flaterr(ME); end
                end
                close_system(nn,0);
            catch ME, r.upgradedUpdateError=flaterr(ME); try close_system(nn,0); catch, end; end
        end
    catch ME
        r.updateError=flaterr(ME); try close_system(name,0); catch, end
    end
    if isempty(report.models), report.models=r; else report.models(end+1)=r; end
end

scores=[report.models.score]; [~,ord]=sort(scores,'descend'); report.ranking=ord;
report.nativeSmokeIndices=find([report.models.smokeOk]); report.anyNativeSmoke=~isempty(report.nativeSmokeIndices);
write_report(report,repDir);
end

function v=g(b,p), try v=get_param(b,p); if isnumeric(v),v=num2str(v);end, catch,v='';end, end
function s=safe(s), s=regexprep(s,'[^A-Za-z0-9_-]','_'); end
function s=esc(s), if isnumeric(s),s=num2str(s);end, s=strrep(char(s),sprintf('\t'),' '); s=strrep(s,sprintf('\n'),' '); s=strrep(s,sprintf('\r'),' '); end
function t=runtime_token(s)
s=lower(s); t=''; P={'artemis','artemis';'rt-lab','rt-lab';'rtlab','rt-lab';'opal-rt','opal-rt';'opcomm','opcomm';'opwrite','opwrite';'opmonitor','opmonitor';'optrigger','optrigger';'oprecorder','oprecorder';'opwait','opwait'};
for k=1:size(P,1), if contains(s,P{k,1}),t=P{k,2};return;end,end
end
function s=flaterr(ME), s=ME.message; try for k=1:numel(ME.stack),s=sprintf('%s | %s:%d',s,ME.stack(k).name,ME.stack(k).line);end,catch,end, s=esc(s); end
function write_report(report,d)
save(fullfile(d,'probe_report.mat'),'report'); fid=fopen(fullfile(d,'probe_report.json'),'w'); fwrite(fid,jsonencode(report,'PrettyPrint',true),'char'); fclose(fid);
fid=fopen(fullfile(d,'model_summary.tsv'),'w'); fprintf(fid,'idx\tname\tblocks\tpowergui\tmachines\truntime\tupdate\tupgrade_update\tsmoke\tupdate_error\tupgrade_error\tsmoke_error\n');
for i=1:numel(report.models),r=report.models(i);fprintf(fid,'%d\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\n',i,esc(r.name),r.blocks,r.powergui,r.machines,r.runtimeBlocks,r.updateOk,r.upgradedUpdateOk,r.smokeOk,esc(r.updateError),esc(r.upgradedUpdateError),esc(r.smokeError));end
fclose(fid);
end
