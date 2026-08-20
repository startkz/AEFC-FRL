function report = aefc_r2022b_diagnose(archivePath,outDir)
%AEFC_R2022B_DIAGNOSE Produce compact, exact migration evidence.
% The report is intentionally small enough to commit back from Actions.

if nargin<1, archivePath='external/IEEE39/model.zip'; end
if nargin<2, outDir='build/ieee39_r2022b_diagnose'; end
if exist(outDir,'dir'), rmdir(outDir,'s'); end
mkdir(outDir); src=fullfile(outDir,'source'); mkdir(src); unzip(archivePath,src); addpath(genpath(src));
M=dir(fullfile(src,'**','*.mdl')); if isempty(M), M=dir(fullfile(src,'**','*.slx')); end
if isempty(M), error('AEFC:NoModel','No model found'); end
modelFile=fullfile(M(1).folder,M(1).name); [~,model,~]=fileparts(modelFile);
report=struct(); report.release=version('-release'); report.model=model; report.modelFile=modelFile;
report.runtimeBlocks=struct('path',{},'blockType',{},'maskType',{},'referenceBlock',{},'linkStatus',{},'token',{},'ports',{});
report.updateOk=false; report.updateException=''; report.updateCauses={};
load_system(modelFile);
B=find_system(model,'LookUnderMasks','all','FollowLinks','on','Type','Block');
for i=1:numel(B)
    b=B{i}; bt=g(b,'BlockType'); mt=g(b,'MaskType'); rb=g(b,'ReferenceBlock'); ls=g(b,'LinkStatus');
    tok=runtime_token([b ' ' bt ' ' mt ' ' rb]);
    if ~isempty(tok)
        p=struct('in',0,'out',0,'enable',0,'trigger',0,'ifaction',0,'state',0,'lconn',0,'rconn',0);
        try
            ph=get_param(b,'PortHandles');
            F=fieldnames(p);
            for k=1:numel(F), if isfield(ph,capfield(F{k})),p.(F{k})=numel(ph.(capfield(F{k}))); end, end
        catch
        end
        r=struct('path',b,'blockType',bt,'maskType',mt,'referenceBlock',rb,'linkStatus',ls,'token',tok,'ports',p);
        report.runtimeBlocks(end+1)=r; %#ok<AGROW>
    end
end
try
    set_param(model,'SimulationCommand','update'); report.updateOk=true;
catch ME
    report.updateException=getReport(ME,'extended','hyperlinks','off');
    report.updateCauses=collect_causes(ME);
end
report.solver=struct('Solver',g(model,'Solver'),'SolverType',g(model,'SolverType'),'FixedStep',g(model,'FixedStep'),'SimulationMode',g(model,'SimulationMode'));
report.powerguiPaths=find_system(model,'LookUnderMasks','all','FollowLinks','on','Regexp','on','Name','(?i)^powergui$');
close_system(model,0);
fid=fopen(fullfile(outDir,'diagnose_report.json'),'w'); fwrite(fid,jsonencode(report,'PrettyPrint',true),'char'); fclose(fid);
fid=fopen(fullfile(outDir,'runtime_blocks.tsv'),'w'); fprintf(fid,'path\tblock_type\tmask_type\treference_block\tlink_status\ttoken\n');
for i=1:numel(report.runtimeBlocks),r=report.runtimeBlocks(i);fprintf(fid,'%s\t%s\t%s\t%s\t%s\t%s\n',esc(r.path),esc(r.blockType),esc(r.maskType),esc(r.referenceBlock),esc(r.linkStatus),esc(r.token));end
fclose(fid);
end

function c=collect_causes(ME)
c={}; Q={ME}; seen=0;
while ~isempty(Q) && seen<50
    x=Q{1}; Q(1)=[]; seen=seen+1;
    for j=1:numel(x.cause)
        q=x.cause{j}; c{end+1}=getReport(q,'extended','hyperlinks','off'); %#ok<AGROW>
        Q{end+1}=q; %#ok<AGROW>
    end
end
end
function f=capfield(s), m=struct('in','Inport','out','Outport','enable','Enable','trigger','Trigger','ifaction','Ifaction','state','State','lconn','LConn','rconn','RConn'); f=m.(s); end
function v=g(b,p), try v=get_param(b,p); if isnumeric(v),v=num2str(v);end, catch,v='';end, end
function t=runtime_token(s),s=lower(s);t='';P={'artemis','artemis';'rt-lab','rt-lab';'rtlab','rt-lab';'opal-rt','opal-rt';'opal','opal';'opcomm','opcomm';'opwrite','opwrite';'opmonitor','opmonitor';'optrigger','optrigger';'oprecorder','oprecorder';'opwait','opwait';'opfrom','opfrom';'opgoto','opgoto'};for k=1:size(P,1),if contains(s,P{k,1}),t=P{k,2};return;end,end,end
function s=esc(s),if isnumeric(s),s=num2str(s);end,s=strrep(char(s),sprintf('\t'),' ');s=strrep(s,sprintf('\n'),' ');s=strrep(s,sprintf('\r'),' ');end
