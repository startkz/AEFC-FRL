function report = aefc_r2022b_migrate_runtime(archivePath,outDir)
%AEFC_R2022B_MIGRATE_RUNTIME Conservative migration of runtime-only blocks.
% It preserves all physical Specialized Power Systems components. Only blocks
% positively identified as ARTEMIS/RT-LAB/OPAL-RT runtime interfaces are
% changed. Blocks with physical conserving ports are refused, not bypassed.

if nargin<1, archivePath='external/IEEE39/model.zip'; end
if nargin<2, outDir='build/ieee39_r2022b_migrated'; end
if exist(outDir,'dir'), rmdir(outDir,'s'); end
mkdir(outDir); src=fullfile(outDir,'source'); mkdir(src); unzip(archivePath,src); addpath(genpath(src));
M=dir(fullfile(src,'**','*.mdl')); if isempty(M),M=dir(fullfile(src,'**','*.slx'));end
if isempty(M),error('AEFC:NoModel','No model found');end
modelFile=fullfile(M(1).folder,M(1).name);[~,model,~]=fileparts(modelFile);
load_system(modelFile);
outFile=fullfile(outDir,'IEEE39bus_R2022b_offline.slx'); save_system(model,outFile); close_system(model,0);
[~,newModel,~]=fileparts(outFile); load_system(outFile);

report=struct();report.release=version('-release');report.source=modelFile;report.output=outFile;
report.replacements=struct('path',{},'token',{},'strategy',{},'inports',{},'outports',{},'lconn',{},'rconn',{});
report.refused=struct('path',{},'token',{},'reason',{});report.updateOk=false;report.updateError='';report.smokeOk=false;report.smokeError='';
B=find_system(newModel,'LookUnderMasks','all','FollowLinks','on','Type','Block');
% Deepest paths first, because replacing a parent invalidates descendants.
depth=cellfun(@(x)sum(x=='/'),B);[~,ord]=sort(depth,'descend');B=B(ord);
for i=1:numel(B)
    b=B{i}; if ~exists_block(b),continue;end
    bt=g(b,'BlockType');mt=g(b,'MaskType');rb=g(b,'ReferenceBlock');tok=runtime_token([b ' ' bt ' ' mt ' ' rb]);
    if isempty(tok),continue;end
    if is_physical_component([b ' ' mt ' ' rb])
        report.refused(end+1)=struct('path',b,'token',tok,'reason','runtime token overlaps physical component; manual review required'); %#ok<AGROW>
        continue
    end
    ph=get_param(b,'PortHandles');nin=countp(ph,'Inport');nout=countp(ph,'Outport');nl=countp(ph,'LConn');nr=countp(ph,'RConn');
    if nl>0 || nr>0
        report.refused(end+1)=struct('path',b,'token',tok,'reason','physical conserving ports present; automatic bypass forbidden'); %#ok<AGROW>
        continue
    end
    try
        strategy=replace_signal_runtime_block(b,nin,nout);
        report.replacements(end+1)=struct('path',b,'token',tok,'strategy',strategy,'inports',nin,'outports',nout,'lconn',nl,'rconn',nr); %#ok<AGROW>
    catch ME
        report.refused(end+1)=struct('path',b,'token',tok,'reason',getReport(ME,'basic','hyperlinks','off')); %#ok<AGROW>
    end
end

% RT-LAB/ARTEMIS target configuration must not be used for offline CI.
try set_param(newModel,'SimulationMode','normal'); catch,end
try set_param(newModel,'SystemTargetFile','grt.tlc'); catch,end
% Prefer the solver already encoded by the physical SPS model unless its name
% explicitly points to a real-time vendor target. In that case use ode23tb;
% powergui remains responsible for SPS network discretization settings.
sv=lower(g(newModel,'Solver'));
if contains(sv,'opal')||contains(sv,'artemis')||contains(sv,'rtlab')||contains(sv,'rt-lab')
    try set_param(newModel,'SolverType','Variable-step','Solver','ode23tb'); catch,end
end
save_system(newModel,outFile);
if ~isempty(report.refused)
    report.updateError=sprintf('%d runtime blocks were refused because safe automatic replacement was not possible.',numel(report.refused));
else
    try set_param(newModel,'SimulationCommand','update');report.updateOk=true;catch ME,report.updateError=getReport(ME,'extended','hyperlinks','off');end
    if report.updateOk
        try sim(newModel,'StopTime','0.02','ReturnWorkspaceOutputs','on');report.smokeOk=true;catch ME,report.smokeError=getReport(ME,'extended','hyperlinks','off');end
    end
end
close_system(newModel,0);
fid=fopen(fullfile(outDir,'migration_report.json'),'w');fwrite(fid,jsonencode(report,'PrettyPrint',true),'char');fclose(fid);
if ~report.smokeOk
    error('AEFC:R2022bMigrationFailed','R2022b physical smoke failed; inspect migration_report.json');
end
end

function n=countp(ph,f),if isfield(ph,f),n=numel(ph.(f));else,n=0;end,end
function tf=exists_block(b),try get_param(b,'Handle');tf=true;catch,tf=false;end,end
function v=g(b,p),try v=get_param(b,p);if isnumeric(v),v=num2str(v);end,catch,v='';end,end
function t=runtime_token(s),s=lower(s);t='';P={'artemis','artemis';'rt-lab','rt-lab';'rtlab','rt-lab';'opal-rt','opal-rt';'opcomm','opcomm';'opwrite','opwrite';'opmonitor','opmonitor';'optrigger','optrigger';'oprecorder','oprecorder';'opwait','opwait';'opfrom','opfrom';'opgoto','opgoto'};for k=1:size(P,1),if contains(s,P{k,1}),t=P{k,2};return;end,end,end
function tf=is_physical_component(s),s=lower(s);P={'synchronous machine','transmission line','transformer','three-phase load','dynamic load','breaker','voltage measurement','current measurement','powergui','voltage source','current source'};tf=false;for k=1:numel(P),if contains(s,P{k}),tf=true;return;end,end,end
function strategy=replace_signal_runtime_block(b,nin,nout)
parent=get_param(b,'Parent');name=get_param(b,'Name');pos=get_param(b,'Position');ph=get_param(b,'PortHandles');
src=cell(1,nin);dst=cell(1,nout);
for k=1:nin,src{k}=[];ln=get_param(ph.Inport(k),'Line');if ln~=-1,src{k}=get_param(ln,'SrcPortHandle');end,end
for k=1:nout,dst{k}=[];ln=get_param(ph.Outport(k),'Line');if ln~=-1,dst{k}=get_param(ln,'DstPortHandle');end,end
for k=1:nin,ln=get_param(ph.Inport(k),'Line');if ln~=-1,try delete_line(ln);catch,end,end,end
for k=1:nout,ln=get_param(ph.Outport(k),'Line');if ln~=-1,try delete_line(ln);catch,end,end,end
delete_block(b);
if nin==0 && nout==0,strategy='delete_unconnected_runtime_block';return;end
p=[parent '/' name];add_block('simulink/Ports & Subsystems/Subsystem',p,'Position',pos);
try delete_line(p,'In1/1','Out1/1');catch,end;try delete_block([p '/In1']);catch,end;try delete_block([p '/Out1']);catch,end
for k=1:nin,add_block('simulink/Ports & Subsystems/In1',[p sprintf('/In%d',k)],'Port',num2str(k));end
for k=1:nout
 add_block('simulink/Ports & Subsystems/Out1',[p sprintf('/Out%d',k)],'Port',num2str(k));
 if k<=nin,add_line(p,sprintf('In%d/1',k),sprintf('Out%d/1',k),'autorouting','on');else,z=[p sprintf('/Zero%d',k)];add_block('simulink/Sources/Constant',z,'Value','0');add_line(p,sprintf('Zero%d/1',k),sprintf('Out%d/1',k),'autorouting','on');end
end
for k=nout+1:nin,t=[p sprintf('/Term%d',k)];add_block('simulink/Sinks/Terminator',t);add_line(p,sprintf('In%d/1',k),sprintf('Term%d/1',k),'autorouting','on');end
nph=get_param(p,'PortHandles');
for k=1:nin,if ~isempty(src{k})&&ishandle(src{k}),try add_line(parent,src{k},nph.Inport(k),'autorouting','on');catch,end,end,end
for k=1:nout
 d=dst{k};if isempty(d),continue;end;if ~iscell(d),d=num2cell(d);end
 for j=1:numel(d),if ~isempty(d{j})&&ishandle(d{j}),try add_line(parent,nph.Outport(k),d{j},'autorouting','on');catch,end,end,end
end
if nin==nout,strategy='signal_passthrough';elseif nin>nout,strategy='passthrough_plus_terminators';else,strategy='passthrough_plus_zero_sources';end
end
