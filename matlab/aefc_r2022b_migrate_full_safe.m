function report = aefc_r2022b_migrate_full_safe(archivePath,outDir)
% Conservative, auditable R2022b migration of the exact EPFL IEEE39 model.
if nargin<1,archivePath='external/IEEE39/model.zip';end
if nargin<2,outDir='build/ieee39_r2022b_full';end
if exist(outDir,'dir'),rmdir(outDir,'s');end
mkdir(outDir);src=fullfile(outDir,'source');mkdir(src);unzip(archivePath,src);addpath(genpath(src));
report=struct('release',version('-release'),'version',version,'archive',archivePath, ...
 'initScripts',{{}},'initErrors',{{}},'standardLineSource','','lineRecords',{{}}, ...
 'runtimeRecords',{{}},'remainingVendorReferences',{{}},'updateOk',false, ...
 'updateError','','smokeOk',false,'smokeError','','outputModel','');

% Execute every original package script; the package contains
% IEEE39BusLineLength.m, which defines TL_*_len variables used by all 34 lines.
S=dir(fullfile(src,'**','*.m'));
for i=1:numel(S)
 f=fullfile(S(i).folder,S(i).name);report.initScripts{end+1}=f;
 try,evalin('base',sprintf('run(''%s'')',strrep(f,'''','''''')));
 catch ME,report.initErrors{end+1}=struct('file',f,'error',getReport(ME,'extended','hyperlinks','off'));end
end
if ~isempty(report.initErrors),finish(report,outDir);error('AEFC:InitFailed','Original model initialization failed');end

M=[dir(fullfile(src,'**','*.mdl'));dir(fullfile(src,'**','*.slx'))];
if isempty(M),error('AEFC:NoModel','No model found');end
modelFile=fullfile(M(1).folder,M(1).name);[~,model,~]=fileparts(modelFile);load_system(modelFile);
outFile=fullfile(outDir,'IEEE39bus_R2022b_offline.slx');save_system(model,outFile);close_system(model,0);
[~,mdl,~]=fileparts(outFile);load_system(outFile);report.outputModel=outFile;

std=discover_dpl();report.standardLineSource=std;
B=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block');L={};D={};
for i=1:numel(B)
 b=B{i};if strcmp(sourceof(b),'op_dpl_lib/Distributed Parameters Line')
  d=struct('path',b,'Frequency',p(b,'Frequency'),'Resistance',p(b,'Resistance'), ...
   'Inductance',p(b,'Inductance'),'Capacitance',p(b,'Capacitance'), ...
   'Length',p(b,'Length'),'Measurements',p(b,'Measurements'));
  L{end+1}=b;D{end+1}=d;
 end
end
report.detectedArtemisLineCount=numel(L);
if numel(L)~=34,save_system(mdl,outFile);close_system(mdl,0);finish(report,outDir);error('AEFC:LineCount','Expected 34 ARTEMIS lines, found %d',numel(L));end

% replace_block preserves the original six electrical connections.
replace_block(mdl,'SourceBlock','op_dpl_lib/Distributed Parameters Line',std,'noprompt');
for i=1:numel(D)
 d=D{i};rec=struct('path',d.path,'ok',false,'lconn',0,'rconn',0,'error','', ...
  'Frequency',d.Frequency,'Resistance',d.Resistance,'Inductance',d.Inductance, ...
  'Capacitance',d.Capacitance,'Length',d.Length);
 try
  setdlg(d.path,'Frequency',d.Frequency);setdlg(d.path,'Resistance',d.Resistance);
  setdlg(d.path,'Inductance',d.Inductance);setdlg(d.path,'Capacitance',d.Capacitance);
  setdlg(d.path,'Length',d.Length);if ~isempty(d.Measurements),setdlg_optional(d.path,'Measurements',d.Measurements);end
  ph=get_param(d.path,'PortHandles');rec.lconn=nport(ph,'LConn');rec.rconn=nport(ph,'RConn');
  rec.ok=(rec.lconn==3 && rec.rconn==3);if ~rec.ok,error('Expected 3+3 conserving ports');end
 catch ME,rec.error=getReport(ME,'extended','hyperlinks','off');end
 report.lineRecords{end+1}=rec;
end
if any(cellfun(@(x)~x.ok,report.lineRecords)),save_system(mdl,outFile);close_system(mdl,0);finish(report,outDir);error('AEFC:LineMigration','Physical-line migration failed');end

% Runtime-only blocks: no physical element may be removed here.
B=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block');[~,ix]=sort(cellfun(@(x)sum(x=='/'),B),'descend');B=B(ix);
for i=1:numel(B)
 b=B{i};if ~existsb(b),continue;end;s=sourceof(b);
 if strcmp(s,'artemis/ARTEMIS/ARTEMIS Guide') || contains(lower([b ' ' p(b,'MaskType')]),'artemis guide')
  ph=get_param(b,'PortHandles');if totalports(ph)~=0,error('AEFC:GuidePorts','ARTEMIS Guide has ports');end
  old=b;delete_block(b);report.runtimeRecords{end+1}=struct('path',old,'source',s,'strategy','delete_zero_port_solver_config','in',0,'out',0);
 elseif startsWith(s,'rtlab/')
  ph=get_param(b,'PortHandles');if nport(ph,'LConn')||nport(ph,'RConn'),error('AEFC:PhysicalRTLAB','Vendor block has conserving ports: %s',b);end
  ni=nport(ph,'Inport');no=nport(ph,'Outport');old=b;strategy=signal_adapter(b,ni,no);
  report.runtimeRecords{end+1}=struct('path',old,'source',s,'strategy',strategy,'in',ni,'out',no);
 end
end

B=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block');
for i=1:numel(B),s=lower(sourceof(B{i}));if contains(s,'op_dpl_lib')||startsWith(s,'rtlab/')||startsWith(s,'artemis/'),report.remainingVendorReferences{end+1}=sprintf('%s => %s',B{i},s);end,end
if ~isempty(report.remainingVendorReferences),save_system(mdl,outFile);close_system(mdl,0);finish(report,outDir);error('AEFC:VendorRemain','Vendor references remain');end
try,set_param(mdl,'SimulationMode','normal');catch,end
try,set_param(mdl,'SystemTargetFile','grt.tlc');catch,end
save_system(mdl,outFile);
try,set_param(mdl,'SimulationCommand','update');report.updateOk=true;catch ME,report.updateError=getReport(ME,'extended','hyperlinks','off');end
if report.updateOk
 try,sim(mdl,'StopTime','0.02','ReturnWorkspaceOutputs','on');report.smokeOk=true;catch ME,report.smokeError=getReport(ME,'extended','hyperlinks','off');end
end
save_system(mdl,outFile);close_system(mdl,0);finish(report,outDir);
if ~report.smokeOk,error('AEFC:SmokeFailed','Migrated real IEEE39 failed 0.02 s R2022b smoke');end
end

function src=discover_dpl()
load_system('sps_lib');C=find_system('sps_lib','LookUnderMasks','all','FollowLinks','on','Regexp','on','Name','(?i)^Distributed Parameters Line$');
for i=1:numel(C)
 try,d=get_param(C{i},'DialogParameters');f=lower(string(fieldnames(d)));need=["frequency","resistance","inductance","capacitance","length"];
  ph=get_param(C{i},'PortHandles');if all(ismember(need,f))&&nport(ph,'LConn')==3&&nport(ph,'RConn')==3,src=C{i};return;end
 catch,end
end
error('AEFC:NoDPL','No R2022b SPS Distributed Parameters Line with required parameters/ports found');
end
function s=sourceof(b),s='';try,s=get_param(b,'SourceBlock');catch,end;if isempty(s),try,s=get_param(b,'ReferenceBlock');catch,end,end,end
function v=p(b,n),v='';try,v=get_param(b,n);catch,end,end
function setdlg(b,n,v),d=get_param(b,'DialogParameters');f=fieldnames(d);j=find(strcmpi(f,n),1);if isempty(j),error('Missing dialog parameter %s',n);end;set_param(b,f{j},v);end
function setdlg_optional(b,n,v),try,setdlg(b,n,v);catch,end,end
function n=nport(ph,f),if isfield(ph,f),n=numel(ph.(f));else,n=0;end,end
function n=totalports(ph),n=0;F=fieldnames(ph);for i=1:numel(F),n=n+numel(ph.(F{i}));end,end
function tf=existsb(b),try,get_param(b,'Handle');tf=true;catch,tf=false;end,end
function strategy=signal_adapter(b,ni,no)
parent=get_param(b,'Parent');name=get_param(b,'Name');pos=get_param(b,'Position');ph=get_param(b,'PortHandles');src=cell(1,ni);dst=cell(1,no);
for k=1:ni,src{k}=[];ln=get_param(ph.Inport(k),'Line');if ln~=-1,src{k}=get_param(ln,'SrcPortHandle');end,end
for k=1:no,dst{k}=[];ln=get_param(ph.Outport(k),'Line');if ln~=-1,dst{k}=get_param(ln,'DstPortHandle');end,end
for k=1:ni,ln=get_param(ph.Inport(k),'Line');if ln~=-1,try,delete_line(ln);catch,end,end,end
for k=1:no,ln=get_param(ph.Outport(k),'Line');if ln~=-1,try,delete_line(ln);catch,end,end,end
delete_block(b);if ni==0&&no==0,strategy='delete';return;end
q=[parent '/' name];add_block('simulink/Ports & Subsystems/Subsystem',q,'Position',pos);try,delete_line(q,'In1/1','Out1/1');catch,end;try,delete_block([q '/In1']);catch,end;try,delete_block([q '/Out1']);catch,end
for k=1:ni,add_block('simulink/Ports & Subsystems/In1',[q sprintf('/In%d',k)],'Port',num2str(k));end
for k=1:no,add_block('simulink/Ports & Subsystems/Out1',[q sprintf('/Out%d',k)],'Port',num2str(k));if k<=ni,add_line(q,sprintf('In%d/1',k),sprintf('Out%d/1',k),'autorouting','on');else,z=[q sprintf('/Zero%d',k)];add_block('simulink/Sources/Constant',z,'Value','0');add_line(q,sprintf('Zero%d/1',k),sprintf('Out%d/1',k),'autorouting','on');end,end
for k=no+1:ni,t=[q sprintf('/Term%d',k)];add_block('simulink/Sinks/Terminator',t);add_line(q,sprintf('In%d/1',k),sprintf('Term%d/1',k),'autorouting','on');end
nph=get_param(q,'PortHandles');for k=1:ni,if ~isempty(src{k})&&ishandle(src{k}),try,add_line(parent,src{k},nph.Inport(k),'autorouting','on');catch,end,end,end
for k=1:no,d=dst{k};if isempty(d),continue;end;if ~iscell(d),d=num2cell(d);end;for j=1:numel(d),if ~isempty(d{j})&&ishandle(d{j}),try,add_line(parent,nph.Outport(k),d{j},'autorouting','on');catch,end,end,end,end
if ni==no,strategy='passthrough';elseif ni>no,strategy='passthrough_terminate_extra_inputs';else,strategy='passthrough_zero_extra_outputs';end
end
function finish(r,d),fid=fopen(fullfile(d,'full_migration_report.json'),'w');fwrite(fid,jsonencode(r,'PrettyPrint',true),'char');fclose(fid);end
