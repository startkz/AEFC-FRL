function report = aefc_r2022b_migrate_manifest(archivePath,outDir,manifestPath)
% R2022b migration using a parameter manifest extracted from the first REAL
% GitHub-hosted R2022b save of the byte-identical EPFL model. This avoids
% querying mask parameters through unresolved ARTEMIS library links.
if nargin<1,archivePath='external/IEEE39/model.zip';end
if nargin<2,outDir='build/ieee39_r2022b_full';end
if nargin<3,manifestPath='configs/ieee39_r2022b_artemis_line_manifest.json';end
if exist(outDir,'dir'),rmdir(outDir,'s');end
mkdir(outDir);src=fullfile(outDir,'source');mkdir(src);unzip(archivePath,src);addpath(genpath(src));
M=jsondecode(fileread(manifestPath));
expected='41db586d592851c4a81205a4cc5c7c770b7a0c48';
if ~strcmp(M.model_git_blob_sha,expected)||numel(M.lines)~=34,error('AEFC:ManifestProvenance','Invalid line manifest provenance/count');end
report=struct('release',version('-release'),'version',version,'archive',archivePath,'manifest',manifestPath, ...
 'manifest_model_sha',M.model_git_blob_sha,'manifest_source_run',M.source_action_run_id, ...
 'initScripts',{{}},'initErrors',{{}},'standardLineSource',M.standard_target,'lineRecords',{{}}, ...
 'runtimeRecords',{{}},'remainingVendorReferences',{{}},'updateOk',false,'updateError','', ...
 'smokeOk',false,'smokeError','','smokeWallSeconds',NaN,'outputModel','');
S=dir(fullfile(src,'**','*.m'));
for i=1:numel(S),f=fullfile(S(i).folder,S(i).name);report.initScripts{end+1}=f;try,evalin('base',sprintf('run(''%s'')',strrep(f,'''','''''')));catch ME,report.initErrors{end+1}=struct('file',f,'error',getReport(ME,'extended','hyperlinks','off'));end,end
if ~isempty(report.initErrors),finish(report,outDir);error('AEFC:InitFailed','Original model initialization failed');end
F=[dir(fullfile(src,'**','*.mdl'));dir(fullfile(src,'**','*.slx'))];if isempty(F),error('AEFC:NoModel','No model found');end
modelFile=fullfile(F(1).folder,F(1).name);[~,model,~]=fileparts(modelFile);load_system(modelFile);
outFile=fullfile(outDir,'IEEE39bus_R2022b_offline.slx');save_system(model,outFile);close_system(model,0);[~,mdl,~]=fileparts(outFile);load_system(outFile);report.outputModel=outFile;
load_system('sps_lib');if ~existsb(M.standard_target),error('AEFC:NoStandardDPL','Missing target block %s',M.standard_target);end
% Verify 34 legacy physical lines before replacement.
B=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block');nlegacy=0;for i=1:numel(B),if strcmp(sourceof(B{i}),'op_dpl_lib/Distributed Parameters Line'),nlegacy=nlegacy+1;end,end
report.detectedArtemisLineCount=nlegacy;if nlegacy~=34,finish(report,outDir);error('AEFC:LineCount','Expected 34 legacy lines, found %d',nlegacy);end
replace_block(mdl,'SourceBlock','op_dpl_lib/Distributed Parameters Line',M.standard_target,'noprompt');
for i=1:numel(M.lines)
 d=M.lines(i);b=[mdl '/' char(d.relative_path)];rec=struct('path',b,'ok',false,'lconn',0,'rconn',0,'error','', ...
  'Frequency',char(d.Frequency),'Resistance',char(d.Resistance),'Inductance',char(d.Inductance), ...
  'Capacitance',char(d.Capacitance),'Length',char(d.Length));
 try
  setdlg(b,'Frequency',char(d.Frequency));setdlg(b,'Resistance',char(d.Resistance));setdlg(b,'Inductance',char(d.Inductance));setdlg(b,'Capacitance',char(d.Capacitance));setdlg(b,'Length',char(d.Length));setdlg_optional(b,'Measurements',char(d.Measurements));
  ph=get_param(b,'PortHandles');rec.lconn=nport(ph,'LConn');rec.rconn=nport(ph,'RConn');rec.ok=(rec.lconn==3&&rec.rconn==3);if ~rec.ok,error('Expected 3+3 conserving ports, got %d+%d',rec.lconn,rec.rconn);end
 catch ME,rec.error=getReport(ME,'extended','hyperlinks','off');end
 report.lineRecords{end+1}=rec;
end
if any(cellfun(@(x)~x.ok,report.lineRecords)),save_system(mdl,outFile);close_system(mdl,0);finish(report,outDir);error('AEFC:LineMigration','One or more physical lines failed migration');end
% Neutralize only nonphysical vendor runtime interfaces.
B=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block');[~,ix]=sort(cellfun(@(x)sum(x=='/'),B),'descend');B=B(ix);
for i=1:numel(B)
 b=B{i};if ~existsb(b),continue;end;s=sourceof(b);
 if strcmp(s,'artemis/ARTEMIS/ARTEMIS Guide')||contains(lower([b ' ' p(b,'MaskType')]),'artemis guide')
  ph=get_param(b,'PortHandles');if totalports(ph)~=0,error('AEFC:GuidePorts','ARTEMIS Guide unexpectedly has ports');end;old=b;delete_block(b);report.runtimeRecords{end+1}=struct('path',old,'source',s,'strategy','delete_zero_port_solver_config','in',0,'out',0);
 elseif startsWith(s,'rtlab/')
  ph=get_param(b,'PortHandles');if nport(ph,'LConn')>0||nport(ph,'RConn')>0,error('AEFC:PhysicalRTLAB','RT-LAB block has conserving ports: %s',b);end
  ni=nport(ph,'Inport');no=nport(ph,'Outport');old=b;st=signal_adapter(b,ni,no);report.runtimeRecords{end+1}=struct('path',old,'source',s,'strategy',st,'in',ni,'out',no);
 end
end
B=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block');for i=1:numel(B),s=lower(sourceof(B{i}));if contains(s,'op_dpl_lib')||startsWith(s,'rtlab/')||startsWith(s,'artemis/'),report.remainingVendorReferences{end+1}=sprintf('%s => %s',B{i},s);end,end
if ~isempty(report.remainingVendorReferences),save_system(mdl,outFile);close_system(mdl,0);finish(report,outDir);error('AEFC:VendorRemain','Vendor references remain');end
try,set_param(mdl,'SimulationMode','normal');catch,end;try,set_param(mdl,'SystemTargetFile','grt.tlc');catch,end;save_system(mdl,outFile);
try,set_param(mdl,'SimulationCommand','update');report.updateOk=true;catch ME,report.updateError=getReport(ME,'extended','hyperlinks','off');end
if report.updateOk
 try,t=tic;sim(mdl,'StopTime','0.02','ReturnWorkspaceOutputs','on');report.smokeWallSeconds=toc(t);report.smokeOk=true;catch ME,report.smokeError=getReport(ME,'extended','hyperlinks','off');end
end
save_system(mdl,outFile);close_system(mdl,0);finish(report,outDir);if ~report.smokeOk,error('AEFC:SmokeFailed','Migrated real IEEE39 failed 0.02 s R2022b smoke');end
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
