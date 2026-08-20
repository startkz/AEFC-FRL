function report = aefc_r2022b_migrate_full(archivePath,outDir)
%AEFC_R2022B_MIGRATE_FULL Evidence-preserving R2015a->R2022b migration.
% 1) Executes all model-package initialization scripts in the base workspace.
% 2) Replaces only the 34 ARTEMIS Distributed Parameters Line blocks with the
%    standard R2022b Specialized Power Systems Distributed Parameters Line.
% 3) Removes/passes through RT-LAB-only communication/logging blocks while
%    preserving the physical power network.
% 4) Requires update-diagram and a real 0.02 s Simulink smoke simulation.

if nargin<1, archivePath='external/IEEE39/model.zip'; end
if nargin<2, outDir='build/ieee39_r2022b_full'; end
if exist(outDir,'dir'), rmdir(outDir,'s'); end
mkdir(outDir); srcDir=fullfile(outDir,'source'); mkdir(srcDir); unzip(archivePath,srcDir); addpath(genpath(srcDir));

report=struct(); report.release=version('-release'); report.version=version; report.archive=archivePath;
report.initScripts={}; report.initErrors={}; report.lineSource=''; report.lines=struct([]); report.runtime=struct([]);
report.updateOk=false; report.updateError=''; report.smokeOk=false; report.smokeError=''; report.outputModel='';

% This package has a single line-length script, but execute every bundled .m
% script so the migrated model uses the same parameter initialization as the
% original source rather than guessed numerical constants.
S=dir(fullfile(srcDir,'**','*.m'));
for i=1:numel(S)
    f=fullfile(S(i).folder,S(i).name); report.initScripts{end+1}=f;
    try
        evalin('base',sprintf('run(''%s'')',strrep(f,'''','''''')));
    catch ME
        report.initErrors(end+1,:)={f,getReport(ME,'extended','hyperlinks','off')};
    end
end
if ~isempty(report.initErrors)
    write_report(report,outDir);
    error('AEFC:InitFailed','One or more original IEEE39 initialization scripts failed under R2022b.');
end

M=[dir(fullfile(srcDir,'**','*.mdl'));dir(fullfile(srcDir,'**','*.slx'))];
if isempty(M),error('AEFC:NoModel','No Simulink model found');end
modelFile=fullfile(M(1).folder,M(1).name); [~,model,~]=fileparts(modelFile);
load_system(modelFile);
outFile=fullfile(outDir,'IEEE39bus_R2022b_offline.slx'); save_system(model,outFile); close_system(model,0);
[~,newModel,~]=fileparts(outFile); load_system(outFile); report.outputModel=outFile;

% Discover the current-release SPS implementation rather than hard-coding a
% release-specific library path.
stdLine=discover_standard_dpl(); report.lineSource=stdLine;

% Capture all original ARTEMIS line instance parameters before replacement.
B=find_system(newModel,'LookUnderMasks','all','FollowLinks','on','Type','Block');
linePaths={}; lineData=struct('path',{},'Frequency',{},'Resistance',{},'Inductance',{},'Capacitance',{},'Length',{},'Measurements',{},'Position',{},'Rotation',{},'Mirror',{});
for i=1:numel(B)
    b=B{i}; src=get_source_block(b);
    if strcmp(src,'op_dpl_lib/Distributed Parameters Line')
        r=struct('path',b,'Frequency',gp(b,'Frequency'),'Resistance',gp(b,'Resistance'), ...
            'Inductance',gp(b,'Inductance'),'Capacitance',gp(b,'Capacitance'), ...
            'Length',gp(b,'Length'),'Measurements',gp(b,'Measurements'), ...
            'Position',gp(b,'Position'),'Rotation',gp(b,'BlockRotation'),'Mirror',gp(b,'BlockMirror'));
        linePaths{end+1}=b; lineData(end+1)=r; %#ok<AGROW>
    end
end
if numel(linePaths)~=34
    report.detectedArtemisLines=numel(linePaths); write_report(report,outDir);
    error('AEFC:UnexpectedLineCount','Expected 34 ARTEMIS distributed lines, found %d.',numel(linePaths));
end

% replace_block preserves the existing conserving-port wiring. We then restore
% all physical line parameters from the original instance expressions.
replaced = replace_block(newModel,'SourceBlock','op_dpl_lib/Distributed Parameters Line',stdLine,'noprompt');
report.replaceBlockResult=replaced;
for i=1:numel(lineData)
    b=lineData(i).path;
    if ~exists_block(b)
        report.lines(end+1)=struct('path',b,'ok',false,'error','Path missing after replace_block'); %#ok<AGROW>
        continue
    end
    try
        set_if_present(b,'Frequency',lineData(i).Frequency);
        set_if_present(b,'Resistance',lineData(i).Resistance);
        set_if_present(b,'Inductance',lineData(i).Inductance);
        set_if_present(b,'Capacitance',lineData(i).Capacitance);
        set_if_present(b,'Length',lineData(i).Length);
        set_if_present(b,'Measurements',lineData(i).Measurements);
        % Keep diagram geometry; replacement generally preserves it, but restore
        % explicitly to make the transformation deterministic.
        if ~isempty(lineData(i).Position),set_param(b,'Position',lineData(i).Position);end
        if ~isempty(lineData(i).Rotation),try set_param(b,'BlockRotation',lineData(i).Rotation);catch,end,end
        if ~isempty(lineData(i).Mirror),try set_param(b,'BlockMirror',lineData(i).Mirror);catch,end,end
        ph=get_param(b,'PortHandles');
        nl=countp(ph,'LConn'); nr=countp(ph,'RConn');
        ok=(nl==3 && nr==3);
        report.lines(end+1)=struct('path',b,'ok',ok,'error','','lconn',nl,'rconn',nr, ...
            'Frequency',lineData(i).Frequency,'Resistance',lineData(i).Resistance, ...
            'Inductance',lineData(i).Inductance,'Capacitance',lineData(i).Capacitance,'Length',lineData(i).Length); %#ok<AGROW>
        if ~ok,error('AEFC:LinePorts','Replacement %s has %d LConn / %d RConn ports',b,nl,nr);end
    catch ME
        report.lines(end+1)=struct('path',b,'ok',false,'error',getReport(ME,'extended','hyperlinks','off')); %#ok<AGROW>
    end
end
if any(~[report.lines.ok])
    save_system(newModel,outFile); close_system(newModel,0); write_report(report,outDir);
    error('AEFC:LineMigrationFailed','At least one physical transmission-line replacement failed.');
end

% Remove ARTEMIS Guide: it is a zero-port solver/configuration block, not a
% physical network element. Abort if its port topology is unexpectedly nonzero.
B=find_system(newModel,'LookUnderMasks','all','FollowLinks','on','Type','Block');
for i=1:numel(B)
    b=B{i}; if ~exists_block(b),continue;end
    src=get_source_block(b);
    if strcmp(src,'artemis/ARTEMIS/ARTEMIS Guide') || contains(lower([b ' ' gp(b,'MaskType')]),'artemis guide')
        ph=get_param(b,'PortHandles'); n=count_all_ports(ph);
        if n~=0,error('AEFC:ArtemisGuidePorts','ARTEMIS Guide unexpectedly has %d ports',n);end
        delete_block(b); report.runtime(end+1)=struct('path',b,'source',src,'strategy','delete_zero_port_solver_config'); %#ok<AGROW>
    end
end

% Replace RT-LAB signal interfaces only. OpComm is a communication transport,
% so preserve its signal semantics as portwise pass-through. OpTrigger and
% OpWriteFile are data-logging blocks; their outputs are nonphysical status/
% trigger signals, so preserve dimensions with a pass-through/zero adapter.
B=find_system(newModel,'LookUnderMasks','all','FollowLinks','on','Type','Block');
[~,ord]=sort(cellfun(@(x)sum(x=='/'),B),'descend'); B=B(ord);
for i=1:numel(B)
    b=B{i}; if ~exists_block(b),continue;end
    src=get_source_block(b);
    if startsWith(src,'rtlab/')
        ph=get_param(b,'PortHandles');
        if countp(ph,'LConn')>0 || countp(ph,'RConn')>0
            error('AEFC:RTLABPhysicalPorts','Automatic replacement forbidden for %s with conserving ports.',b);
        end
        nin=countp(ph,'Inport');nout=countp(ph,'Outport');
        strategy=replace_signal_block(b,nin,nout);
        report.runtime(end+1)=struct('path',b,'source',src,'strategy',strategy,'inports',nin,'outports',nout); %#ok<AGROW>
    end
end

% Verify no vendor physical/runtime reference remains.
B=find_system(newModel,'LookUnderMasks','all','FollowLinks','on','Type','Block'); remain={};
for i=1:numel(B)
    src=get_source_block(B{i}); s=lower(src);
    if contains(s,'op_dpl_lib') || startsWith(s,'rtlab/') || startsWith(s,'artemis/')
        remain{end+1}=sprintf('%s => %s',B{i},src); %#ok<AGROW>
    end
end
report.remainingVendorReferences=remain;
if ~isempty(remain)
    save_system(newModel,outFile); close_system(newModel,0); write_report(report,outDir);
    error('AEFC:VendorReferenceRemain','Vendor references remain after migration.');
end

% Preserve the model's original electrical dynamics/solver unless the legacy
% target explicitly names a real-time vendor. Never replace power-network
% dynamics just to make CI pass.
try set_param(newModel,'SimulationMode','normal');catch,end
try set_param(newModel,'SystemTargetFile','grt.tlc');catch,end
save_system(newModel,outFile);
try
    set_param(newModel,'SimulationCommand','update'); report.updateOk=true;
catch ME
    report.updateError=getReport(ME,'extended','hyperlinks','off');
end
if report.updateOk
    try
        simOut=sim(newModel,'StopTime','0.02','ReturnWorkspaceOutputs','on'); %#ok<NASGU>
        report.smokeOk=true;
    catch ME
        report.smokeError=getReport(ME,'extended','hyperlinks','off');
    end
end
save_system(newModel,outFile); close_system(newModel,0); write_report(report,outDir);
if ~report.smokeOk
    error('AEFC:R2022bPhysicalSmokeFailed','Migrated IEEE39 did not complete real 0.02 s R2022b simulation.');
end
end

function src=discover_standard_dpl()
libs={'powerlib','sps_lib'}; candidates={};
for li=1:numel(libs)
    try load_system(libs{li}); catch,continue;end
    try
        C=find_system(libs{li},'LookUnderMasks','all','FollowLinks','on','Regexp','on','Name','(?i)^Distributed Parameters Line$');
        candidates=[candidates;C(:)]; %#ok<AGROW>
    catch
    end
end
for i=1:numel(candidates)
    b=candidates{i};
    try
        dp=get_param(b,'DialogParameters'); fn=fieldnames(dp); f=lower(string(fn));
        need={'frequency','resistance','inductance','capacitance','length'};
        if all(ismember(need,cellstr(f)))
            ph=get_param(b,'PortHandles');
            if countp(ph,'LConn')==3 && countp(ph,'RConn')==3
                src=b; return
            end
        end
    catch
    end
end
error('AEFC:NoStandardDPL','Could not discover a standard R2022b SPS Distributed Parameters Line block with 3+3 conserving ports.');
end

function src=get_source_block(b)
src=''; try src=get_param(b,'SourceBlock');catch,end
if isempty(src),try src=get_param(b,'ReferenceBlock');catch,end,end
end
function v=gp(b,p),v='';try v=get_param(b,p);catch,end,end
function set_if_present(b,p,v)
if isempty(v),return;end
try
    d=get_param(b,'DialogParameters');
    fn=fieldnames(d); ix=find(strcmpi(fn,p),1);
    if ~isempty(ix),set_param(b,fn{ix},v);else,error('Missing dialog parameter %s',p);end
catch ME
    rethrow(ME)
end
end
function n=countp(ph,f),if isfield(ph,f),n=numel(ph.(f));else,n=0;end,end
function n=count_all_ports(ph),n=0;F=fieldnames(ph);for i=1:numel(F),n=n+numel(ph.(F{i}));end,end
function tf=exists_block(b),try get_param(b,'Handle');tf=true;catch,tf=false;end,end
function strategy=replace_signal_block(b,nin,nout)
parent=get_param(b,'Parent');name=get_param(b,'Name');pos=get_param(b,'Position');ph=get_param(b,'PortHandles');
srcH=cell(1,nin);dstH=cell(1,nout);
for k=1:nin,srcH{k}=[];ln=get_param(ph.Inport(k),'Line');if ln~=-1,srcH{k}=get_param(ln,'SrcPortHandle');end,end
for k=1:nout,dstH{k}=[];ln=get_param(ph.Outport(k),'Line');if ln~=-1,dstH{k}=get_param(ln,'DstPortHandle');end,end
for k=1:nin,ln=get_param(ph.Inport(k),'Line');if ln~=-1,try delete_line(ln);catch,end,end,end
for k=1:nout,ln=get_param(ph.Outport(k),'Line');if ln~=-1,try delete_line(ln);catch,end,end,end
delete_block(b); if nin==0&&nout==0,strategy='delete';return;end
p=[parent '/' name];add_block('simulink/Ports & Subsystems/Subsystem',p,'Position',pos);
try delete_line(p,'In1/1','Out1/1');catch,end;try delete_block([p '/In1']);catch,end;try delete_block([p '/Out1']);catch,end
for k=1:nin,add_block('simulink/Ports & Subsystems/In1',[p sprintf('/In%d',k)],'Port',num2str(k));end
for k=1:nout
 add_block('simulink/Ports & Subsystems/Out1',[p sprintf('/Out%d',k)],'Port',num2str(k));
 if k<=nin,add_line(p,sprintf('In%d/1',k),sprintf('Out%d/1',k),'autorouting','on');else,z=[p sprintf('/Zero%d',k)];add_block('simulink/Sources/Constant',z,'Value','0');add_line(p,sprintf('Zero%d/1',k),sprintf('Out%d/1',k),'autorouting','on');end
end
for k=nout+1:nin,t=[p sprintf('/Term%d',k)];add_block('simulink/Sinks/Terminator',t);add_line(p,sprintf('In%d/1',k),sprintf('Term%d/1',k),'autorouting','on');end
nph=get_param(p,'PortHandles');
for k=1:nin,if ~isempty(srcH{k})&&ishandle(srcH{k}),try add_line(parent,srcH{k},nph.Inport(k),'autorouting','on');catch,end,end,end
for k=1:nout,d=dstH{k};if isempty(d),continue;end;if ~iscell(d),d=num2cell(d);end;for j=1:numel(d),if ~isempty(d{j})&&ishandle(d{j}),try add_line(parent,nph.Outport(k),d{j},'autorouting','on');catch,end,end,end,end
if nin==nout,strategy='portwise_passthrough';elseif nin>nout,strategy='passthrough_plus_terminators';else,strategy='passthrough_plus_zero_outputs';end
end
function write_report(r,d),fid=fopen(fullfile(d,'full_migration_report.json'),'w');fwrite(fid,jsonencode(r,'PrettyPrint',true),'char');fclose(fid);end
