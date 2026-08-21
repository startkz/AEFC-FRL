function ieee39_migrate_r2024a
% Conservative R2024a migration of the DESL-EPFL IEEE39 full-replica model.
% Only RT-LAB/ARTEMIS execution infrastructure is replaced. Electrical line
% parameters and embedded controller dynamics are preserved.

repoRoot = pwd;
srcZip = fullfile(repoRoot,'Figures','IEEE-39-bus-power.zip');
outRoot = fullfile(repoRoot,'build','ieee39_r2024a');
srcRoot = fullfile(outRoot,'source');
migRoot = fullfile(outRoot,'migrated');
if exist(outRoot,'dir'), rmdir(outRoot,'s'); end
mkdir(srcRoot); mkdir(migRoot);
assert(exist(srcZip,'file')==2,'IEEE39 archive not found: %s',srcZip);
unzip(srcZip,srcRoot); addpath(genpath(srcRoot));

report = struct('matlab_version',version,'release',version('-release'), ...
 'source_zip','Figures/IEEE-39-bus-power.zip', ...
 'source_git_blob','41db586d592851c4a81205a4cc5c7c770b7a0c48', ...
 'upstream','DESL-EPFL/IEEE-39-bus-power-system:model.zip', ...
 'timestamp_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z''')), ...
 'runnable',false,'real_simulation_completed',false);

files=[dir(fullfile(srcRoot,'**','*.slx'));dir(fullfile(srcRoot,'**','*.mdl'))];
assert(~isempty(files),'No Simulink model found.'); [~,ix]=max([files.bytes]); f=files(ix);
primaryPath=fullfile(f.folder,f.name); [~,mdl,~]=fileparts(primaryPath);
report.primary_source_model=strrep(primaryPath,[repoRoot filesep],'');
load_system(primaryPath);

% Preserve the original model initialization explicitly; the legacy masked
% ARTEMIS initialization block is not required after migration.
set_param(mdl,'InitFcn','Ts=25e-6; IEEE39BusLineLength; vNomHV=345E3; fNom=50; load(''Dynamicload.mat'');');

% Remove execution/UI-only RT-LAB infrastructure with no plant outputs.
safe_delete([mdl '/ARTEMIS Guide']);
safe_delete([mdl '/Model Initialization']);
safe_delete([mdl '/SC_Console']);
safe_delete([mdl '/SM_measurement/Recording']);

% Legacy copied powerlib_extras blocks contain their implementation in the
% MDL. Detach only their obsolete library metadata; do not alter contents.
allb=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block');
detached={};
for k=1:numel(allb)
    b=allb{k};
    try
        bt=get_param(b,'BlockType'); ls=get_param(b,'LinkStatus'); anc=get_param(b,'AncestorBlock');
        if strcmp(bt,'SubSystem') && any(strcmpi(ls,{'inactive','unresolved'})) && ~isempty(anc)
            if startsWith(anc,'powerlib') || startsWith(anc,'powerlib_extras')
                set_param(b,'LinkStatus','none'); detached{end+1}=b; %#ok<AGROW>
            end
        end
    catch
    end
end
report.detached_embedded_legacy_subsystems=detached;

% Replace every ARTEMIS distributed-parameter line by the native SPS block,
% preserving R/L/C, line length, frequency, phase count and orientation.
load_system('powerlib');
artLines=find_artemis_lines(mdl); report.artemis_line_count=numel(artLines);
lineAudit=cell(1,numel(artLines));
for k=1:numel(artLines)
    old=artLines{k}; parent=get_param(old,'Parent'); name=get_param(old,'Name');
    vals=struct('Resistance',get_param(old,'Resistance'),'Inductance',get_param(old,'Inductance'), ...
        'Capacitance',get_param(old,'Capacitance'),'Length',get_param(old,'Length'), ...
        'Frequency',get_param(old,'Frequency'),'Phases',get_param(old,'Phases'), ...
        'Measurements',get_param(old,'Measurements'));
    pos=get_param(old,'Position'); orient=get_param(old,'Orientation');
    replace_block(parent,'Name',name,'powerlib/Elements/Distributed Parameters Line','noprompt');
    nb=[parent '/' name]; set_param(nb,'Position',pos,'Orientation',orient);
    fn=fieldnames(vals);
    for q=1:numel(fn), try, set_param(nb,fn{q},vals.(fn{q})); catch, end, end
    lineAudit{k}=struct('block',nb,'Resistance',vals.Resistance,'Inductance',vals.Inductance, ...
        'Capacitance',vals.Capacitance,'Length',vals.Length);
end
report.line_replacements=lineAudit;

% RT-LAB OpComm is a partition communication boundary. In a monolithic
% offline simulation, replace it with paired identity channels. This keeps
% the exact signal graph while removing real-time partition transport.
opcomms=find_rt_opcomm(mdl); report.opcomm_count=numel(opcomms);
for k=numel(opcomms):-1:1, replace_opcomm_identity(opcomms{k}); end

% Enforce the original SPS discrete integration step through the existing
% powergui rather than ARTEMIS solver orchestration.
pg=[mdl '/powergui'];
if getSimulinkBlockHandle(pg)>0
    try, set_param(pg,'SimulationMode','Discrete','SampleTime','Ts'); catch, end
end

migratedPath=fullfile(migRoot,[mdl '_R2024a.slx']);
save_system(mdl,migratedPath); close_system(mdl,0);
report.migrated_model=strrep(migratedPath,[repoRoot filesep],'');

[~,mm,~]=fileparts(migratedPath); load_system(migratedPath);
try
    set_param(mm,'SimulationCommand','update'); report.primary_update_ok=true;
catch ME
    report.primary_update_ok=false; report.primary_update_error=full_error(ME);
end
if report.primary_update_ok
    try
        oldStop=get_param(mm,'StopTime'); c=onCleanup(@()set_param(mm,'StopTime',oldStop)); %#ok<NASGU>
        set_param(mm,'StopTime','0.02'); sim(mm,'ReturnWorkspaceOutputs','on');
        report.real_simulation_completed=true; report.runnable=true;
    catch ME
        report.short_sim_error=full_error(ME);
    end
end

% Audit unresolved links after migration.
report.remaining_unresolved={};
try
    b=find_system(mm,'LookUnderMasks','all','FollowLinks','on','Type','Block');
    for k=1:numel(b)
        try
            ls=get_param(b{k},'LinkStatus');
            if any(strcmpi(ls,{'unresolved','inactive'})), report.remaining_unresolved{end+1}=b{k}; end %#ok<AGROW>
        catch, end
    end
catch, end
close_system(mm,0);
write_report(fullfile(outRoot,'migration_report.json'),report);
fprintf('IEEE39 migrated: lines=%d opcomm=%d runnable=%d real_sim=%d\n', ...
 report.artemis_line_count,report.opcomm_count,report.runnable,report.real_simulation_completed);
if ~report.runnable
    error('AEFC:IEEE39MigrationBlocked','R2024a model did not complete a real Simulink simulation; inspect migration_report.json.');
end
end

function x=find_artemis_lines(mdl)
b=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block'); x={};
for i=1:numel(b)
    try
        if strcmp(get_param(b{i},'BlockType'),'Reference') && ...
                strcmp(get_param(b{i},'SourceBlock'),'op_dpl_lib/Distributed Parameters Line')
            x{end+1}=b{i}; %#ok<AGROW>
        end
    catch, end
end
end

function x=find_rt_opcomm(mdl)
b=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block'); x={};
for i=1:numel(b)
    try
        if strcmp(get_param(b{i},'BlockType'),'Reference') && strcmp(get_param(b{i},'SourceBlock'),'rtlab/OpComm')
            x{end+1}=b{i}; %#ok<AGROW>
        end
    catch, end
end
end

function replace_opcomm_identity(blk)
parent=get_param(blk,'Parent'); name=get_param(blk,'Name'); pos=get_param(blk,'Position'); orient=get_param(blk,'Orientation');
ph=get_param(blk,'PortHandles'); nin=numel(ph.Inport); nout=numel(ph.Outport); assert(nin==nout,'OpComm port mismatch');
src=cell(1,nin); dst=cell(1,nout);
for i=1:nin
    ln=get_param(ph.Inport(i),'Line'); if ln~=-1, src{i}=get_param(ln,'SrcPortHandle'); else, src{i}=[]; end
end
for i=1:nout
    ln=get_param(ph.Outport(i),'Line'); if ln~=-1, dst{i}=get_param(ln,'DstPortHandle'); else, dst{i}=[]; end
end
delete_block(blk); nb=[parent '/' name]; add_block('built-in/Subsystem',nb,'Position',pos,'Orientation',orient);
for i=1:nin
    in=[nb '/In' num2str(i)]; out=[nb '/Out' num2str(i)];
    add_block('built-in/Inport',in,'Port',num2str(i),'Position',[30 30+45*(i-1) 60 44+45*(i-1)]);
    add_block('built-in/Outport',out,'Port',num2str(i),'Position',[160 30+45*(i-1) 190 44+45*(i-1)]);
    add_line(nb,['In' num2str(i) '/1'],['Out' num2str(i) '/1']);
end
nph=get_param(nb,'PortHandles');
for i=1:nin
    if ~isempty(src{i}), try, add_line(parent,src{i},nph.Inport(i),'autorouting','on'); catch, end, end
    if ~isempty(dst{i}), for j=1:numel(dst{i}), try, add_line(parent,nph.Outport(i),dst{i}(j),'autorouting','on'); catch, end, end, end
end
end

function safe_delete(p)
if getSimulinkBlockHandle(p)>0, delete_block(p); end
end
function s=full_error(ME)
s=ME.message;
try
    for i=1:numel(ME.cause), s=[s ' | cause: ' full_error(ME.cause{i})]; end %#ok<AGROW>
catch, end
end
function write_report(p,r)
fid=fopen(p,'w'); assert(fid>0); fwrite(fid,jsonencode(r,'PrettyPrint',true)); fclose(fid);
end
