function ieee39_migrate_r2024a_ci
% CI-only strict migration entry used while resolving R2024a numerical fidelity.
% Source physical metadata comes from scripts/ieee39_source_provenance.py;
% this function never asks unavailable ARTEMIS/RT-LAB libraries for metadata.

repoRoot=pwd;
srcZip=fullfile(repoRoot,'Figures','IEEE-39-bus-power.zip');
provPath=fullfile(repoRoot,'build','ieee39_source_provenance.json');
outRoot=fullfile(repoRoot,'build','ieee39_r2024a');
srcRoot=fullfile(outRoot,'source'); migRoot=fullfile(outRoot,'migrated');
assert(exist(srcZip,'file')==2,'IEEE39 archive not found.');
assert(exist(provPath,'file')==2,'Preparsed source provenance not found.');
prov=jsondecode(fileread(provPath));
assert(numel(prov.generators)==10 && numel(prov.opcomm)==8,'Invalid source provenance contract.');
if exist(outRoot,'dir'), rmdir(outRoot,'s'); end
mkdir(srcRoot); mkdir(migRoot); unzip(srcZip,srcRoot); addpath(genpath(srcRoot));

report=struct('matlab_version',version,'release',version('-release'), ...
 'source_zip','Figures/IEEE-39-bus-power.zip', ...
 'source_git_blob','41db586d592851c4a81205a4cc5c7c770b7a0c48', ...
 'source_provenance_schema',prov.schema,'source_mdl_sha256',prov.source_mdl_sha256, ...
 'source_generator_assets',prov.generators,'source_opcomm_inventory',prov.opcomm, ...
 'upstream','DESL-EPFL/IEEE-39-bus-power-system:model.zip', ...
 'timestamp_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z''')), ...
 'native_step_s',25e-6,'opcomm_surrogate','zero_order_hold_Ts_diagnostic', ...
 'runnable',false,'real_simulation_completed',false);

files=[dir(fullfile(srcRoot,'**','*.slx'));dir(fullfile(srcRoot,'**','*.mdl'))];
assert(~isempty(files),'No Simulink model found.'); [~,ix]=max([files.bytes]); f=files(ix);
primaryPath=fullfile(f.folder,f.name); [~,mdl,~]=fileparts(primaryPath);
report.primary_source_model=strrep(primaryPath,[repoRoot filesep],'');
lineRecords=parse_artemis_line_records(primaryPath); report.source_line_parameter_records=numel(lineRecords);
load_system(primaryPath);
set_param(mdl,'InitFcn','Ts=25e-6; IEEE39BusLineLength; vNomHV=345E3; fNom=50; load(''Dynamicload.mat'');');

safe_delete([mdl '/ARTEMIS Guide']); safe_delete([mdl '/Model Initialization']);
safe_delete([mdl '/SC_Console']); safe_delete([mdl '/SM_measurement/Recording']);

% Keep the embedded dynamics of unresolved legacy powerlib/powerlib_extras
% subsystems, but detach stale library metadata.
allb=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block'); detached={};
for k=1:numel(allb)
    b=allb{k};
    try
        bt=get_param(b,'BlockType'); ls=get_param(b,'LinkStatus'); anc=get_param(b,'AncestorBlock');
        if strcmp(bt,'SubSystem') && any(strcmpi(ls,{'inactive','unresolved'})) && ~isempty(anc) && ...
                (startsWith(anc,'powerlib') || startsWith(anc,'powerlib_extras'))
            set_param(b,'LinkStatus','none'); detached{end+1}=b; %#ok<AGROW>
        end
    catch, end
end
report.detached_embedded_legacy_subsystems=detached;

dplTemplate=find_sps_library_block('Distributed Parameters Line');
report.sps_distributed_line_template=dplTemplate;
fprintf('Using native SPS Distributed Parameters Line template: %s\n',dplTemplate);
artLines=find_artemis_lines(mdl); report.artemis_line_count=numel(artLines);
assert(numel(artLines)==numel(lineRecords),'Source line records do not match unresolved ARTEMIS lines.');
lineAudit=cell(1,numel(artLines));
for k=1:numel(artLines)
    old=artLines{k}; parent=get_param(old,'Parent'); name=get_param(old,'Name'); vals=line_record_by_name(lineRecords,name);
    pos=get_param(old,'Position'); orient=get_param(old,'Orientation');
    replace_block(parent,'Name',name,dplTemplate,'noprompt'); nb=[parent '/' name];
    set_param(nb,'Position',pos,'Orientation',orient);
    set_if_present(nb,'Resistance',vals.Resistance); set_if_present(nb,'Inductance',vals.Inductance);
    set_if_present(nb,'Capacitance',vals.Capacitance); set_if_present(nb,'Length',vals.Length);
    set_if_present(nb,'Frequency',vals.Frequency); set_if_present(nb,'Phases',vals.Phases);
    set_if_present(nb,'Measurements',vals.Measurements);
    lineAudit{k}=struct('block',nb,'Resistance',vals.Resistance,'Inductance',vals.Inductance, ...
        'Capacitance',vals.Capacitance,'Length',vals.Length,'template',dplTemplate);
end
report.line_replacements=lineAudit;
report.remaining_artemis_lines_after_replacement=numel(find_artemis_lines(mdl));
assert(report.remaining_artemis_lines_after_replacement==0,'ARTEMIS lines remain after replacement.');

opcomms=find_rt_opcomm(mdl); report.opcomm_count=numel(opcomms); opAudit=cell(1,numel(opcomms));
for k=numel(opcomms):-1:1
    opAudit{k}=replace_opcomm_zoh(opcomms{k},prov.opcomm);
end
report.opcomm_replacements=opAudit;
report.remaining_rt_opcomm_after_replacement=numel(find_rt_opcomm(mdl));
assert(report.remaining_rt_opcomm_after_replacement==0,'RT-LAB OpComm remains after migration.');

pg=[mdl '/powergui']; if getSimulinkBlockHandle(pg)>0, try, set_param(pg,'SimulationMode','Discrete','SampleTime','Ts'); catch, end, end
migratedPath=fullfile(migRoot,[mdl '_R2024a.slx']); save_system(mdl,migratedPath); close_system(mdl,0);
report.migrated_model=strrep(migratedPath,[repoRoot filesep],'');
[~,mm,~]=fileparts(migratedPath); load_system(migratedPath);
try
    set_param(mm,'SimulationCommand','update'); report.primary_update_ok=true;
catch ME
    report.primary_update_ok=false; report.primary_update_error=full_error(ME);
end
if report.primary_update_ok
    oldStop=get_param(mm,'StopTime');
    try
        set_param(mm,'StopTime','0.02'); sim(mm,'ReturnWorkspaceOutputs','on');
        report.real_simulation_completed=true; report.runnable=true;
    catch ME
        report.short_sim_error=full_error(ME);
    end
    try, set_param(mm,'StopTime',oldStop); catch, end
end
report.remaining_unresolved={};
try
    b=find_system(mm,'LookUnderMasks','all','FollowLinks','on','Type','Block');
    for k=1:numel(b)
        try, ls=get_param(b{k},'LinkStatus'); if any(strcmpi(ls,{'unresolved','inactive'})), report.remaining_unresolved{end+1}=b{k}; end, catch, end %#ok<AGROW>
    end
catch, end
close_system(mm,0); write_report(fullfile(outRoot,'migration_report.json'),report);
fprintf('IEEE39 CI migrated: lines=%d opcomm=%d mode=%s update=%d runnable=%d real_sim=%d\n', ...
 report.artemis_line_count,report.opcomm_count,report.opcomm_surrogate,logical_field(report,'primary_update_ok'),report.runnable,report.real_simulation_completed);
if ~report.runnable
    error('AEFC:IEEE39MigrationBlocked','R2024a diagnostic migration did not complete a real short simulation.');
end
end

function p=find_sps_library_block(blockName)
canonical=['sps_lib/Power Grid Elements/' blockName];
try, load_system('sps_lib'); get_param(canonical,'Handle'); p=canonical; return; catch ME, warning('AEFC:SPSCanonicalPath','Canonical SPS path unavailable: %s',ME.message); end
libs={'sps_lib','powerlib','ee_lib'}; hits={};
for i=1:numel(libs)
    try, load_system(libs{i}); h=find_system(libs{i},'LookUnderMasks','all','FollowLinks','on','Type','Block','Name',blockName); hits=[hits; h(:)]; catch ME, warning('AEFC:SPSLibrarySearch','Could not search %s: %s',libs{i},ME.message); end %#ok<AGROW>
end
hits=unique(hits,'stable'); assert(~isempty(hits),'Could not locate native SPS block %s.',blockName);
depth=cellfun(@(s)numel(strfind(s,'/')),hits); lens=cellfun(@numel,hits); [~,ix]=sortrows([depth(:) lens(:)],[1 2]); p=hits{ix(1)};
end

function records=parse_artemis_line_records(path)
txt=fileread(path); parts=regexp(txt,'(?m)^\s*Block \{','split');
records=struct('Name',{},'Resistance',{},'Inductance',{},'Capacitance',{},'Length',{},'Frequency',{},'Phases',{},'Measurements',{});
for i=1:numel(parts)
    p=parts{i};
    if contains(p,'SourceBlock') && contains(p,'op_dpl_lib/Distributed Parameters Line')
        r.Name=qvalue(p,'Name'); r.Resistance=qvalue(p,'Resistance'); r.Inductance=qvalue(p,'Inductance');
        r.Capacitance=qvalue(p,'Capacitance'); r.Length=qvalue(p,'Length'); r.Frequency=qvalue(p,'Frequency');
        r.Phases=qvalue(p,'Phases'); r.Measurements=qvalue(p,'Measurements'); records(end+1)=r; %#ok<AGROW>
    end
end
end
function v=qvalue(txt,key), t=regexp(txt,['(?m)^\s*' regexptranslate('escape',key) '\s+"([^"]*)"'],'tokens','once'); assert(~isempty(t),['Missing MDL parameter ' key]); v=t{1}; end
function r=line_record_by_name(records,name), idx=find(strcmp({records.Name},name)); assert(numel(idx)==1,['Line record not unique: ' name]); r=records(idx); end
function set_if_present(b,p,v), try, set_param(b,p,v); catch ME, warning('AEFC:ParamMigration','Could not set %s on %s: %s',p,b,ME.message); end, end
function x=find_artemis_lines(mdl)
b=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block'); x={};
for i=1:numel(b), try, if strcmp(get_param(b{i},'BlockType'),'Reference') && strcmp(get_param(b{i},'SourceBlock'),'op_dpl_lib/Distributed Parameters Line'), x{end+1}=b{i}; end, catch, end, end %#ok<AGROW>
end
function x=find_rt_opcomm(mdl)
b=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block'); x={};
for i=1:numel(b), try, if strcmp(get_param(b{i},'BlockType'),'Reference') && strcmp(get_param(b{i},'SourceBlock'),'rtlab/OpComm'), x{end+1}=b{i}; end, catch, end, end %#ok<AGROW>
end

function audit=replace_opcomm_zoh(blk,sourceInventory)
parent=get_param(blk,'Parent'); name=get_param(blk,'Name'); full=[parent '/' name]; pos=get_param(blk,'Position'); orient=get_param(blk,'Orientation'); ph=get_param(blk,'PortHandles');
nin=numel(ph.Inport); nout=numel(ph.Outport); assert(nin==nout,'OpComm port mismatch');
sourceSt=''; sourceRate='';
for q=1:numel(sourceInventory)
    if strcmp(sourceInventory(q).path,full), sourceSt=sourceInventory(q).st; sourceRate=sourceInventory(q).subsys_rate; break; end
end
src=cell(1,nin); dst=cell(1,nout);
for i=1:nin, ln=get_param(ph.Inport(i),'Line'); if ln~=-1, src{i}=get_param(ln,'SrcPortHandle'); else, src{i}=[]; end, end
for i=1:nout, ln=get_param(ph.Outport(i),'Line'); if ln~=-1, dst{i}=get_param(ln,'DstPortHandle'); else, dst{i}=[]; end, end
delete_block(blk); nb=full; add_block('built-in/Subsystem',nb,'Position',pos,'Orientation',orient);
for i=1:nin
    y=30+45*(i-1);
    add_block('built-in/Inport',[nb '/In' num2str(i)],'Port',num2str(i),'Position',[25 y 55 y+14]);
    add_block('simulink/Discrete/Zero-Order Hold',[nb '/ZOH' num2str(i)],'SampleTime','Ts','Position',[90 y-3 130 y+17]);
    add_block('built-in/Outport',[nb '/Out' num2str(i)],'Port',num2str(i),'Position',[165 y 195 y+14]);
    add_line(nb,['In' num2str(i) '/1'],['ZOH' num2str(i) '/1']); add_line(nb,['ZOH' num2str(i) '/1'],['Out' num2str(i) '/1']);
end
nph=get_param(nb,'PortHandles');
for i=1:nin
    if ~isempty(src{i}), try, add_line(parent,src{i},nph.Inport(i),'autorouting','on'); catch, end, end
    if ~isempty(dst{i}), for j=1:numel(dst{i}), try, add_line(parent,nph.Outport(i),dst{i}(j),'autorouting','on'); catch, end, end, end
end
audit=struct('block',nb,'source_st',sourceSt,'source_subsys_rate',sourceRate,'ports',nin,'surrogate','zero_order_hold','sample_time','Ts');
end

function v=logical_field(s,f), if isfield(s,f), v=logical(s.(f)); else, v=false; end, end
function safe_delete(p), if getSimulinkBlockHandle(p)>0, delete_block(p); end, end
function s=full_error(ME), s=ME.message; try, for i=1:numel(ME.cause), s=[s ' | cause: ' full_error(ME.cause{i})]; end, catch, end, end %#ok<AGROW>
function write_report(p,r), fid=fopen(p,'w'); assert(fid>0); fwrite(fid,jsonencode(r,'PrettyPrint',true)); fclose(fid); end
