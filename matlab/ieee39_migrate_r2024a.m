function ieee39_migrate_r2024a
% Conservative R2024a migration of the DESL-EPFL IEEE39 full-replica model.
% Electrical line parameters and embedded controller dynamics are preserved.
% RT-LAB OpComm is represented by a base-rate sample/hold boundary rather
% than a pure wire so that the migrated offline model retains communication
% update semantics at the model's native Ts=25 us step.

repoRoot=pwd;
srcZip=fullfile(repoRoot,'Figures','IEEE-39-bus-power.zip');
outRoot=fullfile(repoRoot,'build','ieee39_r2024a');
srcRoot=fullfile(outRoot,'source');
migRoot=fullfile(outRoot,'migrated');
if exist(outRoot,'dir'), rmdir(outRoot,'s'); end
mkdir(srcRoot); mkdir(migRoot);
assert(exist(srcZip,'file')==2,'IEEE39 archive not found: %s',srcZip);
unzip(srcZip,srcRoot); addpath(genpath(srcRoot));

report=struct('matlab_version',version,'release',version('-release'), ...
 'source_zip','Figures/IEEE-39-bus-power.zip', ...
 'source_git_blob','41db586d592851c4a81205a4cc5c7c770b7a0c48', ...
 'upstream','DESL-EPFL/IEEE-39-bus-power-system:model.zip', ...
 'timestamp_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z''')), ...
 'native_step_s',25e-6,'opcomm_surrogate','zero_order_hold_Ts', ...
 'runnable',false,'real_simulation_completed',false);

files=[dir(fullfile(srcRoot,'**','*.slx'));dir(fullfile(srcRoot,'**','*.mdl'))];
assert(~isempty(files),'No Simulink model found.'); [~,ix]=max([files.bytes]); f=files(ix);
primaryPath=fullfile(f.folder,f.name); [~,mdl,~]=fileparts(primaryPath);
report.primary_source_model=strrep(primaryPath,[repoRoot filesep],'');
lineRecords=parse_artemis_line_records(primaryPath);
report.source_line_parameter_records=numel(lineRecords);
load_system(primaryPath);
set_param(mdl,'InitFcn','Ts=25e-6; IEEE39BusLineLength; vNomHV=345E3; fNom=50; load(''Dynamicload.mat'');');

% Capture source-model facts while the legacy references are still readable.
report.source_generator_assets=capture_generator_assets(mdl);
report.source_opcomm_inventory=capture_opcomm_inventory(mdl);

safe_delete([mdl '/ARTEMIS Guide']);
safe_delete([mdl '/Model Initialization']);
safe_delete([mdl '/SC_Console']);
safe_delete([mdl '/SM_measurement/Recording']);

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
assert(numel(artLines)==numel(lineRecords),'Source line record count does not match unresolved line count.');
lineAudit=cell(1,numel(artLines));
for k=1:numel(artLines)
    old=artLines{k}; parent=get_param(old,'Parent'); name=get_param(old,'Name'); vals=line_record_by_name(lineRecords,name);
    pos=get_param(old,'Position'); orient=get_param(old,'Orientation');
    replace_block(parent,'Name',name,dplTemplate,'noprompt');
    nb=[parent '/' name]; set_param(nb,'Position',pos,'Orientation',orient);
    set_if_present(nb,'Resistance',vals.Resistance); set_if_present(nb,'Inductance',vals.Inductance);
    set_if_present(nb,'Capacitance',vals.Capacitance); set_if_present(nb,'Length',vals.Length);
    set_if_present(nb,'Frequency',vals.Frequency); set_if_present(nb,'Phases',vals.Phases); set_if_present(nb,'Measurements',vals.Measurements);
    lineAudit{k}=struct('block',nb,'Resistance',vals.Resistance,'Inductance',vals.Inductance, ...
        'Capacitance',vals.Capacitance,'Length',vals.Length,'template',dplTemplate);
end
report.line_replacements=lineAudit;
report.remaining_artemis_lines_after_replacement=numel(find_artemis_lines(mdl));
assert(report.remaining_artemis_lines_after_replacement==0,'ARTEMIS distributed-line blocks remain after migration.');

opcomms=find_rt_opcomm(mdl); report.opcomm_count=numel(opcomms);
opAudit=cell(1,numel(opcomms));
for k=numel(opcomms):-1:1
    opAudit{k}=replace_opcomm_zoh(opcomms{k});
end
report.opcomm_replacements=opAudit;
report.remaining_rt_opcomm_after_replacement=numel(find_rt_opcomm(mdl));
assert(report.remaining_rt_opcomm_after_replacement==0,'RT-LAB OpComm blocks remain after migration.');

pg=[mdl '/powergui'];
if getSimulinkBlockHandle(pg)>0, try, set_param(pg,'SimulationMode','Discrete','SampleTime','Ts'); catch, end, end
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
fprintf('IEEE39 migrated: lines=%d opcomm=%d mode=%s update=%d runnable=%d real_sim=%d\n', ...
 report.artemis_line_count,report.opcomm_count,report.opcomm_surrogate,logical_field(report,'primary_update_ok'),report.runnable,report.real_simulation_completed);
if ~report.runnable, error('AEFC:IEEE39MigrationBlocked','R2024a model did not complete a real Simulink simulation; inspect migration_report.json.'); end
end

function a=capture_generator_assets(mdl)
template=struct('generator',0,'machine_path','','machine_source','','nominal_parameters','', ...
    'mechanical_parameters','','initial_conditions','','loadflow_path','','vbase_expr','','vbase_volts',NaN,'vlf_expr','','vlf_pu',NaN);
a=repmat(template,1,10);
blocks=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block');
for g=1:10, a(g).generator=g; end
for i=1:numel(blocks)
    b=blocks{i}; g=generator_from_path(b); if ~isfinite(g) || g<1 || g>10, continue; end
    st=''; sb=''; try, st=get_param(b,'SourceType'); catch, end; try, sb=get_param(b,'SourceBlock'); catch, end
    if contains(lower([st ' ' sb]),'synchronous machine')
        a(g).machine_path=b; a(g).machine_source=sb;
        try, a(g).nominal_parameters=get_param(b,'NominalParameters'); catch, end
        try, a(g).mechanical_parameters=get_param(b,'Mechanical'); catch, end
        try, a(g).initial_conditions=get_param(b,'InitialConditions'); catch, end
    elseif contains(lower([st ' ' sb]),'load flow bus') && isempty(a(g).loadflow_path)
        a(g).loadflow_path=b;
        try, a(g).vbase_expr=get_param(b,'Vbase'); a(g).vbase_volts=eval_numeric(a(g).vbase_expr); catch, end
        try, a(g).vlf_expr=get_param(b,'VLF'); a(g).vlf_pu=eval_numeric(a(g).vlf_expr); catch, end
    end
end
assert(all(arrayfun(@(x)~isempty(x.machine_path),a)),'Could not map all ten source synchronous machines.');
assert(all(arrayfun(@(x)isfinite(x.vbase_volts) && x.vbase_volts>0,a)),'Could not map all ten source generator voltage bases.');
end

function x=capture_opcomm_inventory(mdl)
b=find_rt_opcomm(mdl); x=cell(1,numel(b));
for i=1:numel(b)
    z=struct('path',b{i},'st','','subsys_rate','','nbport','','synchronization','','interpolation','');
    for p={'st','subsys_rate','nbport','Synchronization','Interpolation'}
        try, z.(lower(p{1}))=get_param(b{i},p{1}); catch, end
    end
    x{i}=z;
end
end

function p=find_sps_library_block(blockName)
canonical=['sps_lib/Power Grid Elements/' blockName];
try, load_system('sps_lib'); get_param(canonical,'Handle'); p=canonical; return; catch ME, warning('AEFC:SPSCanonicalPath','Canonical SPS path unavailable: %s',ME.message); end
libs={'sps_lib','powerlib','ee_lib'}; hits={};
for i=1:numel(libs)
    try, load_system(libs{i}); h=find_system(libs{i},'LookUnderMasks','all','FollowLinks','on','Type','Block','Name',blockName); hits=[hits; h(:)]; catch ME, warning('AEFC:SPSLibrarySearch','Could not search %s: %s',libs{i},ME.message); end %#ok<AGROW>
end
hits=unique(hits,'stable'); assert(~isempty(hits),'Could not locate native SPS block named "%s".',blockName);
depth=cellfun(@(s)numel(strfind(s,'/')),hits); lens=cellfun(@numel,hits); [~,ix]=sortrows([depth(:) lens(:)],[1 2]); p=hits{ix(1)};
end

function records=parse_artemis_line_records(path)
txt=fileread(path); parts=regexp(txt,'(?m)^\s*Block \{','split'); records=struct('Name',{},'Resistance',{},'Inductance',{},'Capacitance',{},'Length',{},'Frequency',{},'Phases',{},'Measurements',{});
for i=1:numel(parts)
    p=parts{i};
    if contains(p,'SourceBlock') && contains(p,'op_dpl_lib/Distributed Parameters Line')
        r.Name=qvalue(p,'Name'); r.Resistance=qvalue(p,'Resistance'); r.Inductance=qvalue(p,'Inductance'); r.Capacitance=qvalue(p,'Capacitance'); r.Length=qvalue(p,'Length'); r.Frequency=qvalue(p,'Frequency'); r.Phases=qvalue(p,'Phases'); r.Measurements=qvalue(p,'Measurements'); records(end+1)=r; %#ok<AGROW>
    end
end
end
function v=qvalue(txt,key), t=regexp(txt,['(?m)^\s*' regexptranslate('escape',key) '\s+"([^"]*)"'],'tokens','once'); assert(~isempty(t),['Missing MDL parameter ' key]); v=t{1}; end
function r=line_record_by_name(records,name), idx=find(strcmp({records.Name},name)); assert(numel(idx)==1,['Line parameter record is not unique: ' name]); r=records(idx); end
function set_if_present(b,p,v), try, set_param(b,p,v); catch ME, warning('AEFC:ParamMigration','Could not set %s on %s: %s',p,b,ME.message); end, end
function x=find_artemis_lines(mdl)
b=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block'); x={}; for i=1:numel(b), try, if strcmp(get_param(b{i},'BlockType'),'Reference') && strcmp(get_param(b{i},'SourceBlock'),'op_dpl_lib/Distributed Parameters Line'), x{end+1}=b{i}; end, catch, end, end %#ok<AGROW>
end
function x=find_rt_opcomm(mdl)
b=find_system(mdl,'LookUnderMasks','all','FollowLinks','on','Type','Block'); x={}; for i=1:numel(b), try, if strcmp(get_param(b{i},'BlockType'),'Reference') && strcmp(get_param(b{i},'SourceBlock'),'rtlab/OpComm'), x{end+1}=b{i}; end, catch, end, end %#ok<AGROW>
end

function audit=replace_opcomm_zoh(blk)
parent=get_param(blk,'Parent'); name=get_param(blk,'Name'); pos=get_param(blk,'Position'); orient=get_param(blk,'Orientation'); ph=get_param(blk,'PortHandles'); nin=numel(ph.Inport); nout=numel(ph.Outport); assert(nin==nout,'OpComm port mismatch');
st=''; rate=''; try, st=get_param(blk,'st'); catch, end; try, rate=get_param(blk,'subsys_rate'); catch, end
src=cell(1,nin); dst=cell(1,nout);
for i=1:nin, ln=get_param(ph.Inport(i),'Line'); if ln~=-1, src{i}=get_param(ln,'SrcPortHandle'); else, src{i}=[]; end, end
for i=1:nout, ln=get_param(ph.Outport(i),'Line'); if ln~=-1, dst{i}=get_param(ln,'DstPortHandle'); else, dst{i}=[]; end, end
delete_block(blk); nb=[parent '/' name]; add_block('built-in/Subsystem',nb,'Position',pos,'Orientation',orient);
for i=1:nin
    y=30+45*(i-1); in=[nb '/In' num2str(i)]; zh=[nb '/ZOH' num2str(i)]; out=[nb '/Out' num2str(i)];
    add_block('built-in/Inport',in,'Port',num2str(i),'Position',[25 y 55 y+14]);
    add_block('simulink/Discrete/Zero-Order Hold',zh,'SampleTime','Ts','Position',[90 y-3 130 y+17]);
    add_block('built-in/Outport',out,'Port',num2str(i),'Position',[165 y 195 y+14]);
    add_line(nb,['In' num2str(i) '/1'],['ZOH' num2str(i) '/1']); add_line(nb,['ZOH' num2str(i) '/1'],['Out' num2str(i) '/1']);
end
nph=get_param(nb,'PortHandles');
for i=1:nin
    if ~isempty(src{i}), try, add_line(parent,src{i},nph.Inport(i),'autorouting','on'); catch, end, end
    if ~isempty(dst{i}), for j=1:numel(dst{i}), try, add_line(parent,nph.Outport(i),dst{i}(j),'autorouting','on'); catch, end, end, end
end
audit=struct('block',nb,'source_st',st,'source_subsys_rate',rate,'ports',nin,'surrogate','zero_order_hold','sample_time','Ts');
end

function g=generator_from_path(p), t=regexp(p,'/GT\s*(\d+)','tokens','once'); if isempty(t), g=NaN; else, g=str2double(t{1}); end, end
function v=eval_numeric(expr), v=str2double(expr); if ~isfinite(v), try, v=evalin('base',expr); catch, v=NaN; end, end; if ~isscalar(v), v=NaN; end, end
function v=logical_field(s,f), if isfield(s,f), v=logical(s.(f)); else, v=false; end, end
function safe_delete(p), if getSimulinkBlockHandle(p)>0, delete_block(p); end, end
function s=full_error(ME), s=ME.message; try, for i=1:numel(ME.cause), s=[s ' | cause: ' full_error(ME.cause{i})]; end, catch, end, end %#ok<AGROW>
function write_report(p,r), fid=fopen(p,'w'); assert(fid>0); fwrite(fid,jsonencode(r,'PrettyPrint',true)); fclose(fid); end
