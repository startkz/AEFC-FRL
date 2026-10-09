function aefc_ieee39_interface_discovery
% Discover stable signal-edge candidates in the native MathWorks
% IEEE39BusSystem example. The output uses block paths and port numbers,
% not transient line handles, so a later probe can validate the exact
% observation and actuation paths before a closed-loop experiment is run.
assert(~verLessThan('matlab','24.2'),'AEFC:ReleaseTooOld','R2024b required.');
[model,~]=aefc_ieee39_load_official;
repo=pwd;
resDir=fullfile(repo,'results','ieee39_r2024b');
if ~exist(resDir,'dir'), mkdir(resDir); end

blocks=find_system(model,'LookUnderMasks','all','FollowLinks','on','Type','Block');
keys={'Measurements','Generator','AVR','Exciter','Governor','Vref','Pref', ...
      'Rotor','Speed','Velocity','Voltage','Angle','Fault','Bus','Load'};

out=struct;
out.model=model;
out.release=version('-release');
out.mathworks_example=true;
out.generated_at=char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
out.candidates=struct;
for k=1:numel(keys)
    hit={};
    for i=1:numel(blocks)
        if contains(lower(blocks{i}),lower(keys{k}))
            hit{end+1}=blocks{i}; %#ok<AGROW>
        end
    end
    out.candidates.(matlab.lang.makeValidName(keys{k}))=hit;
end

obsEdges=emptyEdgeArray();
ctrlEdges=emptyEdgeArray();
allEdges=emptyEdgeArray();
for i=1:numel(blocks)
    src=blocks{i};
    try
        ph=get_param(src,'PortHandles');
    catch
        continue;
    end
    if ~isfield(ph,'Outport') || isempty(ph.Outport), continue; end
    for p=1:numel(ph.Outport)
        try, lh=get_param(ph.Outport(p),'Line'); catch, lh=-1; end
        if isempty(lh) || all(lh==-1), continue; end
        try, dstHandles=get_param(lh,'DstPortHandle'); catch, dstHandles=[]; end
        if isempty(dstHandles), continue; end
        dstHandles=dstHandles(dstHandles~=-1);
        for d=1:numel(dstHandles)
            try
                dst=get_param(dstHandles(d),'Parent');
                dstPort=double(get_param(dstHandles(d),'PortNumber'));
            catch
                continue;
            end
            [obsScore,obsReason]=scoreObservationEdge(src,dst);
            [ctrlScore,ctrlReason]=scoreControlEdge(src,dst);
            e=struct('src_block',src,'src_port',p,'dst_block',dst,'dst_port',dstPort, ...
                     'observation_score',obsScore,'control_score',ctrlScore, ...
                     'observation_reason',obsReason,'control_reason',ctrlReason);
            allEdges(end+1)=e; %#ok<AGROW>
            if obsScore>0, obsEdges(end+1)=e; end %#ok<AGROW>
            if ctrlScore>0, ctrlEdges(end+1)=e; end %#ok<AGROW>
        end
    end
end

out.edges=allEdges;
out.observation_edges=sortEdges(obsEdges,'observation_score');
out.control_edges=sortEdges(ctrlEdges,'control_score');
out.policy=struct( ...
    'observation_rule','rank speed/velocity/terminal-voltage/angle/measurement signal edges', ...
    'control_rule','rank Vref/Pref/governor/AVR/exciter/reference edges', ...
    'validation_rule','no edge is accepted as an actuator until finite-difference probing produces a nontrivial measured response');
out.note=['Discovery produces candidates only. experiments/matlab/aefc_ieee39_mapping_probe.m ' ...
          'must validate causal actuation and live observation logging before native Results are enabled.'];
writejson(fullfile(resDir,'interface_discovery.json'),out);
close_system(model,0);
end

function a=emptyEdgeArray()
a=struct('src_block',{},'src_port',{},'dst_block',{},'dst_port',{}, ...
         'observation_score',{},'control_score',{}, ...
         'observation_reason',{},'control_reason',{});
end

function a=sortEdges(a,field)
if isempty(a), return; end
v=arrayfun(@(x)x.(field),a);
[~,idx]=sort(v,'descend');
a=a(idx);
end

function [score,reason]=scoreObservationEdge(src,dst)
t=lower([src ' ' dst]);
score=0; reasons={};
terms={{'rotor velocity','rotor_velocity','speed','velocity'},5,'rotor-speed'}; %#ok<CCAT>
if hasAny(t,terms{1,1}), score=score+terms{1,2}; reasons{end+1}=terms{1,3}; end %#ok<AGROW>
if hasAny(t,{'terminal voltage','terminal_voltage','voltage'}), score=score+4; reasons{end+1}='terminal-voltage'; end %#ok<AGROW>
if hasAny(t,{'rotor electrical angle','rotor_electrical_angle','angle'}), score=score+3; reasons{end+1}='rotor-angle'; end %#ok<AGROW>
if hasAny(t,{'measurement','measurements','sensor'}), score=score+2; reasons{end+1}='measurement-path'; end %#ok<AGROW>
if hasAny(t,{'scope','display'}), score=score+1; reasons{end+1}='logged-output'; end %#ok<AGROW>
reason=strjoin(reasons,',');
end

function [score,reason]=scoreControlEdge(src,dst)
t=lower([src ' ' dst]);
score=0; reasons={};
if hasAny(t,{'vref','voltage reference','voltage_ref'}), score=score+7; reasons{end+1}='voltage-reference'; end %#ok<AGROW>
if hasAny(t,{'pref','power reference','power_ref'}), score=score+7; reasons{end+1}='power-reference'; end %#ok<AGROW>
if hasAny(t,{'governor'}), score=score+5; reasons{end+1}='governor'; end %#ok<AGROW>
if hasAny(t,{'avr','exciter'}), score=score+5; reasons{end+1}='avr-exciter'; end %#ok<AGROW>
if hasAny(t,{'reference','setpoint','control'}), score=score+2; reasons{end+1}='reference-control'; end %#ok<AGROW>
if hasAny(t,{'generator','gen1','gen2','gen3','gen4','gen5','gen6','gen7','gen8','gen9','gen10'}), score=score+1; reasons{end+1}='generator-path'; end %#ok<AGROW>
reason=strjoin(reasons,',');
end

function tf=hasAny(text,terms)
tf=false;
for i=1:numel(terms)
    if contains(text,lower(terms{i})), tf=true; return; end
end
end

function writejson(path,obj)
fid=fopen(path,'w');
assert(fid>0,'AEFC:IO','Cannot open %s',path);
c=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(obj,'PrettyPrint',true),'char');
fwrite(fid,sprintf('\n'),'char');
end
