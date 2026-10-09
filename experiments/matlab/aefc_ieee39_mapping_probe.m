function aefc_ieee39_mapping_probe
% Causally validate four generator-local IEEE39 observation/actuation pairs.
% Actuators are distinct-generator Pref control edges. For each actuator, the
% observation selector prefers the same generator's physical rotor speed or
% delta-speed signal and explicitly rejects reference/setpoint signals.

repo = pwd;
resDir = fullfile(repo,'results','ieee39_r2024b');
discPath = fullfile(resDir,'interface_discovery.json');
assert(exist(discPath,'file') == 2,'AEFC:DiscoveryMissing','Run interface discovery first.');
disc = jsondecode(fileread(discPath));

nObs = 4;
nAct = 4;
ctrl = pickGeneratorControls(disc.control_edges,nAct);
assert(numel(ctrl) >= nAct,'AEFC:ControlMapping', ...
    'Need %d distinct generator Pref actuator edges; found %d.',nAct,numel(ctrl));
obs = pairPhysicalObservations(disc.observation_edges,ctrl);
assert(numel(obs) >= nObs,'AEFC:ObservationMapping', ...
    'Need %d paired physical observations; found %d.',nObs,numel(obs));

[original,sourcePath] = aefc_ieee39_load_official;
probeModel = 'IEEE39BusSystem_AEFCProbe';
probePath = fullfile(resDir,[probeModel '.slx']);
if bdIsLoaded(original)
    close_system(original,0);
end
if bdIsLoaded(probeModel)
    close_system(probeModel,0);
end
if exist(probePath,'file') == 2
    delete(probePath);
end
[okCopy,msg] = copyfile(sourcePath,probePath,'f');
assert(okCopy,'AEFC:ProbeCopy','Could not copy official IEEE39 model: %s',msg);
load_system(probePath);
assert(bdIsLoaded(probeModel),'AEFC:ProbeLoad','Probe model did not load.');
try
    set_param(probeModel,'ReturnWorkspaceOutputs','on');
catch
end
try
    set_param(probeModel,'SignalLogging','off');
catch
end

obsNames = cell(nObs,1);
for i = 1:nObs
    src = translatePath(obs(i).src_block,original,probeModel);
    parent = get_param(src,'Parent');
    makeEditable(parent);
    ph = get_param(src,'PortHandles');
    assert(numel(ph.Outport) >= obs(i).src_port,'AEFC:ObservationPort', ...
        'Missing observation port on %s',src);
    sp = ph.Outport(obs(i).src_port);
    obsNames{i} = sprintf('AEFC_PROBE_OBS_%02d',i);
    tw = [parent sprintf('/AEFC_PROBE_TOWS_%02d',i)];
    if getSimulinkBlockHandle(tw) > 0
        delete_block(tw);
    end
    pos = get_param(src,'Position');
    x = pos(3) + 80;
    y = pos(2) + 25*i;
    add_block('simulink/Sinks/To Workspace',tw, ...
        'VariableName',obsNames{i},'SaveFormat','Timeseries', ...
        'Position',[x y x+105 y+30]);
    twPH = get_param(tw,'PortHandles');
    add_line(parent,sp,twPH.Inport,'autorouting','on');
end

injBlocks = cell(nAct,1);
for j = 1:nAct
    src = translatePath(ctrl(j).src_block,original,probeModel);
    dst = translatePath(ctrl(j).dst_block,original,probeModel);
    parent = get_param(src,'Parent');
    assert(strcmp(parent,get_param(dst,'Parent')),'AEFC:CrossHierarchy', ...
        'Control edge crosses hierarchy.');
    makeEditable(parent);
    srcPH = get_param(src,'PortHandles');
    dstPH = get_param(dst,'PortHandles');
    sp = srcPH.Outport(ctrl(j).src_port);
    dp = dstPH.Inport(ctrl(j).dst_port);
    pos = get_param(dst,'Position');
    x = max(10,pos(1)-150);
    y = pos(2) + 25*(j-1);
    sumPath = [parent sprintf('/AEFC_PROBE_SUM_%02d',j)];
    cPath = [parent sprintf('/AEFC_PROBE_INJ_%02d',j)];
    if getSimulinkBlockHandle(sumPath) > 0
        delete_block(sumPath);
    end
    if getSimulinkBlockHandle(cPath) > 0
        delete_block(cPath);
    end
    add_block('simulink/Math Operations/Sum',sumPath, ...
        'Inputs','++','Position',[x y x+35 y+35]);
    add_block('simulink/Sources/Constant',cPath, ...
        'Value','0','Position',[x-100 y+45 x-55 y+75]);
    sumPH = get_param(sumPath,'PortHandles');
    cPH = get_param(cPath,'PortHandles');
    delete_line(parent,sp,dp);
    add_line(parent,sp,sumPH.Inport(1),'autorouting','on');
    add_line(parent,cPH.Outport,sumPH.Inport(2),'autorouting','on');
    add_line(parent,sumPH.Outport,dp,'autorouting','on');
    injBlocks{j} = cPath;
end
save_system(probeModel);

probeStop = 0.30;
epsilon = 1e-3;
for j = 1:nAct
    set_param(injBlocks{j},'Value','0');
end
fastRestart = false;
try
    set_param(probeModel,'FastRestart','on');
    fastRestart = true;
catch
end

y0 = runAndRead(probeModel,probeStop,obsNames);
B = nan(nObs,nAct);
valid = false(1,nAct);
err = cell(1,nAct);
for j = 1:nAct
    for q = 1:nAct
        set_param(injBlocks{q},'Value','0');
    end
    set_param(injBlocks{j},'Value',num2str(epsilon,'%.17g'));
    try
        y1 = runAndRead(probeModel,probeStop,obsNames);
        B(:,j) = (y1-y0)/epsilon;
        valid(j) = all(isfinite(B(:,j))) && norm(B(:,j),2) > 1e-7;
        if ~valid(j)
            err{j} = 'finite response below 1e-7 norm';
        end
    catch ME
        err{j} = [ME.identifier ': ' ME.message];
    end
end
if fastRestart
    try
        set_param(probeModel,'FastRestart','off');
    catch
    end
end

Bfinite = B;
Bfinite(~isfinite(Bfinite)) = 0;
rankB = rank(Bfinite);
condB = cond(Bfinite);
obsGen = arrayfun(@(e) generatorKey(e.src_block),obs,'UniformOutput',false);
ctrlGen = arrayfun(@(e) generatorKey(e.src_block),ctrl,'UniformOutput',false);

mapping = struct;
mapping.model = 'IEEE39BusSystem';
mapping.release = version('-release');
mapping.mapping_type = 'native_r2024b_causal_probe';
mapping.selection_policy = 'paired-generator-physical-speed-v2';
mapping.observation_edges = obs;
mapping.control_edges = ctrl;
mapping.observation_generator_keys = obsGen;
mapping.control_generator_keys = ctrlGen;
mapping.observation_workspace_names = obsNames;
mapping.observation_transform = 'physical speed channels are real scalars; complex fallback uses magnitude';
mapping.baseline_observation = y0(:)';
mapping.probe_stop_time = probeStop;
mapping.probe_epsilon = epsilon;
mapping.sensitivity_matrix = B;
mapping.control_valid = valid;
mapping.control_errors = err;
mapping.sensitivity_rank = rankB;
mapping.sensitivity_condition = condB;
mapping.fast_restart = fastRestart;
mapping.validated = all(valid) && rankB == 4 && isequal(obsGen(:),ctrlGen(:));
mapping.validation_rule = ['Four distinct generator-local Pref actuators must each cause a finite ' ...
    'nontrivial response in the paired physical speed observations; the 4x4 sensitivity matrix must be full rank.'];
mapping.probe_model_file = strrep(probePath,[repo filesep],'');
writejson(fullfile(resDir,'interface_mapping.json'),mapping);

status = struct('validated',mapping.validated, ...
    'valid_controls',sum(valid), ...
    'requested_controls',nAct, ...
    'sensitivity_rank',rankB, ...
    'sensitivity_condition',condB, ...
    'selection_policy',mapping.selection_policy, ...
    'mapping_file','results/ieee39_r2024b/interface_mapping.json');
writejson(fullfile(resDir,'mapping_status.json'),status);
close_system(probeModel,0);
assert(mapping.validated,'AEFC:MappingProbeFailed', ...
    'Generator-local native mapping failed validation. Review interface_mapping.json.');
end

function selected = pickGeneratorControls(edges,n)
selected = edges([]);
seen = {};
for i = 1:numel(edges)
    e = edges(i);
    g = generatorKey(e.src_block);
    if isempty(g) || any(strcmp(seen,g))
        continue;
    end
    txt = lower([e.src_block ' ' e.dst_block]);
    tail = lower(blockTail(e.src_block));
    if ~(strcmp(tail,'pref') || contains(txt,'/pref '))
        continue;
    end
    selected(end+1) = e; %#ok<AGROW>
    seen{end+1} = g; %#ok<AGROW>
    if numel(selected) >= n
        break;
    end
end
end

function selected = pairPhysicalObservations(edges,ctrl)
selected = edges([]);
for j = 1:numel(ctrl)
    g = generatorKey(ctrl(j).src_block);
    bestIdx = 0;
    bestScore = -inf;
    for i = 1:numel(edges)
        e = edges(i);
        if ~strcmp(generatorKey(e.src_block),g)
            continue;
        end
        txt = lower([e.src_block ' ' e.dst_block]);
        if contains(txt,'wref') || contains(txt,'vref') || contains(txt,'pref') || ...
                contains(txt,'reference') || contains(txt,'setpoint')
            continue;
        end
        tail = lower(blockTail(e.src_block));
        score = double(e.observation_score);
        if strcmp(tail,'w')
            score = score + 100;
        end
        if contains(lower(e.src_block),'delta') && contains(lower(e.src_block),'speed')
            score = score + 80;
        end
        if contains(txt,'rotor') && (contains(txt,'speed') || contains(txt,'velocity'))
            score = score + 60;
        end
        if score > bestScore
            bestScore = score;
            bestIdx = i;
        end
    end
    assert(bestIdx > 0,'AEFC:NoPhysicalObservation', ...
        'No non-reference physical observation found for %s.',g);
    selected(end+1) = edges(bestIdx); %#ok<AGROW>
end
end

function g = generatorKey(p)
t = regexp(p,'Generators/(Gen\d+@Bus\d+)','tokens','once');
if isempty(t)
    g = '';
else
    g = t{1};
end
end

function t = blockTail(p)
parts = strsplit(p,'/');
t = strtrim(parts{end});
end

function p = translatePath(p,oldRoot,newRoot)
if startsWith(p,[oldRoot '/'])
    p = [newRoot p(numel(oldRoot)+1:end)];
elseif strcmp(p,oldRoot)
    p = newRoot;
end
end

function makeEditable(block)
try
    s = get_param(block,'LinkStatus');
    if strcmp(s,'resolved')
        set_param(block,'LinkStatus','inactive');
    end
catch
end
end

function y = runAndRead(model,stopTime,names)
simIn = Simulink.SimulationInput(model);
simIn = simIn.setModelParameter('StopTime',num2str(stopTime,'%.17g'), ...
    'ReturnWorkspaceOutputs','on');
out = sim(simIn);
y = zeros(numel(names),1);
for i = 1:numel(names)
    try
        v = out.get(names{i});
    catch
        v = [];
    end
    if isempty(v)
        error('AEFC:WorkspaceOutputMissing','Missing %s.',names{i});
    end
    data = double(extractData(v));
    data = squeeze(data);
    if ~isreal(data)
        data = abs(data);
    end
    data = data(:);
    data = data(isfinite(data));
    assert(~isempty(data),'AEFC:SignalEmpty','No finite samples for %s.',names{i});
    first = max(1,numel(data)-9);
    y(i) = mean(data(first:end));
end
end

function d = extractData(v)
if isa(v,'timeseries')
    d = v.Data;
elseif isa(v,'Simulink.SimulationData.Dataset')
    assert(v.numElements > 0,'AEFC:DatasetEmpty','Workspace Dataset is empty.');
    el = v.getElement(1);
    d = extractData(el.Values);
elseif isa(v,'Simulink.SimulationData.Signal')
    d = extractData(v.Values);
elseif isnumeric(v) || islogical(v)
    d = v;
elseif isstruct(v) && isfield(v,'signals') && isfield(v.signals,'values')
    d = v.signals.values;
elseif isstruct(v) && isfield(v,'Data')
    d = v.Data;
else
    try
        d = v.Data;
    catch
        error('AEFC:WorkspaceFormat','Unsupported output class %s.',class(v));
    end
end
end

function writejson(path,obj)
fid = fopen(path,'w');
assert(fid > 0,'AEFC:IO','Cannot open %s.',path);
c = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(obj,'PrettyPrint',true),'char');
fwrite(fid,sprintf('\n'),'char');
end
