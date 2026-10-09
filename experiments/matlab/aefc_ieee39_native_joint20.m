function aefc_ieee39_native_joint20
% Execute 20 matched seeds x four corruption cells x seven methods on the
% instrumented native MathWorks IEEE39BusSystem model. Every manuscript
% number is derived from per-run SimulationOutput evidence and raw JSONL.
% MethodCode, ScenarioCode, and Seed are numeric tunable System-object
% parameters, so the complete 7x4x20 matrix reuses one Fast Restart compile.
repo=pwd;
cfgPath=fullfile(repo,'configs','ieee39_joint20.json');
cfg=jsondecode(fileread(cfgPath));
mapPath=fullfile(repo,cfg.mapping_path); mapping=jsondecode(fileread(mapPath));
assert(mapping.validated,'AEFC:MappingInvalid','Native mapping must be validated before the joint matrix is enabled.');
assert(isfield(mapping,'selection_policy') && strcmp(mapping.selection_policy,'paired-generator-physical-speed-v2'), ...
    'AEFC:MappingPolicy','Authoritative native matrix requires paired-generator-physical-speed-v2 mapping.');
model=char(cfg.model); modelPath=fullfile(repo,'results','ieee39_r2024b',[model '.slx']);
if exist(modelPath,'file')~=2, aefc_ieee39_build_native_model; end
load_system(modelPath);
try, set_param(model,'ReturnWorkspaceOutputs','on'); catch, end
try, set_param(model,'SignalLogging','off'); catch, end

methods={'FedRL','RobustAgg','AEFC-no-PATBU','AEFC-no-Gate','AEFC-no-TGOPA','AEFC-no-Shield','AEFC-Full'};
scenarios={'clean','knowledge','physical','joint'};
outRoot=fullfile(repo,'results','ieee39_joint20'); if ~exist(outRoot,'dir'),mkdir(outRoot);end
center=double(mapping.baseline_observation(:)); scale=deriveScale(mapping.observation_edges,center);
configId=sha16(cfgPath); commitSha=getenv('GITHUB_SHA'); if isempty(commitSha),commitSha='LOCAL';end
rows=struct([]); rowIdx=0;
ctrl=[model '/AEFC_NATIVE_CONTROLLER'];
dt=double(cfg.sample_time); stopTime=double(cfg.stop_time); tq=(0:dt:stopTime)';

% Compile once. All experimental selectors are numeric tunable parameters.
set_param(ctrl,'MethodCode','1','ScenarioCode','1','Seed','0');
try, set_param(model,'SimulationCommand','update'); catch, end
fastRestart=false;
try, set_param(model,'FastRestart','on'); fastRestart=true; catch, end
assert(fastRestart,'AEFC:FastRestart','Full native matrix requires Fast Restart to avoid per-episode Simscape recompilation.');

for si=1:numel(scenarios)
    scenario=scenarios{si};
    for mi=1:numel(methods)
        method=methods{mi};
        runDir=fullfile(outRoot,scenario,method); if ~exist(runDir,'dir'),mkdir(runDir);end
        for seed=0:double(cfg.seeds)-1
            simIn=Simulink.SimulationInput(model);
            simIn=simIn.setBlockParameter(ctrl,'MethodCode',num2str(mi));
            simIn=simIn.setBlockParameter(ctrl,'ScenarioCode',num2str(si));
            simIn=simIn.setBlockParameter(ctrl,'Seed',num2str(seed));
            simIn=simIn.setModelParameter('StopTime',num2str(stopTime,'%.17g'),'ReturnWorkspaceOutputs','on');
            out=sim(simIn);
            obs=zeros(numel(tq),4);
            for q=1:4
                obs(:,q)=sampleOutput(out,sprintf('AEFC_NATIVE_OBS_%02d',q),tq,1);
            end
            diagv=sampleOutput(out,'AEFC_DIAG',tq,10);
            ushield=sampleOutput(out,'AEFC_SHIELD_VECTOR',tq,4);
            upost=sampleOutput(out,'AEFC_POST_ATTACK_VECTOR',tq,4);
            validateEvidence(diagv,upost,mi,si,cfg);

            z=(obs-center')./scale';
            maxabs=max(abs(z),[],2);
            safe=maxabs<=double(cfg.normalized_safe_limit)+1e-9;
            gate=diagv(:,3)>0.5; dangerous=diagv(:,9)>0.5;
            far=sum(gate & dangerous)/max(sum(dangerous),1);
            attackEnd=double(cfg.attack_end_step)*dt;
            recIdx=findRecovery(tq,maxabs,attackEnd,double(cfg.normalized_recovery_limit),5);
            if isempty(recIdx), recSuccess=0; recTime=stopTime-attackEnd; else, recSuccess=1; recTime=tq(recIdx)-attackEnd; end

            tracePath=fullfile(runDir,sprintf('seed_%02d.jsonl',seed));
            fid=fopen(tracePath,'w'); assert(fid>0,'AEFC:IO','Cannot open trace file.');
            cleanup=onCleanup(@()safeClose(fid)); %#ok<NASGU>
            for k=1:numel(tq)
                r=struct('commit_sha',commitSha,'config_id',configId,'mapping_policy',mapping.selection_policy, ...
                    'seed',seed,'method',method,'scenario',scenario,'time',tq(k), ...
                    'observation_native',obs(k,:),'observation_normalized',z(k,:),'trust_score',diagv(k,1),'risk_upper_bound',diagv(k,2), ...
                    'gate_decision',logical(gate(k)),'belief_distortion',diagv(k,4),'policy_drift',diagv(k,5), ...
                    'shield_intervention',diagv(k,6),'predicted_margin',diagv(k,7),'attack_active',logical(diagv(k,8)>0.5), ...
                    'dangerous_proposal',logical(dangerous(k)),'adapt_accepted',logical(diagv(k,10)>0.5), ...
                    'shield_action',ushield(k,:),'post_shield_attack',upost(k,:),'executed_correction',ushield(k,:)+upost(k,:), ...
                    'safe_native_envelope',logical(safe(k)));
                fwrite(fid,jsonencode(r),'char'); fwrite(fid,sprintf('\n'),'char');
            end
            fclose(fid); fid=-1; clear cleanup;

            rowIdx=rowIdx+1;
            rows(rowIdx).commit_sha=commitSha; %#ok<AGROW>
            rows(rowIdx).config_id=configId; rows(rowIdx).mapping_policy=mapping.selection_policy;
            rows(rowIdx).seed=seed; rows(rowIdx).method=method; rows(rowIdx).scenario=scenario;
            rows(rowIdx).safety_violation=double(any(~safe));
            rows(rowIdx).min_native_margin=min(double(cfg.normalized_safe_limit)-maxabs);
            rows(rowIdx).false_authorization_rate=far;
            rows(rowIdx).max_belief_distortion=max(diagv(:,4));
            rows(rowIdx).max_policy_drift=max(diagv(:,5));
            rows(rowIdx).recovery_success=recSuccess; rows(rowIdx).recovery_time=recTime;
            rows(rowIdx).shield_intervention_ratio=mean(diagv(:,6)>1e-8);
            rows(rowIdx).backend='mathworks_ieee39_r2024b_native';
            s=rows(rowIdx); saveSummary(fullfile(runDir,sprintf('seed_%02d.summary.json',seed)),s);
        end
    end
end

try, set_param(model,'FastRestart','off'); catch, end
T=struct2table(rows); writetable(T,fullfile(outRoot,'summary.csv'));
manifest=struct('backend','mathworks_ieee39_r2024b_native','commit_sha',commitSha,'config_path','configs/ieee39_joint20.json', ...
    'config_id',configId,'mapping_path',cfg.mapping_path,'mapping_policy',mapping.selection_policy, ...
    'seeds',double(cfg.seeds),'methods',{methods},'scenarios',{scenarios},'rows',height(T), ...
    'native_model',model,'native_mapping_validated',logical(mapping.validated), ...
    'sample_time_s',dt,'fast_restart_scope','single compile across full 7x4x20 matrix using numeric tunable selectors', ...
    'evidence_capture','explicit To Workspace via orientation-safe SimulationOutput extraction with per-episode semantic guards');
saveSummary(fullfile(outRoot,'manifest.json'),manifest);
close_system(model,0);
end

function X=sampleOutput(out,name,tq,dim)
try, v=out.get(name); catch, v=[]; end
assert(~isempty(v),'AEFC:OutputMissing','Missing SimulationOutput variable %s',name);
[t,d]=extractSeries(v);
t=double(t(:)); d=double(d);
if ~isreal(d), d=abs(d); end
d=normalizeTimeMajor(d,numel(t),dim,name);
[t,ia]=unique(t,'stable'); d=d(ia,:);
assert(numel(t)>=2,'AEFC:OutputSamples','Signal %s has fewer than two samples.',name);
if numel(t)==numel(tq) && max(abs(t-tq))<=1e-9
    X=d(:,1:dim);
else
    X=interp1(t,d(:,1:dim),tq,'nearest','extrap');
end
assert(isequal(size(X),[numel(tq) dim]),'AEFC:OutputShape','Signal %s did not normalize to time-major %dx%d.',name,numel(tq),dim);
assert(all(isfinite(X),'all'),'AEFC:OutputFinite','Signal %s contains non-finite values.',name);
end

function d=normalizeTimeMajor(d,nT,dim,name)
sz=size(d);
if isvector(d) && dim==1
    assert(numel(d)==nT,'AEFC:OutputSamples','Scalar signal %s sample count mismatch.',name);
    d=d(:); return;
end
if numel(d)~=nT*dim
    error('AEFC:OutputDimension','Signal %s has %d values; expected %d samples x %d channels.',name,numel(d),nT,dim);
end
nd=ndims(d);
timeDim=find(sz==nT,1,'last');
assert(~isempty(timeDim),'AEFC:OutputTimeDimension','Cannot identify time dimension for %s.',name);
perm=[timeDim setdiff(1:nd,timeDim,'stable')];
d=permute(d,perm);
d=reshape(d,nT,[]);
assert(size(d,2)==dim,'AEFC:OutputDimension','Signal %s normalized to %d channels; expected %d.',name,size(d,2),dim);
end

function validateEvidence(diagv,upost,mi,si,cfg)
tol=1e-10;
assert(size(diagv,2)==10 && size(upost,2)==4,'AEFC:EvidenceShape','Native evidence vector shape mismatch.');
if any(mi==[1 2 6])
    assert(max(abs(diagv(:,6)))<=tol,'AEFC:InterventionSemantics','Method code %d must have zero shield intervention.',mi);
end
if any(mi==[1 2 3])
    assert(max(abs(diagv(:,1)-0.5))<=tol,'AEFC:TrustSemantics','Method code %d must retain fixed trust 0.5.',mi);
end
active=false(size(upost,1),1); active(double(cfg.attack_start_step)+1:double(cfg.attack_end_step)+1)=true;
amp=max(abs(upost),[],2);
if any(si==[1 2])
    assert(max(amp)<=tol,'AEFC:PostAttackSemantics','Clean/knowledge-only scenario contains post-shield physical attack.');
else
    assert(all(amp(~active)<=tol),'AEFC:PostAttackWindow','Post-shield attack appears outside configured window.');
    assert(all(amp(active)>tol),'AEFC:PostAttackWindow','Post-shield attack missing inside configured window.');
    assert(max(amp)<=double(cfg.post_shield_control_attack)+1e-10,'AEFC:PostAttackBound','Post-shield attack exceeds configured bound.');
end
assert(sum(diagv(:,8)>0.5)==double(cfg.attack_end_step-cfg.attack_start_step+1),'AEFC:AttackFlag','Attack-window diagnostic length mismatch.');
end

function [t,d]=extractSeries(v)
if isa(v,'timeseries')
    t=v.Time; d=v.Data;
elseif isa(v,'Simulink.SimulationData.Dataset')
    assert(v.numElements>0,'AEFC:DatasetEmpty','Dataset output is empty.');
    el=v.getElement(1); [t,d]=extractSeries(el.Values);
elseif isa(v,'Simulink.SimulationData.Signal')
    [t,d]=extractSeries(v.Values);
elseif isstruct(v)
    if isfield(v,'time') && isfield(v,'signals') && isfield(v.signals,'values')
        t=v.time; d=v.signals.values;
    elseif isfield(v,'Time') && isfield(v,'Data')
        t=v.Time; d=v.Data;
    else
        error('AEFC:OutputFormat','Unsupported struct output for native evidence.');
    end
else
    try, t=v.Time; d=v.Data;
    catch, error('AEFC:OutputFormat','Unsupported output class %s.',class(v)); end
end
end

function safeClose(fid)
if isnumeric(fid) && isscalar(fid) && fid>0
    try, fclose(fid); catch, end
end
end

function idx=findRecovery(t,maxabs,attackEnd,lim,holdN)
idx=[]; candidates=find(t>attackEnd);
for k=1:numel(candidates)
    i=candidates(k); j=min(numel(t),i+holdN-1);
    if j-i+1==holdN && all(maxabs(i:j)<=lim), idx=i; return; end
end
end

function s=deriveScale(edges,center)
s=zeros(4,1);
for i=1:4
    reason=''; try, reason=lower(edges(i).observation_reason); catch, end
    if contains(reason,'rotor-speed'), s(i)=0.02;
    elseif contains(reason,'terminal-voltage'), s(i)=0.10;
    elseif contains(reason,'rotor-angle'), s(i)=0.35;
    else, s(i)=max(0.02,0.10*max(abs(center(i)),1e-3)); end
end
end

function id=sha16(path)
bytes=uint8(fileread(path)); md=java.security.MessageDigest.getInstance('SHA-256'); md.update(bytes); d=typecast(md.digest(),'uint8');
hex=lower(reshape(dec2hex(d,2).',1,[])); id=hex(1:16);
end

function saveSummary(path,obj)
fid=fopen(path,'w'); assert(fid>0,'AEFC:IO','Cannot open %s',path); c=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(obj,'PrettyPrint',true),'char'); fwrite(fid,sprintf('\n'),'char');
end
