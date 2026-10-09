function aefc_ieee39_supplemental_cell(kind,value)
% Run one supplemental native IEEE39 cell with 20 matched seeds.
% Main Experiment results/ieee39_joint20 is read-only and never modified.
%
% kind/value:
%   intensity : joint attack scale gamma; methods 1,2,3,4,7
%   gate      : targeted unsafe learned-proposal blend alpha; methods 4,7
%   qmin      : AEFC-Full QMinOverride
%   rho       : AEFC-Full RhoMaxOverride
%   trust     : AEFC-Full TrustRegionOverride (Delta_theta)

repo=pwd; kind=char(string(kind)); value=double(value);
cfgPath=fullfile(repo,'configs','ieee39_joint20.json'); cfg=jsondecode(fileread(cfgPath));
mapPath=fullfile(repo,cfg.mapping_path); mapping=jsondecode(fileread(mapPath));
assert(mapping.validated,'AEFC:MappingInvalid','Supplemental native runs require validated mapping.');
assert(strcmp(mapping.selection_policy,'paired-generator-physical-speed-v2'), ...
    'AEFC:MappingPolicy','Supplemental native runs require authoritative paired mapping.');

attackScale=1.0; stressAlpha=0.0; rhoOverride=-1.0; qOverride=-1.0; trustOverride=-1.0;
switch lower(kind)
    case 'intensity'
        methodCodes=[1 2 3 4 7];
        methodNames={'FedRL','RobustAgg','AEFC-no-PATBU','AEFC-no-Gate','AEFC-Full'};
        attackScale=value;
    case 'gate'
        methodCodes=[4 7]; methodNames={'AEFC-no-Gate','AEFC-Full'};
        stressAlpha=value;
    case 'qmin'
        methodCodes=7; methodNames={'AEFC-Full'}; qOverride=value;
    case 'rho'
        methodCodes=7; methodNames={'AEFC-Full'}; rhoOverride=value;
    case 'trust'
        methodCodes=7; methodNames={'AEFC-Full'}; trustOverride=value;
    otherwise
        error('AEFC:SupplementalKind','Unknown supplemental kind: %s',kind);
end
assert(attackScale>=0,'AEFC:SupplementalValue','Attack scale must be nonnegative.');
assert(stressAlpha>=0 && stressAlpha<=1,'AEFC:SupplementalValue','Gate stress alpha must be in [0,1].');

tag=safeTag(value); outRoot=fullfile(repo,'results','supplemental_native',kind,tag);
if ~exist(outRoot,'dir'), mkdir(outRoot); end

model=char(cfg.model); modelPath=fullfile(repo,'results','ieee39_r2024b',[model '.slx']);
if exist(modelPath,'file')~=2, aefc_ieee39_build_native_model; end
load_system(modelPath);
try, set_param(model,'ReturnWorkspaceOutputs','on'); catch, end
try, set_param(model,'SignalLogging','off'); catch, end
ctrl=[model '/AEFC_NATIVE_CONTROLLER'];
set_param(ctrl,'System','AEFCSupplementalNativeController','SimulateUsing','Interpreted execution');
try, set_param(model,'SimulationCommand','update'); catch ME, close_system(model,0); rethrow(ME); end

center=double(mapping.baseline_observation(:)); scale=deriveScale(mapping.observation_edges,center);
commitSha=getenv('GITHUB_SHA'); if isempty(commitSha), commitSha='LOCAL'; end
experimentId=sha16text(sprintf('%s|%.17g|%s',kind,value,commitSha));
dt=double(cfg.sample_time); stopTime=double(cfg.stop_time); tq=(0:dt:stopTime)';
scenarioCode=4; scenario='joint'; rows=struct([]); rowIdx=0;

set_param(ctrl,'MethodCode',num2str(methodCodes(1)),'ScenarioCode','4','Seed','0', ...
    'AttackScale',num2str(attackScale,'%.17g'),'ProposalStressAlpha',num2str(stressAlpha,'%.17g'), ...
    'RhoMaxOverride',num2str(rhoOverride,'%.17g'),'QMinOverride',num2str(qOverride,'%.17g'), ...
    'TrustRegionOverride',num2str(trustOverride,'%.17g'));
try, set_param(model,'SimulationCommand','update'); catch, end
fastRestart=false;
try, set_param(model,'FastRestart','on'); fastRestart=true; catch, end
assert(fastRestart,'AEFC:FastRestart','Supplemental native cell requires Fast Restart.');

for mi=1:numel(methodCodes)
    mc=methodCodes(mi); method=methodNames{mi};
    runDir=fullfile(outRoot,method); if ~exist(runDir,'dir'), mkdir(runDir); end
    for seed=0:double(cfg.seeds)-1
        simIn=Simulink.SimulationInput(model);
        simIn=simIn.setBlockParameter(ctrl,'MethodCode',num2str(mc));
        simIn=simIn.setBlockParameter(ctrl,'ScenarioCode',num2str(scenarioCode));
        simIn=simIn.setBlockParameter(ctrl,'Seed',num2str(seed));
        simIn=simIn.setBlockParameter(ctrl,'AttackScale',num2str(attackScale,'%.17g'));
        simIn=simIn.setBlockParameter(ctrl,'ProposalStressAlpha',num2str(stressAlpha,'%.17g'));
        simIn=simIn.setBlockParameter(ctrl,'RhoMaxOverride',num2str(rhoOverride,'%.17g'));
        simIn=simIn.setBlockParameter(ctrl,'QMinOverride',num2str(qOverride,'%.17g'));
        simIn=simIn.setBlockParameter(ctrl,'TrustRegionOverride',num2str(trustOverride,'%.17g'));
        simIn=simIn.setModelParameter('StopTime',num2str(stopTime,'%.17g'),'ReturnWorkspaceOutputs','on');
        out=sim(simIn);

        obs=zeros(numel(tq),4);
        for q=1:4, obs(:,q)=sampleOutput(out,sprintf('AEFC_NATIVE_OBS_%02d',q),tq,1); end
        diagv=sampleOutput(out,'AEFC_DIAG',tq,10);
        ushield=sampleOutput(out,'AEFC_SHIELD_VECTOR',tq,4);
        upost=sampleOutput(out,'AEFC_POST_ATTACK_VECTOR',tq,4);
        validateEvidence(diagv,upost,mc,cfg,attackScale);

        z=(obs-center')./scale'; maxabs=max(abs(z),[],2);
        safe=maxabs<=double(cfg.normalized_safe_limit)+1e-9;
        gate=diagv(:,3)>0.5; dangerous=diagv(:,9)>0.5;
        dangerousCount=sum(dangerous); dangerousAuth=sum(gate & dangerous);
        far=dangerousAuth/max(dangerousCount,1);
        attackEnd=double(cfg.attack_end_step)*dt;
        recIdx=findRecovery(tq,maxabs,attackEnd,double(cfg.normalized_recovery_limit),5);
        if isempty(recIdx), recSuccess=0; recTime=stopTime-attackEnd; else, recSuccess=1; recTime=tq(recIdx)-attackEnd; end

        tracePath=fullfile(runDir,sprintf('seed_%02d.jsonl',seed));
        fid=fopen(tracePath,'w'); assert(fid>0,'AEFC:IO','Cannot open trace file.');
        cleanup=onCleanup(@()safeClose(fid)); %#ok<NASGU>
        for k=1:numel(tq)
            r=struct('commit_sha',commitSha,'experiment_id',experimentId,'experiment_kind',kind, ...
                'level_value',value,'mapping_policy',mapping.selection_policy,'seed',seed,'method',method, ...
                'scenario',scenario,'time',tq(k),'observation_native',obs(k,:),'observation_normalized',z(k,:), ...
                'trust_score',diagv(k,1),'risk_upper_bound',diagv(k,2),'gate_decision',logical(gate(k)), ...
                'belief_distortion',diagv(k,4),'policy_drift',diagv(k,5),'shield_intervention',diagv(k,6), ...
                'predicted_margin',diagv(k,7),'attack_active',logical(diagv(k,8)>0.5), ...
                'dangerous_proposal',logical(dangerous(k)),'adapt_accepted',logical(diagv(k,10)>0.5), ...
                'shield_action',ushield(k,:),'post_shield_attack',upost(k,:), ...
                'executed_correction',ushield(k,:)+upost(k,:),'safe_native_envelope',logical(safe(k)));
            fwrite(fid,jsonencode(r),'char'); fwrite(fid,sprintf('\n'),'char');
        end
        fclose(fid); fid=-1; clear cleanup;

        rowIdx=rowIdx+1; rows(rowIdx).commit_sha=commitSha; %#ok<AGROW>
        rows(rowIdx).experiment_id=experimentId; rows(rowIdx).experiment_kind=kind; rows(rowIdx).level_value=value;
        rows(rowIdx).mapping_policy=mapping.selection_policy; rows(rowIdx).seed=seed; rows(rowIdx).method=method;
        rows(rowIdx).scenario=scenario; rows(rowIdx).safety_violation=double(any(~safe));
        rows(rowIdx).min_native_margin=min(double(cfg.normalized_safe_limit)-maxabs);
        rows(rowIdx).false_authorization_rate=far; rows(rowIdx).dangerous_proposal_count=dangerousCount;
        rows(rowIdx).dangerous_authorization_count=dangerousAuth;
        rows(rowIdx).dangerous_proposal_rate=mean(dangerous);
        rows(rowIdx).gate_authorization_rate=mean(gate);
        rows(rowIdx).max_belief_distortion=max(diagv(:,4)); rows(rowIdx).max_policy_drift=max(diagv(:,5));
        rows(rowIdx).mean_trust=mean(diagv(:,1)); rows(rowIdx).max_risk=max(diagv(:,2));
        rows(rowIdx).recovery_success=recSuccess; rows(rowIdx).recovery_time=recTime;
        rows(rowIdx).shield_intervention_ratio=mean(diagv(:,6)>1e-8);
        rows(rowIdx).mean_shield_intervention=mean(diagv(:,6)); rows(rowIdx).max_shield_intervention=max(diagv(:,6));
        rows(rowIdx).backend='mathworks_ieee39_r2024b_native_supplemental';
        saveSummary(fullfile(runDir,sprintf('seed_%02d.summary.json',seed)),rows(rowIdx));
    end
end

try, set_param(model,'FastRestart','off'); catch, end
T=struct2table(rows); writetable(T,fullfile(outRoot,'summary.csv'));
manifest=struct('backend','mathworks_ieee39_r2024b_native_supplemental','commit_sha',commitSha, ...
    'experiment_id',experimentId,'experiment_kind',kind,'level_value',value,'mapping_policy',mapping.selection_policy, ...
    'seeds',double(cfg.seeds),'methods',{methodNames},'scenario',scenario,'rows',height(T),'attack_scale',attackScale, ...
    'proposal_stress_alpha',stressAlpha,'rho_override',rhoOverride,'qmin_override',qOverride, ...
    'trust_region_override',trustOverride,'main_experiment_untouched',true,'sample_time_s',dt, ...
    'evidence_capture','orientation-safe SimulationOutput extraction with supplemental semantic guards');
saveSummary(fullfile(outRoot,'manifest.json'),manifest);
close_system(model,0);
end

function X=sampleOutput(out,name,tq,dim)
try, v=out.get(name); catch, v=[]; end
assert(~isempty(v),'AEFC:OutputMissing','Missing SimulationOutput variable %s',name);
[t,d]=extractSeries(v); t=double(t(:)); d=double(d); if ~isreal(d), d=abs(d); end
d=normalizeTimeMajor(d,numel(t),dim,name); [t,ia]=unique(t,'stable'); d=d(ia,:);
assert(numel(t)>=2,'AEFC:OutputSamples','Signal %s has fewer than two samples.',name);
if numel(t)==numel(tq) && max(abs(t-tq))<=1e-9, X=d(:,1:dim);
else, X=interp1(t,d(:,1:dim),tq,'nearest','extrap'); end
assert(isequal(size(X),[numel(tq) dim]),'AEFC:OutputShape','Signal %s shape mismatch.',name);
assert(all(isfinite(X),'all'),'AEFC:OutputFinite','Signal %s contains non-finite values.',name);
end

function d=normalizeTimeMajor(d,nT,dim,name)
sz=size(d);
if isvector(d) && dim==1
    assert(numel(d)==nT,'AEFC:OutputSamples','Scalar signal %s sample count mismatch.',name); d=d(:); return;
end
if numel(d)~=nT*dim, error('AEFC:OutputDimension','Signal %s value count mismatch.',name); end
nd=ndims(d); timeDim=find(sz==nT,1,'last');
assert(~isempty(timeDim),'AEFC:OutputTimeDimension','Cannot identify time dimension for %s.',name);
perm=[timeDim setdiff(1:nd,timeDim,'stable')]; d=permute(d,perm); d=reshape(d,nT,[]);
assert(size(d,2)==dim,'AEFC:OutputDimension','Signal %s channel count mismatch.',name);
end

function validateEvidence(diagv,upost,mc,cfg,attackScale)
tol=1e-10; assert(size(diagv,2)==10 && size(upost,2)==4,'AEFC:EvidenceShape','Evidence shape mismatch.');
if any(mc==[1 2 6]), assert(max(abs(diagv(:,6)))<=tol,'AEFC:InterventionSemantics','Method must have zero shield intervention.'); end
if any(mc==[1 2 3]), assert(max(abs(diagv(:,1)-0.5))<=tol,'AEFC:TrustSemantics','Method must retain fixed trust 0.5.'); end
active=false(size(upost,1),1); active(double(cfg.attack_start_step)+1:double(cfg.attack_end_step)+1)=true;
amp=max(abs(upost),[],2); bound=double(cfg.post_shield_control_attack)*attackScale;
if attackScale<=tol
    assert(max(amp)<=tol,'AEFC:PostAttackSemantics','Zero-scale run contains post-shield attack.');
else
    assert(all(amp(~active)<=tol),'AEFC:PostAttackWindow','Post-shield attack appears outside configured window.');
    assert(all(amp(active)>tol),'AEFC:PostAttackWindow','Post-shield attack missing inside configured window.');
    assert(max(amp)<=bound+1e-10,'AEFC:PostAttackBound','Post-shield attack exceeds scaled bound.');
end
assert(sum(diagv(:,8)>0.5)==double(cfg.attack_end_step-cfg.attack_start_step+1),'AEFC:AttackFlag','Attack flag length mismatch.');
end

function [t,d]=extractSeries(v)
if isa(v,'timeseries'), t=v.Time; d=v.Data;
elseif isa(v,'Simulink.SimulationData.Dataset'), assert(v.numElements>0,'AEFC:DatasetEmpty','Dataset is empty.'); el=v.getElement(1); [t,d]=extractSeries(el.Values);
elseif isa(v,'Simulink.SimulationData.Signal'), [t,d]=extractSeries(v.Values);
elseif isstruct(v)
    if isfield(v,'time') && isfield(v,'signals') && isfield(v.signals,'values'), t=v.time; d=v.signals.values;
    elseif isfield(v,'Time') && isfield(v,'Data'), t=v.Time; d=v.Data;
    else, error('AEFC:OutputFormat','Unsupported struct output.'); end
else
    try, t=v.Time; d=v.Data; catch, error('AEFC:OutputFormat','Unsupported output class %s.',class(v)); end
end
end

function safeClose(fid)
if isnumeric(fid) && isscalar(fid) && fid>0, try, fclose(fid); catch, end, end
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

function tag=safeTag(v)
tag=strrep(sprintf('%.6g',v),'.','p'); tag=strrep(tag,'-','m'); tag=strrep(tag,'+','');
end

function id=sha16text(txt)
md=java.security.MessageDigest.getInstance('SHA-256'); md.update(uint8(txt)); d=typecast(md.digest(),'uint8');
hex=lower(reshape(dec2hex(d,2).',1,[])); id=hex(1:16);
end

function saveSummary(path,obj)
fid=fopen(path,'w'); assert(fid>0,'AEFC:IO','Cannot open %s',path); c=onCleanup(@()fclose(fid)); %#ok<NASGU>
fwrite(fid,jsonencode(obj,'PrettyPrint',true),'char'); fwrite(fid,sprintf('\n'),'char');
end
