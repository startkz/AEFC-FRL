function out=aefc_ieee39_gateway_r2024a(cmd,varargin)
% Persistent single-plant IEEE39 gateway for MATLAB Engine / Python.
% Requires R2024a+ Simulink.Simulation. The gateway refuses initialization
% unless the strict bridge audit has already certified the migrated model.
persistent S
if nargin<1, error('AEFC:Gateway','Command required.'); end
cmd=lower(char(cmd));
switch cmd
    case 'init'
        if ~isempty(S), S=shutdown(S); end
        modelPath=char(varargin{1}); auditPath=char(varargin{2});
        if numel(varargin)>=3, stopTime=double(varargin{3}); else, stopTime=60; end
        audit=jsondecode(fileread(auditPath));
        assert(isfield(audit,'bridge_contract_ready') && audit.bridge_contract_ready,'Strict bridge audit has not certified this model.');
        assert(strcmp(audit.release,'2024a'),'Gateway provenance expects R2024a audit.');
        [modelDir,mdl,~]=fileparts(modelPath); addpath(modelDir);
        sourceDir=fullfile(fileparts(fileparts(modelDir)),'source','model'); if exist(sourceDir,'dir')==7, addpath(sourceDir); end
        load_system(modelPath);
        taps=install_taps(mdl);
        sm=simulation(mdl);
        sm=setModelParameter(sm,'StopTime',sprintf('%.17g',stopTime));
        controls=audit.generator_controls;
        basePref=zeros(1,10); baseVref=zeros(1,10);
        for g=1:10
            basePref(g)=str2double(controls(g).pref_value); baseVref(g)=str2double(controls(g).vref_value);
            assert(isfinite(basePref(g)) && isfinite(baseVref(g)),'Non-numeric audited control baseline.');
            sm=setBlockParameter(sm,controls(g).pref_path,'Value',sprintf('%.17g',basePref(g)));
            sm=setBlockParameter(sm,controls(g).vref_path,'Value',sprintf('%.17g',baseVref(g)));
        end
        S=struct('mdl',mdl,'model_path',modelPath,'audit_path',auditPath,'audit',audit,'sm',sm,'taps',{taps}, ...
            'base_pref',basePref,'base_vref',baseVref,'time',0,'stop_time',stopTime);
        out=status(S);
    case 'step'
        assert(~isempty(S),'Gateway is not initialized.');
        dPref=double(varargin{1}); dVref=double(varargin{2}); targetTime=double(varargin{3});
        dPref=dPref(:)'; dVref=dVref(:)'; assert(numel(dPref)==10 && numel(dVref)==10,'Actions must contain 10 Pref and 10 Vref deltas.');
        assert(all(isfinite(dPref)) && all(isfinite(dVref)),'Actions must be finite.');
        assert(targetTime>S.time && targetTime<=S.stop_time,'Pause time must increase and remain within stop time.');
        for g=1:10
            S.sm=setBlockParameter(S.sm,S.audit.generator_controls(g).pref_path,'Value',sprintf('%.17g',S.base_pref(g)+dPref(g)));
            S.sm=setBlockParameter(S.sm,S.audit.generator_controls(g).vref_path,'Value',sprintf('%.17g',S.base_vref(g)+dVref(g)));
        end
        finalStep=step(S.sm,PauseTime=targetTime);
        S.time=double(S.sm.Time);
        out=observation(S,logical(finalStep));
    case 'observe'
        assert(~isempty(S),'Gateway is not initialized.'); out=observation(S,false);
    case 'status'
        assert(~isempty(S),'Gateway is not initialized.'); out=status(S);
    case 'close'
        if ~isempty(S), S=shutdown(S); end; S=[]; out=struct('closed',true);
    otherwise
        error('AEFC:Gateway','Unknown command %s.',cmd);
end
end

function o=observation(S,finalStep)
simout=S.sm.SimulationOutput; wm=zeros(1,10); vpu=zeros(1,10); vabc=cell(1,10);
for g=1:10
    w=data_of(simout.get(sprintf('aefc_rt_wm_g%d',g))); vv=reshape_vabc(data_of(simout.get(sprintf('aefc_rt_vabc_g%d',g))));
    wm(g)=w(end); vabc{g}=vv(end,:);
    vb=S.audit.generator_vbase_volts(g); vpu(g)=sqrt(mean(double(vabc{g}).^2))/(vb/sqrt(3));
end
o=struct('time_s',S.time,'wm_pu',wm,'vpu',vpu,'vabc_volts',{vabc},'final_step',logical(finalStep), ...
    'all_finite',all(isfinite(wm)) && all(isfinite(vpu)),'runtime','Simulink.Simulation','release',version('-release'));
end

function x=status(S)
x=struct('initialized',true,'model',S.model_path,'audit',S.audit_path,'time_s',S.time,'stop_time_s',S.stop_time, ...
    'runtime','Simulink.Simulation','release',version('-release'),'agent_count',10,'coupled_plant',true);
end

function S=shutdown(S)
try, if ~isempty(S.sm), terminate(S.sm); end, catch, end
try, remove_taps(S.taps); catch, end
try, if bdIsLoaded(S.mdl), close_system(S.mdl,0); end, catch, end
end

function taps=install_taps(mdl)
taps=cell(1,20); k=0;
for g=1:10
    k=k+1; taps{k}=make_tap(mdl,sprintf('Wm_G%d',g),sprintf('aefc_rt_wm_g%d',g),k);
    k=k+1; taps{k}=make_tap(mdl,sprintf('V_bus_G%d',g),sprintf('aefc_rt_vabc_g%d',g),k);
end
end
function t=make_tap(mdl,tag,varName,k)
fn=sprintf('AEFC_RT_From_%02d',k); ln=sprintf('AEFC_RT_Log_%02d',k); fp=[mdl '/' fn]; lp=[mdl '/' ln]; y=30+30*k;
add_block('simulink/Signal Routing/From',fp,'GotoTag',tag,'Position',[40 y 120 y+14]);
add_block('simulink/Sinks/To Workspace',lp,'VariableName',varName,'SaveFormat','Timeseries','Position',[180 y-2 300 y+16]);
add_line(mdl,[fn '/1'],[ln '/1'],'autorouting','on'); t=struct('from',fp,'log',lp);
end
function remove_taps(taps)
for i=numel(taps):-1:1
    try, if getSimulinkBlockHandle(taps{i}.log)>0, delete_block(taps{i}.log); end, catch, end
    try, if getSimulinkBlockHandle(taps{i}.from)>0, delete_block(taps{i}.from); end, catch, end
end
end
function d=data_of(x)
if isa(x,'timeseries'), d=double(x.Data); elseif isstruct(x) && isfield(x,'signals'), d=double(x.signals.values); else, try, d=double(x.Data); catch, d=double(x); end, end
assert(~isempty(d),'Runtime logged signal is empty.');
end
function v=reshape_vabc(v)
v=squeeze(double(v)); if isvector(v) && numel(v)==3, v=reshape(v,1,3); end; if size(v,2)~=3 && size(v,1)==3, v=v.'; end; assert(size(v,2)==3,'Expected three-phase voltage.');
end
