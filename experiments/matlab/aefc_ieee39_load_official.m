function [model,path]=aefc_ieee39_load_official
% Load the official MathWorks IEEE39 model materialized by bootstrap.
repo=pwd;
resDir=fullfile(repo,'results','ieee39_r2024b');
resolutionPath=fullfile(resDir,'model_resolution.json');
assert(exist(resolutionPath,'file')==2,'AEFC:ResolutionMissing', ...
    'model_resolution.json is missing; run aefc_ieee39_bootstrap first.');
r=jsondecode(fileread(resolutionPath));
assert(isfield(r,'resolved_model') && ~isempty(r.resolved_model),'AEFC:ResolutionEmpty', ...
    'No resolved official IEEE39 model is recorded.');
path=char(r.resolved_model);
assert(exist(path,'file')==2,'AEFC:ResolvedModelMissing', ...
    'Resolved official IEEE39 model does not exist in this runner: %s',path);
[~,model]=fileparts(path);
assert(strcmp(model,'IEEE39BusSystem'),'AEFC:UnexpectedModel', ...
    'Expected IEEE39BusSystem.slx, got %s.',path);
if ~bdIsLoaded(model), load_system(path); end
end
