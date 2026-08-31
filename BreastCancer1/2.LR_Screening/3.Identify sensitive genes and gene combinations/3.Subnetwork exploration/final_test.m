clear,clc;
% load results/adduwavelet-500-(all)-hESC.mat
% load data/prewavelet-hESC.mat
% load H_point_value_concat-R.mat
% load prewavelet-R.mat

script_dir = fileparts(mfilename('fullpath'));
load(fullfile(script_dir, '..', '2.Disturbance handling', 'adduwavelet-100-(all)-R-merged.mat'));
load(fullfile(script_dir, '..', '..', '2.Build a gene network', 'prewavelet-R.mat'));

% Filter genes below threshold
new_point = find(mean(H_point_value,2)<0.6);
new_name = name(new_point);

% Filter non-zero connections
new_maprho = maprho(new_point,new_point);
nonzero_rows = any(new_maprho, 2);
nonzero_cols = any(new_maprho, 1);
new_maprho = new_maprho(nonzero_rows,nonzero_cols);
new_Rectime= Rec_time(new_point,:);
new_Rectime=new_Rectime(nonzero_rows,:);
new_point=new_point(nonzero_rows);
new_name=new_name( nonzero_rows);

% Train initial network
[Hn0,tn0] = net_train(new_maprho,new_Rectime,10,3);
sample = 1:size(new_Rectime,1);

% Calculate total combinations
n=0;
for k = 1:size(new_Rectime,1)
    n = n + nchoosek(size(new_Rectime,1),k);
end

