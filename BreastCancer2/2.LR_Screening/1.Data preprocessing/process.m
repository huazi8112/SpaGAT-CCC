% process.m (BreastCancer2) — Full pipeline: receptor CSV -> Sobol/ori -> db4 wavelet trend.
% Merges preprocess.m + preWAVE.m in one script; saves ori for inspection and wavelet tensor for downstream GCN.

clear all
clc

script_dir = fileparts(mfilename('fullpath'));

%% 1) Read receptor-by-cell matrix and apply p_data
data = readtable(fullfile(script_dir, 'receptor_expr_by_cell_A2.csv'), 'Delimiter', ',', 'VariableNamingRule', 'preserve');

gene_names = data{:, 1};
point = gene_names;

numeric_data = data(:, vartype('numeric'));
data_array = table2array(numeric_data);
result = data_array;

data_r = p_data(result);

% Low-expression filter (same threshold as preprocess.m)
index_0 = mean(data_r, 2) <= 0.2;
data_r(index_0, :) = [];
point(index_0) = [];

index_ercc = startsWith(point, 'ERCC');
data_r(index_ercc, :) = [];
point(index_ercc) = [];

%% 2) Stable-interval smoothing
time = detectStableInterval(data_r);
processed_result = data_r;

if time(1) == 0
    data_temp = p_data(data_r(:, 1:4));
    processed_result(:, 1:4)=data_temp;
end

for k = 2:size(time,2)
    col1 = k * 2 -1;
    col2 = col1 + 3;
    if time(k) == 0
        data_temp = p_data(data_r(:, col1-2:col2));
        processed_result(:, col1:col2)=data_temp(:,3:end);
    end
end

%% 3) Log, 5x interpolation, Sobol sampling with zero-column repair (preprocess.m)
processed_result = 10 * log(1 + processed_result);

interpFactor = 5;
processed_resultnew = zeros(size(processed_result, 1), size(processed_result, 2) * interpFactor);
for i = 1:size(processed_result, 1)
    current_row = processed_result(i, :);
    original_x = 1:size(processed_result, 2);
    target_x = linspace(1, size(processed_result, 2), size(processed_result, 2) * interpFactor);
    temp = interp1(original_x, current_row, target_x, 'linear');
    processed_resultnew(i, :) = temp;
end

t_data = processed_resultnew;
totalCols = size(processed_result, 2) * interpFactor;
targetCols = 100;
eps_zero = 1e-12;
sobol = sobolset(1, 'Skip', 1e3, 'Leap', 1e2);
sobol = scramble(sobol, 'MatousekAffineOwen');
rng(123, 'twister');

candidateCols = find(any(abs(t_data) > eps_zero, 1));
if isempty(candidateCols)
    candidateCols = (1:totalCols)';
end

numCandidates = numel(candidateCols);
batchSize = numCandidates;
mapped = floor(net(sobol, batchSize) * numCandidates) + 1;
idx = unique(candidateCols(mapped));

if numel(idx) < targetCols
    remaining = setdiff(candidateCols, idx, 'stable');
    idx = [idx; remaining(:)];
end

if numel(idx) < targetCols
    remaining_all = setdiff((1:totalCols)', idx, 'stable');
    idx = [idx; remaining_all(:)];
end

if numel(idx) < targetCols
    error('Not enough columns to reach targetCols.');
end

selectedColumns = sort(idx(1:targetCols));
t_data = t_data(:, selectedColumns);

zero_rows_after_sample = find(~any(abs(t_data) > eps_zero, 2));
replace_pos = targetCols;
for rr = zero_rows_after_sample'
    row_non_zero_cols = find(abs(processed_resultnew(rr, :)) > eps_zero);
    if ~isempty(row_non_zero_cols)
        new_col = row_non_zero_cols(find(~ismember(row_non_zero_cols, selectedColumns), 1, 'first'));
        if ~isempty(new_col)
            selectedColumns(replace_pos) = new_col;
            replace_pos = replace_pos - 1;
            if replace_pos < 1
                replace_pos = targetCols;
            end
        end
    end
end

selectedColumns = sort(selectedColumns);
t_data = processed_resultnew(:, selectedColumns);

non_zero_row = any(abs(t_data) > eps_zero, 2);
t_data = t_data(non_zero_row, :);
point = point(non_zero_row);

%% 4) Save ori + Excel-aligned cell array (before wavelet)
save_path_ori = fullfile(script_dir, 't_wavelet-all-ori-L-A2.mat');
save(save_path_ori, 't_data', 'point');

num_cols = size(t_data, 2);
output_data = cell(size(t_data, 1) + 1, num_cols + 1);
output_data{1, 1} = '';
output_data(1, 2:end) = num2cell(1:num_cols);
output_data(2:end, 1) = point;
output_data(2:end, 2:end) = num2cell(t_data);

%% 5) Wavelet trend (sizes derived from t_data)
[mGene, nSamp] = size(t_data);
exp = t_data;

wavelet_name = 'db4';
level = 5;
trend = zeros(mGene, nSamp);

for i = 1:mGene
    [C, L] = wavedec(exp(i, :), level, wavelet_name);
    trend(i, :) = wrcoef('a', C, L, wavelet_name, level);
end

gene_idx = 1;
figure;
subplot(2, 1, 1);
plot(1:nSamp, exp(gene_idx, :), 'b-', 'LineWidth', 1.5);
xlabel('Sample index');
ylabel('Expression value');
title(['Original data - Gene ', num2str(gene_idx)]);
grid on;

subplot(2, 1, 2);
plot(1:nSamp, trend(gene_idx, :), 'r-', 'LineWidth', 2);
xlabel('Sample index');
ylabel('Trend component');
title(['Trend component - Gene ', num2str(gene_idx)]);
grid on;

trend(abs(trend) < 0.0001) = 0;
trend = abs(trend);
t_data = trend;

non_empty = any(t_data ~= 0, 2);
t_data = t_data(non_empty, :);
point = point(non_empty);

%% 6) Wavelet tensor (basename matches GCNtest.py in ../2.Build a gene network/)
output_file = fullfile(script_dir, 't_wavelet-all-L-A2.mat');
save(output_file, 't_data', 'point');
fprintf('Saved: %s\n', save_path_ori);
fprintf('Saved: %s\n', output_file);
