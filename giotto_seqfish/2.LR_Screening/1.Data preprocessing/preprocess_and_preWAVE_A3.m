clear all; clc; close all;
script_dir = fileparts(mfilename('fullpath'));
preproc_dir = fullfile(script_dir, '..', '..', '1.Preprocessing');

%% ========== preprocess.m: receptor A2 matrix preprocessing (no intermediate .mat) ==========

data = readtable(fullfile(preproc_dir, 'Ligand_expr_by_cell_A3.csv'), ...
    'Delimiter', ',', 'VariableNamingRule', 'preserve');

gene_names = data{:, 1};
point = gene_names;

numeric_data = data(:, vartype('numeric'));
data_array = table2array(numeric_data);
result = data_array;

data_r = p_data(result);
%% L0.3 R0.3
index_0 = mean(data_r, 2) <= 0.3;
data_r(index_0, :) = [];
point(index_0) = [];

index_ercc = startsWith(point, 'ERCC');
data_r(index_ercc, :) = [];
point(index_ercc) = [];

time = detectStableInterval(data_r);
m = size(data_r, 1);
n = size(data_r, 2);
window_size = 5;
processed_result = data_r;

if time(1) == 0
    data_temp = p_data(data_r(:, 1:4));
    processed_result(:, 1:4) = data_temp;
end

for k = 2:size(time,2)
    col1 = k * 2 -1;
    col2 = col1 + 3;
    if time(k) == 0
        data_temp = p_data(data_r(:, col1-2:col2));
        processed_result(:, col1:col2) = data_temp(:,3:end);
    end
end

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

rng(123, 'twister');
t_data = processed_resultnew;
totalCols = size(processed_result, 2) * interpFactor;
targetCols = 100;
eps_zero = 1e-12;
sobol = sobolset(1, 'Skip', 1e3, 'Leap', 1e2);
sobol = scramble(sobol, 'MatousekAffineOwen');


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
used_slots = false(1, targetCols); % track which slots have been replaced
for rr = zero_rows_after_sample'
    row_non_zero_cols = find(abs(processed_resultnew(rr, :)) > eps_zero);
    candidates = setdiff(row_non_zero_cols, selectedColumns); % cols not yet selected
    if ~isempty(candidates)
        new_col = candidates(randi(numel(candidates))); % random pick among candidates
        % find an unused slot to replace (prefer slots not yet touched)
        free_slots = find(~used_slots);
        if ~isempty(free_slots)
            slot = free_slots(randi(numel(free_slots)));
        else
            % all slots used once; pick a random slot (last resort)
            slot = randi(targetCols);
        end
        selectedColumns(slot) = new_col;
        used_slots(slot) = true;
    end
end

selectedColumns = sort(selectedColumns);
t_data = processed_resultnew(:, selectedColumns);

num_cols = size(t_data, 2);
output_data = cell(size(t_data, 1) + 1, num_cols + 1);
output_data{1, 1} = '';
output_data(1, 2:end) = num2cell(1:num_cols);
output_data(2:end, 1) = point;
output_data(2:end, 2:end) = num2cell(t_data);

%% ========== preWAVE.m: wavelet trend ==========

exp = t_data;
[m, n] = size(exp);

wavelet_name = 'db4';
level = 5;

trend = zeros(m, n);
for i = 1:m
    [C, L] = wavedec(exp(i, :), level, wavelet_name);
    trend(i, :) = wrcoef('a', C, L, wavelet_name, level);
end

gene_idx = 1;
figure;
subplot(2, 1, 1);
plot(1:n, exp(gene_idx, :), 'b-', 'LineWidth', 1.5);
xlabel('Sample index');
ylabel('Expression value');
title(['Original data - Gene ', num2str(gene_idx)]);
grid on;

subplot(2, 1, 2);
plot(1:n, trend(gene_idx, :), 'r-', 'LineWidth', 2);
xlabel('Sample index');
ylabel('Trend component');
title(['Trend component - Gene ', num2str(gene_idx)]);
grid on;

trend(abs(trend) < 0.0001) = 0;
trend = abs(trend);
t_data = trend;

non_empty = any(t_data ~= 0, 2);
t_data = t_data(non_empty, :);
point = point(non_empty, :);

output_file = fullfile(script_dir, 't_wavelet-all-L-A31.mat');
save(output_file, 't_data', 'point');
disp(['Trend components saved to: ', output_file]);
clear