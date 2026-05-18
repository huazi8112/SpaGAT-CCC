clear all
script_dir = fileparts(mfilename('fullpath'));
preproc_dir = fullfile(script_dir, '..', '..', '1.Preprocessing');

% Read data table (preserve original column names)
data = readtable(fullfile(preproc_dir, 'receptor_expr_by_cell_A2.csv'), 'Delimiter', ',', 'VariableNamingRule', 'preserve');

% Extract gene names (assuming first column contains gene names)
gene_names = data{:, 1};
point = gene_names;

% Extract numeric data columns
numeric_data = data(:, vartype('numeric'));
data_array = table2array(numeric_data);
result = data_array;

% Data preprocessing
data_r = p_data(result);

% Filter low-expression genes (mean <= 1)
index_0 = mean(data_r, 2) <= 0.2;
data_r(index_0, :) = [];
point(index_0) = [];

% Remove ERCC genes
index_ercc = startsWith(point, 'ERCC');
data_r(index_ercc, :) = [];
point(index_ercc) = [];

% Detect stable intervals
time = detectStableInterval(data_r);
[m, n] = size(data_r);
window_size = 5;
processed_result = data_r;

% Process unstable intervals
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

% Log transform
processed_result = 10 * log(1 + processed_result);

% Interpolate data
interpFactor = 5;


processed_resultnew = zeros(size(processed_result, 1), size(processed_result, 2) * interpFactor);
for i = 1:size(processed_result, 1)
    current_row = processed_result(i, :);
    original_x = 1:size(processed_result, 2);
    target_x = linspace(1, size(processed_result, 2), size(processed_result, 2) * interpFactor);
    temp = interp1(original_x, current_row, target_x, 'linear');
    processed_resultnew(i, :) = temp;
end

% Select 500 columns via Sobol low-discrepancy sampling for more even coverage
t_data = processed_resultnew;
totalCols = size(processed_result, 2) * interpFactor;
targetCols = 100;
eps_zero = 1e-12;
sobol = sobolset(1, 'Skip', 1e3, 'Leap', 1e2);
sobol = scramble(sobol, 'MatousekAffineOwen');
rng(123, 'twister'); % fix seed for reproducibility of scrambling

% Prefer sampling from columns that are not all-zero globally
candidateCols = find(any(abs(t_data) > eps_zero, 1));
if isempty(candidateCols)
    candidateCols = (1:totalCols)';
end

numCandidates = numel(candidateCols);
batchSize = numCandidates; % one Sobol point per candidate column
mapped = floor(net(sobol, batchSize) * numCandidates) + 1;
idx = unique(candidateCols(mapped));

% Fallback: if uniqueness after mapping is insufficient, fill with remaining indices
if numel(idx) < targetCols
    remaining = setdiff(candidateCols, idx, 'stable');
    idx = [idx; remaining(:)];
end

% If non-zero candidates are fewer than targetCols, fill from all columns
if numel(idx) < targetCols
    remaining_all = setdiff((1:totalCols)', idx, 'stable');
    idx = [idx; remaining_all(:)];
end

if numel(idx) < targetCols
    error('Not enough columns to reach targetCols.');
end

selectedColumns = sort(idx(1:targetCols));
t_data = t_data(:, selectedColumns);

% Repair rows that become all-zero after sampling by injecting row-active columns
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

% Remove all-zero rows and keep point aligned
non_zero_row = any(abs(t_data) > eps_zero, 2);
t_data = t_data(non_zero_row, :);
point = point(non_zero_row);

% Save results (Malignant ligands dataset)
save(fullfile(script_dir, 't_wavelet-all-ori-L-A2.mat'), 't_data', 'point')

% Prepare Excel output
num_cols = size(t_data, 2);
output_data = cell(size(t_data, 1) + 1, num_cols + 1);
output_data{1, 1} = '';
output_data(1, 2:end) = num2cell(1:num_cols);
output_data(2:end, 1) = point;
output_data(2:end, 2:end) = num2cell(t_data);
clear;

