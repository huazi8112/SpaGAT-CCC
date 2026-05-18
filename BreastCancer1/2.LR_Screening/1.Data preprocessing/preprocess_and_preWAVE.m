clear all; clc; close all;
script_dir = fileparts(mfilename('fullpath'));
preproc_dir = fullfile(script_dir, '..', '..', '1.Preprocessing');
csv_ligand = fullfile(preproc_dir, 'ligand_expr_by_cell.csv');
csv_receptor = fullfile(preproc_dir, 'receptor_expr_by_cell.csv');

% L/R use same pipeline; row filter cutoffs match prior comments (%L 0.1  R 0.2).
jobs(1).csv_path = csv_ligand;
jobs(1).mean_cutoff = 0.1;
jobs(1).out_mat = 't_wavelet-all-L1.mat';
jobs(1).tag = 'ligand';

jobs(2).csv_path = csv_receptor;
jobs(2).mean_cutoff = 0.2;
jobs(2).out_mat = 't_wavelet-all-R1.mat';
jobs(2).tag = 'receptor';

for ji = 1:numel(jobs)
    job = jobs(ji);
    if ~isfile(job.csv_path)
        error('Missing %s CSV:\n%s', job.tag, job.csv_path);
    end

    %% ========== preprocess: matrix prep (no intermediate .mat) ==========

    data = readtable(job.csv_path, 'Delimiter', ',', 'VariableNamingRule', 'preserve');

    if width(data) < 1
        error('readtable returned no columns:\n%s', job.csv_path);
    end

    vn = data.Properties.VariableNames;
    if ismember('feature', vn)
        gene_names = data.feature;
    else
        vnStr = string(vn);
        iFeat = find(startsWith(vnStr, "feature"), 1);
        if ~isempty(iFeat)
            gene_names = data{:, iFeat};
        else
            gene_names = data{:, 1};
        end
    end

    point = gene_names;

    numeric_data = data(:, vartype('numeric'));
    data_array = table2array(numeric_data);
    result = data_array;

    data_r = p_data(result);

    index_0 = mean(data_r, 2) <= job.mean_cutoff;
    data_r(index_0, :) = [];
    point(index_0) = [];

    index_ercc = startsWith(point, 'ERCC');
    data_r(index_ercc, :) = [];
    point(index_ercc) = [];

    time = detectStableInterval(data_r);
    [m, n] = size(data_r);
    window_size = 5;
    processed_result = data_r;

    if time(1) == 0
        data_temp = p_data(data_r(:, 1:4));
        processed_result(:, 1:4) = data_temp;
    end

    for k = 2:size(time, 2)
        col1 = k * 2 - 1;
        col2 = col1 + 3;
        if time(k) == 0
            data_temp = p_data(data_r(:, col1 - 2:col2));
            processed_result(:, col1:col2) = data_temp(:, 3:end);
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
    t_data = processed_resultnew(:, selectedColumns);

    non_zero_row = any(abs(t_data) > eps_zero, 2);
    t_data = t_data(non_zero_row, :);
    point = point(non_zero_row);

    num_cols = size(t_data, 2);
    output_data = cell(size(t_data, 1) + 1, num_cols + 1);
    output_data{1, 1} = '';
    output_data(1, 2:end) = num2cell(1:num_cols);
    output_data(2:end, 1) = point;
    output_data(2:end, 2:end) = num2cell(t_data);

    %% ========== preWAVE: wavelet trend ==========

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
    title(sprintf('%s — original (gene %d)', job.tag, gene_idx));
    grid on;

    subplot(2, 1, 2);
    plot(1:n, trend(gene_idx, :), 'r-', 'LineWidth', 2);
    xlabel('Sample index');
    ylabel('Trend component');
    title(sprintf('%s — trend', job.tag));
    grid on;

    trend(abs(trend) < 0.0001) = 0;
    trend = abs(trend);
    t_data = trend;

    non_empty = any(t_data ~= 0, 2);
    t_data = t_data(non_empty, :);
    point = point(non_empty);

    output_file = fullfile(script_dir, job.out_mat);
    save(output_file, 't_data', 'point');
    disp(['[', job.tag, '] Trend components saved to: ', output_file]);
end
