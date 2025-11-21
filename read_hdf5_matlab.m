% read_hdf5_matlab.m
% Example script for reading Bloch Simulator HDF5 files in MATLAB
%
% This script demonstrates:
% 1. Reading HDF5 files exported from Bloch Simulator
% 2. Accessing all data and parameters
% 3. Creating visualizations
% 4. Performing analysis
%
% Requirements: MATLAB R2011a or later (for HDF5 support)

%% ========================================================================
%  FUNCTION: Read complete HDF5 file
%  ========================================================================

function data = read_hdf5_complete(filename)
    % Read all data and parameters from Bloch Simulator HDF5 file
    %
    % Parameters:
    %   filename - Path to HDF5 file
    %
    % Returns:
    %   data - Structure containing all arrays and parameters

    fprintf('Reading file: %s\n', filename);
    fprintf('============================================================\n');

    data = struct();

    % 1. Read magnetization data
    fprintf('\n1. Loading magnetization data...\n');
    data.mx = h5read(filename, '/mx');
    data.my = h5read(filename, '/my');
    data.mz = h5read(filename, '/mz');
    data.signal = h5read(filename, '/signal');
    fprintf('   Magnetization size: [%s]\n', num2str(size(data.mx)));

    % 2. Read coordinate arrays
    fprintf('\n2. Loading coordinate arrays...\n');
    data.time = h5read(filename, '/time');
    data.positions = h5read(filename, '/positions');
    data.frequencies = h5read(filename, '/frequencies');
    fprintf('   Time points: %d\n', length(data.time));
    fprintf('   Duration: %.3f ms\n', data.time(end) * 1000);
    fprintf('   Positions: [%s]\n', num2str(size(data.positions)));
    fprintf('   Frequencies: [%s]\n', num2str(size(data.frequencies)));

    % 3. Read tissue parameters
    fprintf('\n3. Loading tissue parameters...\n');
    data.tissue = struct();
    tissue_attrs = {'name', 't1', 't2', 'density', 't2_star'};
    for i = 1:length(tissue_attrs)
        attr_name = tissue_attrs{i};
        try
            value = h5readatt(filename, '/tissue', attr_name);
            data.tissue.(attr_name) = value;
            fprintf('   %s: %s\n', attr_name, num2str(value));
        catch
            % Attribute doesn't exist (e.g., t2_star is optional)
        end
    end

    % 4. Read sequence parameters (if available)
    try
        info = h5info(filename, '/sequence_parameters');
        fprintf('\n4. Loading sequence parameters...\n');
        data.sequence_parameters = struct();

        % Read attributes
        for i = 1:length(info.Attributes)
            attr_name = info.Attributes(i).Name;
            value = h5readatt(filename, '/sequence_parameters', attr_name);
            data.sequence_parameters.(attr_name) = value;
            fprintf('   %s: %s\n', attr_name, num2str(value));
        end

        % Read datasets (like waveforms)
        for i = 1:length(info.Datasets)
            dataset_name = info.Datasets(i).Name;
            value = h5read(filename, ['/sequence_parameters/' dataset_name]);
            data.sequence_parameters.(dataset_name) = value;
            fprintf('   %s: size=[%s]\n', dataset_name, num2str(size(value)));
        end
    catch
        fprintf('\n4. No sequence parameters found in file\n');
        data.sequence_parameters = struct();
    end

    % 5. Read simulation parameters (if available)
    try
        info = h5info(filename, '/simulation_parameters');
        fprintf('\n5. Loading simulation parameters...\n');
        data.simulation_parameters = struct();

        % Read attributes
        for i = 1:length(info.Attributes)
            attr_name = info.Attributes(i).Name;
            value = h5readatt(filename, '/simulation_parameters', attr_name);
            data.simulation_parameters.(attr_name) = value;
            fprintf('   %s: %s\n', attr_name, num2str(value));
        end

        % Read datasets
        for i = 1:length(info.Datasets)
            dataset_name = info.Datasets(i).Name;
            value = h5read(filename, ['/simulation_parameters/' dataset_name]);
            data.simulation_parameters.(dataset_name) = value;
            fprintf('   %s: size=[%s]\n', dataset_name, num2str(size(value)));
        end
    catch
        fprintf('\n5. No simulation parameters found in file\n');
        data.simulation_parameters = struct();
    end

    % 6. Read metadata
    fprintf('\n6. Loading metadata...\n');
    data.metadata = struct();
    try
        data.metadata.export_timestamp = h5readatt(filename, '/', 'export_timestamp');
        fprintf('   export_timestamp: %s\n', data.metadata.export_timestamp);
    catch
    end
    try
        data.metadata.simulator_version = h5readatt(filename, '/', 'simulator_version');
        fprintf('   simulator_version: %s\n', data.metadata.simulator_version);
    catch
    end

    fprintf('\n============================================================\n');
    fprintf('Loading complete!\n\n');
end

%% ========================================================================
%  FUNCTION: Plot magnetization evolution
%  ========================================================================

function plot_magnetization_evolution(data, position_idx, freq_idx)
    % Plot magnetization evolution over time
    %
    % Parameters:
    %   data - Data structure from read_hdf5_complete()
    %   position_idx - Position index to plot (default: 1)
    %   freq_idx - Frequency index to plot (default: 1)

    if nargin < 2, position_idx = 1; end
    if nargin < 3, freq_idx = 1; end

    time_ms = data.time * 1000;  % Convert to ms

    % Check data dimensions
    dims = size(data.mx);
    if length(dims) == 3  % Time-resolved
        mx = squeeze(data.mx(:, position_idx, freq_idx));
        my = squeeze(data.my(:, position_idx, freq_idx));
        mz = squeeze(data.mz(:, position_idx, freq_idx));
    elseif length(dims) == 2  % Endpoint only
        fprintf('This is endpoint data (no time evolution)\n');
        return;
    else
        fprintf('Unexpected data shape\n');
        return;
    end

    % Create figure
    figure('Position', [100, 100, 1200, 800]);

    % Mx
    subplot(2, 2, 1);
    plot(time_ms, mx, 'b-', 'LineWidth', 1.5);
    xlabel('Time (ms)');
    ylabel('M_x');
    title('Transverse Magnetization (x)');
    grid on;

    % My
    subplot(2, 2, 2);
    plot(time_ms, my, 'r-', 'LineWidth', 1.5);
    xlabel('Time (ms)');
    ylabel('M_y');
    title('Transverse Magnetization (y)');
    grid on;

    % Mz
    subplot(2, 2, 3);
    plot(time_ms, mz, 'g-', 'LineWidth', 1.5);
    xlabel('Time (ms)');
    ylabel('M_z');
    title('Longitudinal Magnetization');
    grid on;

    % Magnitude
    subplot(2, 2, 4);
    mxy = sqrt(mx.^2 + my.^2);
    plot(time_ms, mxy, 'Color', [0.5 0 0.5], 'LineWidth', 1.5);
    xlabel('Time (ms)');
    ylabel('|M_{xy}|');
    title('Transverse Magnitude');
    grid on;

    % Overall title
    tissue_name = data.tissue.name;
    sgtitle(sprintf('Magnetization Evolution - %s\nPosition %d, Frequency %d', ...
            tissue_name, position_idx, freq_idx), ...
            'FontSize', 14, 'FontWeight', 'bold');
end

%% ========================================================================
%  FUNCTION: Plot signal
%  ========================================================================

function plot_signal(data, position_idx, freq_idx)
    % Plot complex signal over time
    %
    % Parameters:
    %   data - Data structure from read_hdf5_complete()
    %   position_idx - Position index to plot (default: 1)
    %   freq_idx - Frequency index to plot (default: 1)

    if nargin < 2, position_idx = 1; end
    if nargin < 3, freq_idx = 1; end

    time_ms = data.time * 1000;

    % Check data dimensions
    dims = size(data.signal);
    if length(dims) == 3
        signal = squeeze(data.signal(:, position_idx, freq_idx));
    elseif length(dims) == 2
        fprintf('Endpoint data - no time evolution\n');
        return;
    else
        fprintf('Unexpected signal shape\n');
        return;
    end

    % Create figure
    figure('Position', [100, 100, 1200, 600]);

    % Real and imaginary parts
    subplot(2, 1, 1);
    plot(time_ms, real(signal), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Real');
    hold on;
    plot(time_ms, imag(signal), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Imaginary');
    xlabel('Time (ms)');
    ylabel('Signal');
    title('Complex Signal Components');
    legend('Location', 'best');
    grid on;

    % Magnitude
    subplot(2, 1, 2);
    plot(time_ms, abs(signal), 'Color', [0.5 0 0.5], 'LineWidth', 1.5);
    xlabel('Time (ms)');
    ylabel('|Signal|');
    title('Signal Magnitude');
    grid on;

    % Overall title
    sgtitle(sprintf('MRI Signal - Position %d, Frequency %d', ...
            position_idx, freq_idx), ...
            'FontSize', 14, 'FontWeight', 'bold');
end

%% ========================================================================
%  FUNCTION: Plot spatial profile
%  ========================================================================

function plot_spatial_profile(data, freq_idx, time_idx)
    % Plot spatial profile of magnetization
    %
    % Parameters:
    %   data - Data structure from read_hdf5_complete()
    %   freq_idx - Frequency index to plot (default: 1)
    %   time_idx - Time index to plot (default: last time point)

    if nargin < 2, freq_idx = 1; end
    if nargin < 3, time_idx = size(data.mz, 1); end

    positions = data.positions;

    % Extract data
    dims = size(data.mz);
    if length(dims) == 3  % Time-resolved
        mz = squeeze(data.mz(time_idx, :, freq_idx));
        mx = squeeze(data.mx(time_idx, :, freq_idx));
        my = squeeze(data.my(time_idx, :, freq_idx));
        mxy = sqrt(mx.^2 + my.^2);
    elseif length(dims) == 2  % Endpoint
        mz = data.mz(:, freq_idx);
        mx = data.mx(:, freq_idx);
        my = data.my(:, freq_idx);
        mxy = sqrt(mx.^2 + my.^2);
    else
        fprintf('Unexpected data shape\n');
        return;
    end

    % Plot along z-axis (typical slice-select direction)
    z_pos = positions(3, :) * 100;  % Convert to cm

    % Create figure
    figure('Position', [100, 100, 1400, 500]);

    % Mz profile
    subplot(1, 2, 1);
    plot(z_pos, mz, 'go-', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'g');
    xlabel('Position (cm)');
    ylabel('M_z');
    title('Longitudinal Magnetization Profile');
    grid on;
    hold on;
    yline(0, 'k--', 'Alpha', 0.3);

    % Mxy profile
    subplot(1, 2, 2);
    plot(z_pos, mxy, 'mo-', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'm');
    xlabel('Position (cm)');
    ylabel('|M_{xy}|');
    title('Transverse Magnetization Profile');
    grid on;

    % Overall title
    freq = data.frequencies(freq_idx);
    if length(dims) == 3
        time_ms = data.time(time_idx) * 1000;
        sgtitle(sprintf('Spatial Profile - Frequency: %.1f Hz, Time: %.3f ms', ...
                freq, time_ms), 'FontSize', 14, 'FontWeight', 'bold');
    else
        sgtitle(sprintf('Spatial Profile - Frequency: %.1f Hz (Endpoint)', freq), ...
                'FontSize', 14, 'FontWeight', 'bold');
    end
end

%% ========================================================================
%  FUNCTION: Quick analysis
%  ========================================================================

function quick_analysis(data)
    % Perform quick analysis and print summary statistics
    %
    % Parameters:
    %   data - Data structure from read_hdf5_complete()

    fprintf('\n============================================================\n');
    fprintf('QUICK ANALYSIS\n');
    fprintf('============================================================\n');

    % Time info
    fprintf('\nTime Domain:\n');
    fprintf('  Duration: %.3f ms\n', data.time(end) * 1000);
    fprintf('  Time points: %d\n', length(data.time));
    if length(data.time) > 1
        fprintf('  Time step: %.3f µs\n', mean(diff(data.time)) * 1e6);
    end

    % Spatial info
    fprintf('\nSpatial Domain:\n');
    fprintf('  Positions: %d\n', size(data.positions, 2));
    fprintf('  Position range (z): %.3f to %.3f cm\n', ...
            min(data.positions(3, :)) * 100, ...
            max(data.positions(3, :)) * 100);

    % Frequency info
    fprintf('\nFrequency Domain:\n');
    fprintf('  Frequencies: %d\n', length(data.frequencies));
    fprintf('  Frequency range: %.1f to %.1f Hz\n', ...
            min(data.frequencies), max(data.frequencies));

    % Magnetization statistics
    dims = size(data.mx);
    if length(dims) == 3
        fprintf('\nMagnetization Statistics (final time point):\n');
        mx_final = squeeze(data.mx(end, :, :));
        my_final = squeeze(data.my(end, :, :));
        mz_final = squeeze(data.mz(end, :, :));

        fprintf('  Mx: min=%.4f, max=%.4f\n', min(mx_final(:)), max(mx_final(:)));
        fprintf('  My: min=%.4f, max=%.4f\n', min(my_final(:)), max(my_final(:)));
        fprintf('  Mz: min=%.4f, max=%.4f\n', min(mz_final(:)), max(mz_final(:)));

        % Find peak transverse magnetization
        mxy = sqrt(data.mx.^2 + data.my.^2);
        [max_mxy, max_idx] = max(mxy(:));
        [t_idx, p_idx, f_idx] = ind2sub(size(mxy), max_idx);

        fprintf('\nPeak Transverse Magnetization:\n');
        fprintf('  |Mxy|_max: %.4f\n', max_mxy);
        fprintf('  At time index: %d (%.3f ms)\n', t_idx, data.time(t_idx) * 1000);
        fprintf('  At position index: %d\n', p_idx);
        fprintf('  At frequency index: %d\n', f_idx);
    end

    % Tissue parameters
    fprintf('\nTissue Parameters:\n');
    if isfield(data.tissue, 'name')
        fprintf('  Name: %s\n', data.tissue.name);
    end
    if isfield(data.tissue, 't1')
        fprintf('  T1: %.1f ms\n', data.tissue.t1 * 1000);
    end
    if isfield(data.tissue, 't2')
        fprintf('  T2: %.1f ms\n', data.tissue.t2 * 1000);
    end
    if isfield(data.tissue, 't2_star')
        fprintf('  T2*: %.1f ms\n', data.tissue.t2_star * 1000);
    end

    fprintf('\n============================================================\n\n');
end

%% ========================================================================
%  EXAMPLE USAGE
%  ========================================================================

% Example 1: Load and visualize a single file
function example_basic()
    fprintf('\n============================================================\n');
    fprintf('EXAMPLE 1: Basic Loading and Visualization\n');
    fprintf('============================================================\n\n');

    % Replace with your actual HDF5 file path
    filename = 'example_data.h5';

    if ~exist(filename, 'file')
        fprintf('Error: File not found: %s\n', filename);
        fprintf('Please export data from the Bloch Simulator GUI first.\n');
        return;
    end

    % Load data
    data = read_hdf5_complete(filename);

    % Quick analysis
    quick_analysis(data);

    % Create visualizations
    plot_magnetization_evolution(data, 1, 1);
    plot_signal(data, 1, 1);

    % If multiple positions, show spatial profile
    if size(data.positions, 2) > 1
        plot_spatial_profile(data, 1);
    end

    fprintf('Visualization complete!\n');
end

% Example 2: Batch processing multiple files
function example_batch()
    fprintf('\n============================================================\n');
    fprintf('EXAMPLE 2: Batch Processing\n');
    fprintf('============================================================\n\n');

    % List of HDF5 files to process
    files = dir('*.h5');

    if isempty(files)
        fprintf('No HDF5 files found in current directory\n');
        return;
    end

    % Process each file
    for i = 1:length(files)
        fprintf('\nProcessing file %d/%d: %s\n', i, length(files), files(i).name);

        try
            data = read_hdf5_complete(files(i).name);
            quick_analysis(data);
        catch ME
            fprintf('Error processing %s: %s\n', files(i).name, ME.message);
        end
    end

    fprintf('\nBatch processing complete!\n');
end

%% ========================================================================
%  MAIN SCRIPT
%  ========================================================================

% Uncomment the example you want to run:

% example_basic();
% example_batch();

% Or load a specific file:
% data = read_hdf5_complete('your_file.h5');
% quick_analysis(data);
% plot_magnetization_evolution(data, 1, 1);

fprintf('\n');
fprintf('============================================================\n');
fprintf('HDF5 READING EXAMPLES FOR BLOCH SIMULATOR (MATLAB)\n');
fprintf('============================================================\n');
fprintf('\nAvailable functions:\n');
fprintf('  data = read_hdf5_complete(filename)  - Load complete HDF5 file\n');
fprintf('  quick_analysis(data)                 - Print summary statistics\n');
fprintf('  plot_magnetization_evolution(data)   - Plot Mx, My, Mz vs time\n');
fprintf('  plot_signal(data)                    - Plot complex signal\n');
fprintf('  plot_spatial_profile(data)           - Plot spatial profile\n');
fprintf('\nUsage:\n');
fprintf('  >> data = read_hdf5_complete(''mydata.h5'');\n');
fprintf('  >> quick_analysis(data);\n');
fprintf('  >> plot_magnetization_evolution(data, 1, 1);\n');
fprintf('============================================================\n\n');
