%% BROADNESS ANALYSIS PIPELINE

% 0.  Startup
% 1.  Perform BROADNESS with MCS to test significant networks
%     – Temporal dimension, 100 permutations, max_abs: basis of sign of max value in absolute terms
% 2A. Visualise to explore: ALL plots
% 2B. Visualise to explore: each main brain network separately
% 3A. ANOVA for significant differences between all conditions
%     – Non-parametric, cluster-corrected ANOVA
%     – Important! How we permute condition labels!
% 3B. Posthoc t-tests for all three contrasts
%     – Looking within the significant clusters from ANOVA
%     – Cluster-based permutation t-tests
%     – Controls for Family-wise error rate (FWER)
% 3C. Holm-Bonferroni
%     – Correct for multiple post-hoc comparisons
% 3D. Build timepoint-wise tables for supplementary information
%     - For each PC and each contrast
%     - Export as CSV
% 4A. Figures: Variance explained
% 4B. Figures: Time series, one per component/brain network, overlaid
%              significance, condition (masked, unmasked) is legend 
% 4C. Figures: Time-series, one plot per condition, component/brain networks is legend
% 4D. Figures: Brain renders of included networks

%% 0) STARTUP

clear 
close all
clc

% Setup directories
path_home = '/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/3.0_BROADNESS_MEG_AuditoryRecognition-main/BROADNESS_Toolbox';
addpath(path_home)
BROADNESS_Startup(path_home);

% Creating output directory
outputdir = '/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/outputs/statistics_image';
if ~exist(outputdir, 'dir'), mkdir(outputdir); end

% Make directory to save images
figdir = '/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/outputs/figures_image';
if ~exist(figdir,'dir'), mkdir(figdir); end

%% 1) PERFORM BROADNESS (ALTERNATIVE SCENARIO WITH OPTIONAL INPUTS)

%%% ------------------- USER SETTINGS ------------------- %%%

list_participants = dir('/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/image_data/participants/*.mat');
disp(['loading participant 1 / ' num2str(length(list_participants))])
load([list_participants(1).folder '/' list_participants(1).name]);
SS = size(data); %CHANGED from Data to data
DATA = zeros(SS(1),876,SS(3),length(list_participants)); %preallocating space for all data
DATA(:,:,:,1) = data(:,1:876,:); %storing data for participant 1 %CHANGED from Data to data
for ii = 2:length(list_participants) %over participants
    disp(['loading participant ' num2str(ii) ' / ' num2str(length(list_participants))])
    load([list_participants(ii).folder '/' list_participants(ii).name]); %loading data for participant ii
    DATA(:,:,:,ii) = data(:,1:876,:); %storing data progressively for each participant %CHANGED from Data to data
end
% Load time from the file named:
load('/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/image_data/time/time.mat','time');

% Optional arguments
time_window = [-0.100 3.400]; %THIS WILL AFFECT PART 3
permutations_num = 100;
randomization = 1; %only time-points
sign_eigenvect = 'max_abs'; %based on sign of max value in absolute terms

%%% ------------------ COMPUTATION --------------------- %%%

BROADNESS = BROADNESS_NetworkEstimation(DATA, time, ...
                                'time_window', time_window, 'permutations_num', permutations_num, 'randomization', randomization, 'sign_eigenvect', sign_eigenvect); %call with optional parameters

%%% ------------------ SAVE INFO FOR FIGURES --------------------- %%%

% Used for: 4A. VARIANCE EXPLAINED BY PRINCIPAL COMPONENTS (PC)
var_exp = BROADNESS.Variance_BrainNetworks;
sign_bn = BROADNESS.Significant_BrainNetworks;
save(fullfile(outputdir, 'PC_variance_explained.mat'), 'var_exp', 'sign_bn', '-v7.3');

% Used for: 4B and 4C. TIME SERIES FOR EACH PC AND FOR EACH CONDITION
J = BROADNESS.TimeSeries_BrainNetworks; % time-points, principal components, conditions, subjects
save(fullfile(outputdir, 'timeseries_4D.mat'), 'J');
% hb_results = output from ttests surviving hb correction, finalised in 3C
% and saved in posthoc_withholm.mat as ttest_cluster_results

% Used for: 4D. THRESHOLDED BRAIN IMAGES PER PC
% dum = 3D array of source-space data used for PCA (size = nSources × nTime × nConditions)
dum = mean(BROADNESS.OriginalData,4); %average it across participants
% OUT.W / act_patt = spatial weights (eigenvectors) for each PCA component (size = nSources × nPCs)
act_patt = BROADNESS.ActivationPatterns_BrainNetworks;
save(fullfile(outputdir, 'for_thresholded_brain_images.mat'), 'dum', 'act_patt', '-v7.3');


%% 2A) BROADNESS VISUALIZATION

% SHOWS ALL PLOTS!

% This section generates 5 plots, described as follows:
%   #1) Dynamic brain activity map of the original data
%   #2) Variance explained by the networks
%   #3) Time series of the networks
%   #4) Activation patterns of the networks (3D)
%   #5) Activation patterns of the networks (nifti images)

%%% ------------------- USER SETTINGS ------------------- %%%

% Minimal user settings: output folder and MNI coordinates of original brain voxel data
Options = [];
Options.name_nii = '/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/outputs'; %output folder
load([path_home '/BROADNESS_External/MNI152_8mm_coord_dyi.mat']); %all voxels MNI coordinates
Options.MNI_coords = MNI8;
Options.Labels = {'Degraded1','ClearCue','Degraded2'}; %experimental condition labels

%%% ------------------ COMPUTATION --------------------- %%%

% Visualize ALL brain networks features
BROADNESS_Visualizer(BROADNESS,Options)

%% 2B) BROADNESS VISUALIZATION (ALTERNATIVE WITH OPTIONAL INPUTS)

% USEFUL FOR SHOWING ONE NETWORK AT A TIME ON TEMPLATE!

% This section demonstrates the same function as above,  
% but with optional settings provided. Any missing arguments  
% will automatically use their default values.  

%%% ------------------- ADDITIONAL USER SETTINGS ------------------- %%%

% Additional user settings: output folder
Options.WhichPlots = [0 0 0 1 0]; %which plots to be generated
Options.ncomps = [12]; %indices of PCs to be plotted (all plots)
Options.ncomps_var = 60; %number of PCs to be plotted (only in Variance plot)
Options.Labels = {'Degraded1','ClearCue','Degraded2'}; %experimental condition labels
Options.color_PCs = [
    0.4,    0.761,  0.647;   % teal-green
    0.988,  0.553,  0.384;   % coral
    0.553,  0.627,  0.796;   % periwinkle
    0.906,  0.541,  0.765;   % pink-purple
    0.651,  0.847,  0.329;   % green
    1.000,  0.851,  0.184;   % yellow
    0.898,  0.769,  0.580;   % beige
    0.702,  0.702,  0.702    % gray
]; %RGB code colors for PCs
Options.color_conds = [
    0.106,  0.620,  0.467;   % green
    0.851,  0.373,  0.008;   % orange
    0.459,  0.439,  0.702;   % purple
    0.906,  0.161,  0.541;   % magenta
    0.400,  0.651,  0.118;   % lime green
    0.902,  0.671,  0.008;   % gold
    0.651,  0.463,  0.114;   % brown
    0.4,    0.4,    0.4      % gray
]; %RGB code colors for experimental conditions

% If you wish to remove the cerebellum voxels (not included in 3D brain template (#4)), please set 'remove_cerebellum_label' to 1
% NOTE: This removal works only for 8mm brain
remove_cerebellum_label = 0;
if remove_cerebellum_label == 1
    load([path_home '/BROADNESS_External/cerebellum_coords.mat']); %only cerebellar voxels
    % Remove cerebellar voxels since they are not included in the 3D brain template (#4)
    [~, idx_cerebellum] = ismember(MNI8, cerebellum_coords, 'rows');  % find cerebellum indexes in MNI coordinates matrix (all voxels)
    MNI8(idx_cerebellum~=0,:) = nan; %assigning nans to MNI coordinates matrix
    Options.MNI_coords = MNI8; %assigning the MNI coordinates of your data for visualization purposes (both 3D main template (#4) and nifti images (#5))
end

%%% ------------------ COMPUTATION --------------------- %%%

% Visualize ONLY SOME brain networks features
BROADNESS_Visualizer(BROADNESS,Options)

%% 3A) –> 3D): STATISTICS SETTINGS AND PREP
%  ANOVA, post-hoc cluster t-tests, Holm, tables 

% SETTINGS (edit if you want different values)
alpha_pt      = 0.05;            % per-timepoint threshold for cluster formation
cluster_alpha = 0.05;            % cluster-level p threshold (reporting)
nperm         = 10000;            % number of permutations
stattype      = 'sum';           % cluster-mass (recommended for MEG)
contrasts     = [1 3; 1 2; 3 2]; % rows: [reference testing], 1=masked1, 2=unmasked, 3=masked2
alpha_family  = 0.05;            % family alpha for Holm (per-PC)
PCs           = 4;               % PCs to include in further analysis and plots

% Ensure required vars exist
if ~exist('J','var') || ~exist('time','var')
    error('This section expects J (time x nPC x nCond x nSubj) and time vector in workspace.');
end

% Trim time-window (optional)
starting_time = -0.100;
ending_time   = 3.400;
reduced_index = find(time >= starting_time & time <= ending_time);
lesstime = time(reduced_index);

% Reorder J to (PC, time, cond, subj) for easy indexing like you did before
dataa = permute(J, [2,1,3,4]);                  % PC x time x cond x subj
dataa_lesstime = dataa(:, reduced_index, :, :); % PC x time_in_window x cond x subj

% Read number of PCs to analyze from main script variable 'PCs'
if PCs > size(dataa_lesstime,1)
    error('Requested PCs > computed PCs. Adjust ''PCs'' or recompute BROADNESS.');
end

% Storage containers
Permtest_results_ANOVA = cell(PCs,1);        % significant ANOVA clusters per PC
Permtest_results_meta  = cell(PCs,1);        % meta outputs (F map, null)
Permtest_posthoc       = cell(PCs, size(contrasts,1)); % clusters returned per contrast per PC

% Also store per-timepoint maps for tables: Pmap, Tmap, mask (pre-Holm)
Pmap_all = nan(PCs, length(lesstime), size(contrasts,1));
Tmap_all = nan(PCs, length(lesstime), size(contrasts,1));
mask_all = false(PCs, length(lesstime), size(contrasts,1)); % 1 if sample included in any cluster (pre-Holm)

% Master summary rows
master_rows = {}; % will collect: PC, contrast_idx, anova_cluster_idx, cluster_idx, tstart, tend, cluster_p, Tmax, holm_pass

%% 3A) ANOVA
%  Cluster-based repeated-measures

for pp = 1:PCs
    % Build data for ANOVA: time x subjects x conditions
    tmp = squeeze(dataa_lesstime(pp,:,:,:));  % time x cond x subj
    S_anova = struct();
    S_anova.data = permute(tmp, [1 3 2]);      % time x subjects x conditions
    S_anova.nperm = nperm;
    S_anova.alpha = alpha_pt;
    S_anova.threshold = cluster_alpha;
    S_anova.time = lesstime;
    S_anova.stattype = stattype;
    
    fprintf('PC %d: running cluster-based repeated-measures ANOVA (time=%d, subj=%d, cond=%d)\n', ...
        pp, size(S_anova.data,1), size(S_anova.data,2), size(S_anova.data,3));
    
    [OUT_anova, OUTmeta] = clusterbased_permutationtest_anova(S_anova); %#ok<ASGLU>
    Permtest_results_ANOVA{pp} = OUT_anova;
    Permtest_results_meta{pp}  = OUTmeta;
end

% Save ANOVA results (so you can reload without re-running)
save(fullfile(outputdir, 'anova_perm_results_4_run.mat'), 'Permtest_results_ANOVA', 'Permtest_results_meta', '-v7.3');

% Permtest_results_ANOVA structure
% Rows = PCs/networks tested
% If no significant clusters for that PC: cell is []
% If significant clusters: cell is a struct array, one element per surviving cluster

%% 3B) POST-HOC CLUSTER T-TESTS
%  Within ANOVA clusters

% For each PC, for each contrast: restrict t-tests to ANOVA-significant cluster windows
for pp = 1:PCs
    OUT_anova = Permtest_results_ANOVA{pp};
    if isempty(OUT_anova)
        % No omnibus clusters for this PC -> nothing to do for post-hoc
        fprintf('PC %d: no ANOVA-significant clusters -> skipping post-hoc.\n', pp);
        continue;
    end
    
    % Build full data_for_anova: time x subjects x conditions (for this PC)
    tmp = squeeze(dataa_lesstime(pp,:,:,:));  % time x cond x subj
    data_for_anova = permute(tmp, [1 3 2]);   % time x subjects x conditions
    [ntime, nsubj, kcond] = size(data_for_anova);
    
    for c = 1:size(contrasts,1)
        refIdx = contrasts(c,1);
        testIdx= contrasts(c,2);
        collected_clusters = []; % will append clusters (struct array)
        % per-timepoint maps initialized for this contrast
        Pmap = nan(ntime,1);
        Tmap = nan(ntime,1);
        sigmask = false(ntime,1);
        
        % iterate ANOVA clusters, run t-test restricted to each cluster's timepoints
        for a = 1:numel(OUT_anova)
            an_mask = OUT_anova(a).mask(:); % binary vector length ntime
            times_in_cluster = lesstime(an_mask);
            if isempty(times_in_cluster)
                continue;
            end
            
            % build inputs for paired cluster-based permutation t-test (restricted to ANOVA cluster times)
            S_tt = struct();
            S_tt.reference = squeeze(data_for_anova(an_mask, :, refIdx)); % time_in_cluster x subj
            S_tt.testing   = squeeze(data_for_anova(an_mask, :, testIdx));% time_in_cluster x subj
            S_tt.nperm     = nperm;
            S_tt.alpha     = alpha_pt;
            S_tt.threshold = cluster_alpha;
            S_tt.time      = times_in_cluster;
            S_tt.stattype  = stattype;
            
            % run cluster permutation t-test (returns clusters within the ANOVA window)
            [OUT_tt, OUT_meta] = clusterbased_permutationtest_ttest(S_tt);
            % OUT_tt is a struct array of clusters (may be empty). Each cluster has .time and .pvalue etc.
            
            % compute per-timepoint paired t and p for this restricted window (for table output)
            % (we recompute per-timepoint t/p here because the t-test function returns cluster summaries only)
            tvec = zeros(sum(an_mask),1);
            pvec = ones(sum(an_mask),1);
            for tt_local = 1:sum(an_mask)
                [~, ptemp, ~, stats] = ttest(S_tt.reference(tt_local,:)', S_tt.testing(tt_local,:)');
                pvec(tt_local) = ptemp;
                tvec(tt_local) = stats.tstat;
            end
            % fill maps at global time indices
            global_idx = find(an_mask);
            Pmap(global_idx) = pvec;
            Tmap(global_idx) = tvec;
            
            % union significance mask from permutation clusters (pre-Holm)
            if ~isempty(OUT_tt)
                for k = 1:numel(OUT_tt)
                    % OUT_tt(k).time gives [tstart tend] absolute times (within lesstime)
                    t0 = OUT_tt(k).time(1); t1 = OUT_tt(k).time(2);
                    idx_mask = (lesstime >= t0 & lesstime <= t1);
                    sigmask(idx_mask) = true;
                    
                    % annotate the returned cluster by which ANOVA cluster it belonged to
                    OUT_tt(k).anova_cluster_idx = a;
                end
                % append clusters for this contrast
                collected_clusters = [collected_clusters; OUT_tt(:)]; %#ok<AGROW>
            else
                fprintf('PC %d – Contrast %d – Cluster %d: did not survive correction\n', pp, c, a);
            end
        end % end ANOVA clusters loop
        
        % store results for this PC x contrast
        Permtest_posthoc{pp,c} = collected_clusters;
        Pmap_all(pp,:,c) = Pmap;
        Tmap_all(pp,:,c) = Tmap;
        mask_all(pp,:,c) = sigmask;
    end % end contrasts
end % end PCs

% Save intermediate post-hoc results (before Holm)
save(fullfile(outputdir, 'posthoc_preholm_4_run.mat'), 'Permtest_posthoc', 'Pmap_all', 'Tmap_all', 'mask_all', '-v7.3');

% Permtest_posthoc structure
% Rows = PCs/networks tested
% Columns = contrasts (in this case: 3 contrasts)
% If a PC×contrast has no significant t-test clusters, cell is []

%% 3C) HOLM-BONFERRONI ACROSS CONTRASTS (PER PC)

% Compute the minimal cluster p per contrast per PC and apply Holm sequential
ttest_cluster_results = cell(PCs, size(contrasts,1)); % will hold cluster structs with Holm flags
summary_table = {}; % summary rows for master summary

for pp = 1:PCs
    % minimal cluster-level p per contrast (Inf if no clusters)
    minP = inf(size(contrasts,1),1);
    allClusterPs = cell(size(contrasts,1),1);
    for c = 1:size(contrasts,1)
        CL = Permtest_posthoc{pp,c};
        if isempty(CL)
            allClusterPs{c} = [];
            minP(c) = Inf;
        else
            pvals = arrayfun(@(x) x.pvalue, CL);
            allClusterPs{c} = pvals;
            minP(c) = min(pvals);
        end
    end
    
    % Holm procedure across the m contrasts (sequential Bonferroni)
    m = numel(minP);
    p_for_holm = minP;
    p_for_holm(isinf(p_for_holm)) = 1; % absent contrasts -> treated as non-significant
    [ps_sorted, sort_idx] = sort(p_for_holm);
    holm_reject_sorted = false(m,1);
    for i = 1:m
        thresh_i = alpha_family / (m - i + 1);
        if ps_sorted(i) <= thresh_i
            holm_reject_sorted(i) = true;
        else
            break; % stop sequentially as Holm prescribes
        end
    end
    % map back to original order
    holm_reject = false(m,1);
    holm_reject(sort_idx) = holm_reject_sorted;

    % Print HB summary for this PC
    for c = 1:size(contrasts,1)
        if holm_reject(c)
            fprintf('PC %d — Contrast %d (Cond %d vs Cond %d) SURVIVED Holm-Bonferroni (p_min = %.4f)\n', ...
                pp, c, contrasts(c,1), contrasts(c,2), minP(c));
        else
            fprintf('PC %d — Contrast %d (Cond %d vs Cond %d) did NOT survive Holm-Bonferroni (p_min = %.4f)\n', ...
                pp, c, contrasts(c,1), contrasts(c,2), minP(c));
        end
    end

    % Mark clusters as Holm significant or not and build final ttest_cluster_results
    for c = 1:size(contrasts,1)
        CL = Permtest_posthoc{pp,c}; % struct array
        if isempty(CL)
            ttest_cluster_results{pp,c} = [];
            continue;
        end
        % create S with cluster summaries + Holm flag
        S = CL;
        for k = 1:numel(S)
            S(k).holm_significant = holm_reject(c);
            % also create a binary mask per cluster for plotting (over lesstime)
            t0 = S(k).time(1); t1 = S(k).time(2);
            S(k).mask = (lesstime >= t0 & lesstime <= t1);
            % add row to master summary
            summary_row = {pp, c, S(k).anova_cluster_idx, k, t0, t1, S(k).pvalue, S(k).Tvalue, holm_reject(c)};
            master_rows = [master_rows; summary_row]; %#ok<AGROW>
        end
        ttest_cluster_results{pp,c} = S;
    end
end

% Save cluster-level results and Holm decisions
save(fullfile(outputdir, 'posthoc_withholm_4_run.mat'), 'Permtest_posthoc', 'ttest_cluster_results', '-v7.3');

%% 3D) BUILD PER-PC PER-CONTRAST TABLES + MASTER SUMMARY
% For each PC and contrast build a CSV and MATLAB struct with rows:
% 1) Time (lesstime)
% 2) Significant_timepoints (1/0)   <- pre-Holm cluster membership
% 3) Holm_timepoints (1/0)          <- sample marked significant only if contrast passed Holm
% 4) P_values (p if pre-Holm significant else 'N.S.')
% 5) T_values (t if pre-Holm significant else 'N.S.')

for pp = 1:PCs
    for c = 1:size(contrasts,1)
        refIdx = contrasts(c,1); testIdx = contrasts(c,2);
        Pmap = squeeze(Pmap_all(pp,:,c))'; % row vector length ntime
        Tmap = squeeze(Tmap_all(pp,:,c))';
        sigmask = squeeze(mask_all(pp,:,c))'; % logical vector pre-Holm
        % determine Holm decision for this contrast
        CL = ttest_cluster_results{pp,c};
        if isempty(CL)
            holm_pass = false;
        else
            holm_pass = CL(1).holm_significant; % all clusters in a contrast share the same Holm decision
        end
        % final holm mask: only samples that were in pre-Holm clusters AND contrast passed Holm
        holm_mask = sigmask & holm_pass;
        
        % Build cell array for CSV where first row header contains times
        header = ['Measure', arrayfun(@(x) sprintf('%.3f', x), lesstime, 'UniformOutput', false)];
        outcell = cell(5, length(lesstime)+1);
        outcell(1,:) = ['Time', num2cell(lesstime)]; % row of times (numeric converted to cells)
        % Row 2: Significant time points (pre-Holm, 1/0)
        outcell(2,1) = {'Significant_timepoints'};
        outcell(2,2:end) = num2cell(double(sigmask));
        % Row 3: Holm-surviving time points (1/0)
        outcell(3,1) = {'Holm_timepoints'};
        outcell(3,2:end) = num2cell(double(holm_mask));
        % Row 4: P values (show numeric for pre-Holm significant else 'N.S.')
        outcell(4,1) = {'P_values'};
        for tt = 1:length(lesstime)
            if sigmask(tt)
                outcell{4, tt+1} = Pmap(tt);
            else
                outcell{4, tt+1} = 'N.S.';
            end
        end
        % Row 5: T values (same convention)
        outcell(5,1) = {'T_values'};
        for tt = 1:length(lesstime)
            if sigmask(tt)
                outcell{5, tt+1} = Tmap(tt);
            else
                outcell{5, tt+1} = 'N.S.';
            end
        end
        
        % Save CSV
        fname_csv = fullfile(outputdir, sprintf('PC%d_Cond%dvCond%d_timepoint_table.csv', pp, refIdx, testIdx));
        try
            writecell(outcell, fname_csv);
        catch
            warning('Could not write CSV %s (writecell may not be supported on older MATLAB).', fname_csv);
            % fallback: save as .mat if needed
        end
        
        % Save MATLAB struct for later plotting/inspection
        table_struct = struct();
        table_struct.lesstime = lesstime;
        table_struct.sigmask = sigmask;
        table_struct.holm_mask = holm_mask;
        table_struct.Pmap = Pmap;
        table_struct.Tmap = Tmap;
        table_struct.refIdx = refIdx;
        table_struct.testIdx = testIdx;
        table_struct.holm_pass = holm_pass;
        save(fullfile(outputdir, sprintf('PC%d_Cond%dvCond%d_timepoint_table.mat', pp, refIdx, testIdx)), 'table_struct', '-v7.3');
    end
end

% Build master summary table from master_rows
% Columns: PC, contrast_idx, anova_cluster_idx, cluster_idx, tstart, tend, cluster_p, Tmax, holm_pass
if ~isempty(master_rows)
    master_tbl = cell2table(master_rows, 'VariableNames', {'PC','ContrastIdx','ANOVA_cluster','ClusterIdx','tstart','tend','cluster_p','Tmax','Holm_pass'});
    save(fullfile(outputdir,'posthoc_master_summary_4_run.mat'),'master_tbl','-v7.3');
    % also export CSV
    try
        writetable(master_tbl, fullfile(outputdir,'posthoc_master_summary_4.csv'));
    catch
        warning('Could not write master CSV summary.');
    end
else
    master_tbl = table();
    save(fullfile(outputdir,'posthoc_master_summary_4_run.mat'),'master_tbl','-v7.3');
end

fprintf('Sections 3A-3D completed. Results saved to %s\n', outputdir);

% Notes:
% - plotting code should read table_struct.holm_mask (or ttest_cluster_results) to draw Holm-surviving clusters
% - you can re-run section 3D if you want only to regenerate tables from saved posthoc_withholm.mat


%% Significanse summary

if exist('master_tbl','var') && ~isempty(master_tbl)

    % Keep only clusters where the contrast passed Holm-Bonferroni
    sig_tbl = master_tbl(master_tbl.Holm_pass == 1, :);

    if ~isempty(sig_tbl)
        % Duration of each significant cluster
        sig_tbl.Duration_s  = sig_tbl.tend - sig_tbl.tstart;
        sig_tbl.Duration_ms = sig_tbl.Duration_s * 1000;

        % Unique combinations of PC (network) and contrast index
        combos = unique(sig_tbl(:, {'PC','ContrastIdx'}), 'rows');

        % Prepare summary table
        nCombos = height(combos);
        summary_PC_contrast = combos;
        summary_PC_contrast.nClusters       = zeros(nCombos,1);
        summary_PC_contrast.TotalDuration_ms = zeros(nCombos,1);

        % Fill in number of clusters and total duration per PC × contrast
        for i = 1:nCombos
            pc_i   = combos.PC(i);
            cidx_i = combos.ContrastIdx(i);

            mask = (sig_tbl.PC == pc_i) & (sig_tbl.ContrastIdx == cidx_i);

            summary_PC_contrast.nClusters(i)        = sum(mask);
            summary_PC_contrast.TotalDuration_ms(i) = sum(sig_tbl.Duration_ms(mask));
        end

        % Add human-readable contrast labels (based on your contrasts order)
        % ContrastIdx = 1: Degraded1 vs Degraded2  (1 vs 3)
        % ContrastIdx = 2: Degraded1 vs ClearCue   (1 vs 2)
        % ContrastIdx = 3: Degraded2 vs ClearCue   (3 vs 2)
        summary_PC_contrast.ContrastLabel = strings(nCombos,1);
        for i = 1:nCombos
            switch summary_PC_contrast.ContrastIdx(i)
                case 1
                    summary_PC_contrast.ContrastLabel(i) = "Degraded1 vs Degraded2";
                case 2
                    summary_PC_contrast.ContrastLabel(i) = "Degraded1 vs ClearCue";
                case 3
                    summary_PC_contrast.ContrastLabel(i) = "Degraded2 vs ClearCue";
                otherwise
                    summary_PC_contrast.ContrastLabel(i) = "Unknown contrast";
            end
        end

        % ---- Print compact summary to Command Window ----
        fprintf('\n=== Summary of Holm-corrected pairwise differences (per PC × contrast) ===\n');
        for i = 1:nCombos
            pc    = summary_PC_contrast.PC(i);
            cidx  = summary_PC_contrast.ContrastIdx(i);
            nC    = summary_PC_contrast.nClusters(i);
            dur   = summary_PC_contrast.TotalDuration_ms(i);
            label = summary_PC_contrast.ContrastLabel(i);
            fprintf('PC %d — Contrast %d (%s): %d clusters, total %.0f ms significant\n', ...
                pc, cidx, label, nC, dur);
        end

        % Optional overall stats across all PCs & contrasts
        total_nClusters = sum(summary_PC_contrast.nClusters);
        total_dur_ms    = sum(summary_PC_contrast.TotalDuration_ms);
        fprintf('Overall: %d clusters across all PCs and contrasts, total %.0f ms significant.\n', ...
            total_nClusters, total_dur_ms);

        % ---- Save summary for later inspection (non-destructive) ----
        save(fullfile(outputdir,'posthoc_PC_contrast_summary_for_paper.mat'), ...
            'summary_PC_contrast','total_nClusters','total_dur_ms','-v7.3');

    else
        fprintf('\n=== Summary per PC × contrast ===\nNo Holm-corrected clusters found in master_tbl.\n');
    end
else
    fprintf('\n=== Summary per PC × contrast skipped ===\n');
    fprintf('Variable ''master_tbl'' not found or empty.\n');
end

%% Significant summary with intervals

%% 3F) Time-windowed summary (Flounders-style + custom windows)
% Uses master_tbl to summarise significant time within specific temporal windows.
% This section only reads master_tbl and creates new summary variables.

if exist('master_tbl','var') && ~isempty(master_tbl)

    % Keep only clusters from Holm-passing contrasts
    sig_tbl = master_tbl(master_tbl.Holm_pass == 1, :);

    if ~isempty(sig_tbl)
        % Duration of each significant cluster
        sig_tbl.Duration_s  = sig_tbl.tend - sig_tbl.tstart;
        sig_tbl.Duration_ms = sig_tbl.Duration_s * 1000;

        % ============================================================
        % 1) Define time windows
        % ============================================================

        % ---------- Flounders-inspired windows (in seconds) ----------
        % A: 0–0.3 s, B: 0.3–0.6 s, C: 0.6–1.0 s, D: 1.0–2.0 s
        fl_edges = [
            0.0 0.3;
            0.3 0.6;
            0.6 1.0;
            1.0 2.0];
        fl_labels = [ ...
            "F1_0-300ms"; ...
            "F2_300-600ms"; ...
            "F3_600-1000ms"; ...
            "F4_1000-2000ms"];

        % ---------- Your custom windows (in seconds) ----------
        % 0–1, 0–1.5, 1–2, 1.5–2, 2–3.4
        win_edges = [
            0.0 1.0;
            0.0 1.5;
            1.0 2.0;
            1.5 2.0;
            2.0 3.4];
        win_labels = [ ...
            "W1_0-1s"; ...
            "W2_0-1.5s"; ...
            "W3_1-2s"; ...
            "W4_1.5-2s"; ...
            "W5_2-3.4s"];

        % Unique combinations of PC and ContrastIdx
        combos  = unique(sig_tbl(:, {'PC','ContrastIdx'}), 'rows');
        nCombos = height(combos);

        % ---------- Helper function: overlap between two intervals ----------
        overlap_fun = @(a1,a2,b1,b2) max(0, min(a2,b2) - max(a1,b1)); % in seconds

        % ============================================================
        % 2) Flounders-window summary (per PC × contrast)
        % ============================================================

        nFl      = size(fl_edges,1);
        rows_fl  = {};  % {PC, ContrastIdx, WindowLabel, nClusters, TotalDuration_ms}

        for i = 1:nCombos
            pc_i   = combos.PC(i);
            cidx_i = combos.ContrastIdx(i);

            mask_pc_c = (sig_tbl.PC == pc_i) & (sig_tbl.ContrastIdx == cidx_i);
            clust_i   = sig_tbl(mask_pc_c, :);

            for w = 1:nFl
                wstart = fl_edges(w,1);
                wend   = fl_edges(w,2);

                nC = 0;
                totalDur_s = 0;

                for r = 1:height(clust_i)
                    t0 = clust_i.tstart(r);
                    t1 = clust_i.tend(r);
                    ov = overlap_fun(t0,t1,wstart,wend);
                    if ov > 0
                        nC = nC + 1;
                        totalDur_s = totalDur_s + ov;
                    end
                end

                if nC > 0
                    rows_fl(end+1,:) = {pc_i, cidx_i, fl_labels(w), nC, totalDur_s*1000}; %#ok<AGROW>
                end
            end
        end

        if ~isempty(rows_fl)
            summary_flounders = cell2table(rows_fl, ...
                'VariableNames', {'PC','ContrastIdx','WindowLabel','nClusters','TotalDuration_ms'});
        else
            summary_flounders = table();
        end

        % ============================================================
        % 3) Custom-window summary (per PC × contrast)
        % ============================================================

        nW       = size(win_edges,1);
        rows_cw  = {};  % {PC, ContrastIdx, WindowLabel, nClusters, TotalDuration_ms}

        for i = 1:nCombos
            pc_i   = combos.PC(i);
            cidx_i = combos.ContrastIdx(i);

            mask_pc_c = (sig_tbl.PC == pc_i) & (sig_tbl.ContrastIdx == cidx_i);
            clust_i   = sig_tbl(mask_pc_c, :);

            for w = 1:nW
                wstart = win_edges(w,1);
                wend   = win_edges(w,2);

                nC = 0;
                totalDur_s = 0;

                for r = 1:height(clust_i)
                    t0 = clust_i.tstart(r);
                    t1 = clust_i.tend(r);
                    ov = overlap_fun(t0,t1,wstart,wend);
                    if ov > 0
                        nC = nC + 1;
                        totalDur_s = totalDur_s + ov;
                    end
                end

                if nC > 0
                    rows_cw(end+1,:) = {pc_i, cidx_i, win_labels(w), nC, totalDur_s*1000}; %#ok<AGROW>
                end
            end
        end

        if ~isempty(rows_cw)
            summary_customwindows = cell2table(rows_cw, ...
                'VariableNames', {'PC','ContrastIdx','WindowLabel','nClusters','TotalDuration_ms'});
        else
            summary_customwindows = table();
        end

        % ============================================================
        % 4) Print compact text summary
        % ============================================================

        fprintf('\n=== Flounders-style window summary (per PC × contrast) ===\n');
        if ~isempty(summary_flounders)
            for i = 1:height(summary_flounders)
                pc   = summary_flounders.PC(i);
                cidx = summary_flounders.ContrastIdx(i);
                win  = summary_flounders.WindowLabel(i);
                nC   = summary_flounders.nClusters(i);
                dur  = summary_flounders.TotalDuration_ms(i);
                fprintf('PC %d — Contrast %d — %s: %d clusters, total %.0f ms\n', ...
                    pc, cidx, win, nC, dur);
            end
        else
            fprintf('No Holm-corrected clusters fall within the Flounders windows.\n');
        end

        fprintf('\n=== Custom window summary (per PC × contrast) ===\n');
        if ~isempty(summary_customwindows)
            for i = 1:height(summary_customwindows)
                pc   = summary_customwindows.PC(i);
                cidx = summary_customwindows.ContrastIdx(i);
                win  = summary_customwindows.WindowLabel(i);
                nC   = summary_customwindows.nClusters(i);
                dur  = summary_customwindows.TotalDuration_ms(i);
                fprintf('PC %d — Contrast %d — %s: %d clusters, total %.0f ms\n', ...
                    pc, cidx, win, nC, dur);
            end
        else
            fprintf('No Holm-corrected clusters fall within the custom windows.\n');
        end

        % ============================================================
        % 5) Save summaries for later use
        % ============================================================

        save(fullfile(outputdir,'posthoc_timewindow_summary_for_paper.mat'), ...
            'summary_flounders','summary_customwindows', ...
            'fl_edges','fl_labels', ...
            'win_edges','win_labels', ...
            '-v7.3');

    else
        fprintf('\n=== Time-windowed summary ===\nNo Holm-corrected clusters found in master_tbl.\n');
    end
else
    fprintf('\n=== Time-windowed summary skipped ===\n');
    fprintf('Variable ''master_tbl'' not found or empty.\n');
end




%% 4A. VARIANCE EXPLAINED BY PRINCIPAL COMPONENTS (PC)
%clear all
load('/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/outputs/statistics_image/PC_variance_explained.mat');
export_l = 1; % 1 to export the fig
colorline = [0.60 0.03 0.15; 0.3686 0.6314 0.7412; 0.1882 0.4902 0.8118; 0.0784 0.1569 0.5255; 0 0 0];
figure;
plot(var_exp(1:20),'Color',colorline(1,:),'Linewidth',3,'DisplayName','No rand');
hold on;
grid minor
set(gcf,'color','w')
box on
if export_l == 0
    legend('show');
end
if export_l ==1
    exportgraphics(gcf,'/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/outputs/figures_image/PC_variance_explained.pdf','Resolution',300)
end


%% Sissels tilføjelse


%% Sissels tilføjelse


load('/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/outputs/statistics_image/PC_variance_explained.mat');

export_l = 1; % 1 to export the fig

% ---- Farver ----
curve_gray    = [0.55 0.55 0.55];  % Grå til kurven
rest_gray     = curve_gray;        % Resten samme grå som kurven

% Hierarkisk farverækkefølge til de første 4 markeringer (mørk -> lys)
first4_colors = [
    0.60, 0.03, 0.15;   % 1: mørk bordeaux
    0.00, 0.12, 0.60;   % 2: dyb blå
    0.00, 0.60, 0.20;   % 3: grøn
    0.80, 0.40, 0.00    % 4: varm orange/guld
];

% ---- Figur: rektangulær ----
figure('Units','centimeters','Position',[2 2 19 11]);  % [x y width height]
hold on;

% ---- Plot variance explained (PC 1..20) ----
p_curve = plot(var_exp(1:20), 'Color', curve_gray, 'LineWidth', 3, ...
    'DisplayName', 'Variance explained');

% ---- Forbered signifikante markeringer inden for 1..20 ----
sign_bn = unique(sign_bn);                           % fjern dubletter
sign_bn = sign_bn(~isnan(sign_bn));                  % fjern NaN
sign_bn = sign_bn(sign_bn >= 1 & sign_bn <= 20);     % begræns til plottområde
sign_bn = sort(sign_bn(:));                          % sorter stigende og til kolonne

% Del op i de første 4 og resten
n_first   = min(4, numel(sign_bn));
idx_first = sign_bn(1:n_first);
idx_rest  = sign_bn(n_first+1:end);


% ---- Annotér de første 4 med varians-værdien i samme farve som stjernen ----
if ~isempty(idx_first)
    % Standard placering: til højre og let løftet
    x_offset_right = 0.32;  % vandret forskydning til højre (typisk 0.25–0.40)
    y_offset_up    = 0.30;  % lodret op for #2–#3 og normalt for #4 (før ekstra løft)

    % Ekstra løft for #4 (gul/orange) for at undgå overlap med kurven
    y_extra_up_4   = 1.3;  % finjustér 0.45–0.60 efter behov

    % Ekstra ned for #1 (rød/bordeaux) – 0.50 mere end de andre
    y_extra_down_1 = 0.50;

    for k = 1:numel(idx_first)
        pc_idx = idx_first(k);
        val    = var_exp(pc_idx);
        label  = sprintf('%.2f%%', val);  % 2 decimaler

        % Udgangspunkt: alle til højre for stjernen
        x_pos  = pc_idx + x_offset_right;
        y_pos  = val + y_offset_up;

        % Specialcases
        if k == 1
            % #1 (rød): rykkes ned 0.50 mere end de andre
            y_pos = val + y_offset_up - y_extra_down_1;
        elseif k == 4
            % #4 (gul/orange): løftes ekstra
            y_pos = val + y_extra_up_4;
        end

        text(x_pos, y_pos, label, ...
            'Color', first4_colors(k,:), ...
            'FontWeight', 'normal', ...
            'FontSize', 9, ...
            'HorizontalAlignment', 'left', ...   % teksten starter ved x_pos
            'VerticalAlignment',   'middle', ...
            'Interpreter', 'none', ...
            'Clipping', 'on');
    end
end



% ---- Plot markeringer: første 4 hver sin farve (STJERNER, lidt mindre) ----
h_first = gobjects(0);
for k = 1:n_first
    h_first(k,1) = scatter(idx_first(k), var_exp(idx_first(k)), 80, '*', ...
        'MarkerEdgeColor', first4_colors(k,:), ...
        'LineWidth', 1.6, ...
        'DisplayName', sprintf('Network #%d', k));
end

% ---- Plot resten i grå (STJERNER, lidt mindre) ----
h_rest = gobjects(0);
if ~isempty(idx_rest)
    h_rest = scatter(idx_rest, var_exp(idx_rest), 80, '*', ...
        'MarkerEdgeColor', rest_gray, ...
        'LineWidth', 1.4, ...
        'DisplayName', 'Other networks');
end

% ---- Skriftstørrelser (EN tand mindre overalt) ----
base_fs     = 10;   % tick labels + legend
label_fs    = 10;   % akse-titler
title_fs    = 12;   % figur-titel

% ---- Akser, titel, style ----
ax = gca;
set(gcf, 'Color', 'w');
ax.LineWidth = 0.4;
ax.FontSize  = base_fs;  % styrer tick-labels m.m.

xlabel('PCA components (brain networks)', 'FontWeight','bold', 'FontSize', label_fs);
ylabel('Variance explained (%)',           'FontWeight','bold', 'FontSize', label_fs);
title('Variance explained by Principal Components', 'FontSize', title_fs);

% Giv x-aksen lidt luft til venstre
xlim([0 20]);
set(gca,'XTick',1:20);

% Lidt pæn margin i aksens indre område
set(gca,'LooseInset', max(get(gca,'TightInset'), 0.02*[1 1 1 1]));

% ---- Grid/box som før ----
%grid minor; 
box on;



% ---- Legend (uden firkant og lidt mindre font) ----
legend_handles = [p_curve; h_first(:); h_rest(:)];
legend_handles = legend_handles(isgraphics(legend_handles));  % filtrér ugyldige

lgd = legend(legend_handles, 'Location', 'northeast');
lgd.Box = 'off';           % fjern firkant
lgd.Color = 'w';
lgd.FontSize = base_fs;    % mindre legend-tekst

% ---- Export figure ----
if export_l == 1
    outdir = '/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/outputs/figures_image';
    if ~exist(outdir,'dir'), mkdir(outdir); end

    set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 19 11],'PaperSize',[19 11]);
    exportgraphics(gcf, fullfile(outdir, 'PC_variance_explained_sign.pdf'), 'Resolution', 300);
end





%% 4B. TIME SERIES FOR EACH PC, ALL CONDITIONS, WITH SIGNIFICANT CONTRASTS

addpath('/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/3.0_BROADNESS_MEG_AuditoryRecognition-main/BROADNESS_Toolbox')
outputdir = '/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/outputs/statistics_image';
figdir = '/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/outputs/figures_image';

load(fullfile(outputdir, 'posthoc_withholm_2_run.mat'));  % hb_results = ttest_cluster_results
load(fullfile(outputdir, 'timeseries_4D.mat'));     % J
load('/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/image_data/time/time.mat');

hb_results = ttest_cluster_results;  % Holm-corrected cluster structs

% Each row is [ref test]
contrasts = [1 3; 1 2; 3 2]; % same as for ttests

% Colours for plots
% CONDITIONS (waveform)
masked1_color = [0.60, 0.03, 0.15];   % Oxblood red
unmasked_color = [0.00, 0.00, 0.00];  % Black
masked2_color = [0.00, 0.12, 0.60];   % Deep blue
% CONTRASTS (line / sig window)
contrast1_purple = [0.55, 0.05, 0.60]; % Purple (between oxblood & deep blue)
contrast2_dark   = [0.40, 0.06, 0.08]; % Dark reddish (between oxblood & black)
contrast3_navy   = [0.03, 0.18, 0.50]; % Dark navy (between deep blue & black)

% Explicit contrast colors (RGB)
% !! Reversed because order is reversed in plot !!
% In plot, Masked1 vs Masked2 is the line closest to waveform, 
% Masked1 vs Unmasked is in the middle,
% Masked2 vs Unmasked is at the bottom, closest to the x-axis
contrast_colors = [ ...
    contrast3_navy; ...    % Masked2 vs Unmasked
    contrast2_dark; ...    % Masked1 vs Unmasked
    contrast1_purple];     % Masked1 vs Masked2

PCs = 4;        % number of PCs to plot
export_l = 0;   % export figures? 1=yes, 0=no
col_l = 1;      % show colored significant windows? 1=yes, 0=no
ylimm = [];     % auto y limits

% Prepare data: PC x time x subject x condition
data = permute(J,[2 1 4 3]);

close all
for pc = 1:PCs
    
    % Build waveplot input
    S = struct();
    S.ii = pc; % assign figure numbers so that each PC goes into its own figure
    S.data = data(pc,:,:,:);
    S.condition_n = 1:3; % which conditions? Masked1, Unmasked, Masked2
    S.conds = {'Degraded1','ClearCue','Degraded2'}; % names of conditions for labels
    S.time_real = time(:); % x-axis time vector (length = number of timepoints)
    S.STE = 2; % show standard error shading (shadow)
    S.transp = 0.2; % transparency of shading
    S.colorline = [   % waveform colors, order must correspond to the order of S.condition_n
        masked1_color;    % oxblood red
        unmasked_color;   % black
        masked2_color];   % deep blue
    S.ROI_n = 1;
    S.ROIs_labels = {1};
    S.groups = {1};
    S.gsubj = {1:size(data,3)};
    S.x_lim = [-0.1 3.4];
    S.y_lim = ylimm;
    S.legendl = 0;       % waveplot should NOT draw legend, call the legend after the waveplot (because waveplot overwrites handles when drawing significance)
    S.lineplot = [20 1]; % lines instead of shaded patches, 20: vertical spacing of lines, 1: orientation (top vs bottom)

    % -------- Build signtp (significant time windows) reversed order -----------------
    sig_intervals = {};
    sig_cols = [];
    numContrasts = size(contrasts,1);
    for ci = 1:numContrasts
        clusters_ci = hb_results{pc, ci};
        if isempty(clusters_ci), continue; end
        for k = 1:numel(clusters_ci)
            % unified interval field
            if isfield(clusters_ci(k),'time_interval')
                interval = clusters_ci(k).time_interval;
            else
                interval = clusters_ci(k).time;
            end
            % store time window
            sig_intervals{end+1} = interval;

            % *** REVERSED ORDERING ***
            % ci = 1 → highest line
            % ci = 2 → middle line
            % ci = 3 → lowest line
            sig_cols(end+1) = (numContrasts - ci + 1);
        end
    end
    
    S.signtp = sig_intervals;

    if col_l && ~isempty(sig_cols)
        S.signtp_col = sig_cols;
        S.colorsign = contrast_colors;  % contrast color palette
    else
        S.signtp_col = [];
        S.colorsign = {};
    end

    % Run the waveform plotter
    waveplot_groups_local_v2(S);

    % Add legends
    main_ax = gca; hold(main_ax,'on');
    
    % 1) Condition legend (top-right) — dummy handles drawn on MAIN axes
    nCond = numel(S.conds);
    cond_handles = gobjects(nCond,1);
    for ic = 1:nCond
        col = S.colorline(min(ic,size(S.colorline,1)),:);
        cond_handles(ic) = plot(main_ax, NaN, NaN, '-', 'Color', col, 'LineWidth', 1.5);
    end
    legend(main_ax, cond_handles, S.conds, 'Location','northeast', 'Box','off');
    
    % 2) Contrast legend (bottom) — create invisible axes FIRST, then plot dummy lines INTO it
    % Position: [left, bottom, width, height] (normalized). Tune bottom/height to avoid overlapping x-label.
    ax2_pos = [0.10, -0.01, 0.80, 0.06];   % left, bottom, width, height
    ax2 = axes('Position', ax2_pos, 'Visible','off'); hold(ax2,'on');
    
    % create dummy contrast lines IN THE ORDER you want (left→right)
    % Here we make the legend left->right: Contrast1, Contrast2, Contrast3
    h2 = gobjects(3,1);
    h2(1) = plot(ax2, NaN, NaN, '-', 'Color', contrast_colors(3,:), 'LineWidth', 3); % Contrast 1 (left)
    h2(2) = plot(ax2, NaN, NaN, '-', 'Color', contrast_colors(2,:), 'LineWidth', 3); % Contrast 2 (middle)
    h2(3) = plot(ax2, NaN, NaN, '-', 'Color', contrast_colors(1,:), 'LineWidth', 3); % Contrast 3 (right)
    
    % Legend labels in the same order -> they will appear left-to-right exactly as specified
    legend(ax2, h2, { 'Degraded1 vs Degraded2', 'Degraded1 vs ClearCue', 'Degraded2 vs ClearCue' }, ...
           'Orientation','horizontal', 'Box','off', 'Location','north');

    % Hide axes decorations to be safe
    ax2.XAxis.Visible = 'off';
    ax2.YAxis.Visible = 'off';
    
    % Return drawing focus to the main axes
    axes(main_ax);

    % Title and axis labels
    title(sprintf('PC %d – Time series for all conditions', pc), 'FontSize', 14, 'FontWeight','bold');
    xlabel('Time (s)', 'FontSize', 12);
    ylabel('Amplitude', 'FontSize', 12);

    % Export
    if export_l
        outname = fullfile(figdir, sprintf('PC%d_conditions_with_significance_3_run.pdf', pc));
        exportgraphics(gcf, outname, 'Resolution',300);
    end
end

%% 4C. TIME SERIES FOR EACH CONDITION, ALL PCs IN EACH FIGURE

addpath('/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/3.0_BROADNESS_MEG_AuditoryRecognition-main/BROADNESS_Toolbox')
outputdir = '/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/outputs/statistics_image';
figdir = '/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/outputs/figures_image';

load(fullfile(outputdir, 'timeseries_4D.mat')); % J
load('/Users/sisselhoejgaard/Desktop/5_semster/Bachelor/Data/image_data/time/time.mat');

conds = 3;                 % Masked1, Unmasked, Masked2
export_l = 1;
ylimm = [-1500 1000];

% Data shape: condition x time x subjects x PCs
data = permute(J,[3 1 4 2]);

% Stable PC line colors (same everywhere)
pc_colors = [
    0.60, 0.03, 0.15; % PC1 = oxblood red 
    0.00, 0.12, 0.60; % PC2 = dark blue 
    0.00, 0.60, 0.20; % PC3 – greenish 
    0.80, 0.40, 0.00; % PC4 – orange
    ];

close all
for cond_i = 1:conds

    S = struct();
    S.ii = cond_i;
    S.data = data(cond_i,:,:,:);  % ROI x time x subj x PCs
    S.condition_n = 1:4;          % PC1, PC2
    S.conds = {'Network #1','Network #2', 'Network #3','Network #4' };
    S.ROI_n = 1;
    S.ROIs_labels = {1};
    S.time_real = time(:);
    S.STE = 2;
    S.transp = 0.3;
    S.colorline = pc_colors;      % stable PC colors
    S.groups = {1};
    S.gsubj = {1:size(data,3)};
    S.x_lim = [-0.1 3.4];
    S.y_lim = ylimm;
    S.lineplot = [];
    S.signtp = {};
    S.signtp_col = [];
    S.legendl = 0;

    waveplot_groups_local_v2(S);

    % Ensure grid is visible
    main_ax = gca;
    set(main_ax, 'XMinorGrid','on', 'YMinorGrid','on', 'Layer','top');
    grid minor; box on;

    % Labels
    xlabel('Time (s)', 'FontSize',12,'FontWeight','bold');
    ylabel('Amplitude', 'FontSize',12,'FontWeight','bold');
    conditionNames = {'Masked1','Unmasked','Masked2'};
    title([conditionNames{cond_i} ' — Time Series of PCs'], ...
        'FontSize',13,'FontWeight','bold');

    % Legend for PCs
    legend(S.conds, 'Location','northeast', 'Box','off','FontSize',11);
    grid minor; box on;

    % Export
    if export_l
        outname = fullfile(figdir, sprintf('Condition%d_PC_timeseries.pdf', cond_i));
        exportgraphics(gcf, outname,'Resolution',300);
    end

end

%% 4D. THRESHOLDED BRAIN IMAGES PER PC

% Elisa has info about how to use visualise the output files using Freesurfer

% Need dum = (3D, nSources x nTime x nConditions), sign adjusted average across participants
% Need OUT.W / act_patt = spatial weights (eigenvectors) for each PC
%dum = BROADNESS.OriginalData;
%OUT.W = BROADNESS.ActivationPatterns_BrainNetworks;
load('/Users/verarudi/Projects/BROADNESS/BROADNESS_Output/statistics_image/for_thresholded_brain_images.mat');
% PCs = 2 % if not still in workspace

path_home = '/Users/verarudi/Projects/BROADNESS/BROADNESS_Toolbox';
maskk = load_nii([path_home '/BROADNESS_External/MNI152_8mm_brain_diy.nii.gz']); %getting the mask for creating the figure
OUT.W = act_patt;

% Plotting weights in the brain (nifti images)
dum_averaged = mean(dum(:,:,1:3),3); %averaged across conditions
C = cov(dum_averaged');

for ii = 1:PCs(end) %over significant PCs
    wcoeff(:,ii) = OUT.W(:,ii)'*C;
    threshold = mean(abs(wcoeff(:,ii))) + std(abs(wcoeff(:,ii)));
    SO = zeros(size(wcoeff,1),1);
    for jj = 1:size(wcoeff,1)
        if abs(wcoeff(jj,ii)) > threshold
            SO(jj) = wcoeff(jj,ii);
        else
            SO(jj) = 0;
        end
    end
    % Building nifti image
    SS = size(maskk.img);
    dumimg = zeros(SS(1),SS(2),SS(3),1); %no time here so size of 4D = 1
    for jj = 1:size(SO,1) %over brain sources
        dumm = find(maskk.img == jj); %finding index of sources ii in mask image (MNI152_8mm_brain_diy.nii.gz)
        [i1,i2,i3] = ind2sub([SS(1),SS(2),SS(3)],dumm); %getting subscript in 3D from index
        dumimg(i1,i2,i3,:) = SO(jj,:); %storing values for all time-points in the image matrix
    end
    nii = make_nii(dumimg,[8 8 8]); %making nifti image from 3D data matrix (and specifying the 8 mm of resolution)
    nii.img = dumimg; %storing matrix within image structure
    nii.hdr.hist = maskk.hdr.hist; %copying some information from maskk
    disp(['saving nifti image - component ' num2str(ii)])
    save_nii(nii,[outputdir '/thresholded_actpattern_PCAmaineffect_PC_' num2str(ii) '.nii']); %printing image
end
