%please add all scripts and functions in folder code and folder utilities
%to your path.
%% bhv analysis 
foldername = '*\matlab_data\bhvdata'; %folder containing demo data.
cd(foldername);
bhv_ana;

%% state space analysis on Monkey 1(Fig2 and 4).
foldername = '*\matlab_data\neurondata\';%folder containing demo data.
filename = 'M1_len3.mat';% data name
mainlen3;
%% state space analysis on Monkey 2 length-2 trials.
foldername = '*\matlab_data\neurondata\';%folder containing demo data.
filename = 'M2_len2.mat';%data name
mainlen2;
%% anatomical organization analysis. (Fig 5)
foldername = '*\matlab_data\neurondata\';
cd(foldername);
C_aligns;% calculating cluster index for neural alignments in each rank.
C_tuning;% calculating cluster index for spatial tuning curves in each rank