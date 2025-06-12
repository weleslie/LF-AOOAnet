clear all;
clc;
close all;

angRes = 5; % Angular resolution of LFs in ./results and ./input
factor = 4; % upscaling factor
% resultsFolder = ['./Results_', num2str(factor), 'x_AFO/'];
% resultsFolder = ['D:/gitclone/LF-AOOAnet-ECCVversion/Results_', num2str(factor), 'x_AFO/'];
resultsFolder = ['D:/gitclone/DPT-main/DPT-main/code/Results_', num2str(factor), 'x_LFT/'];
inputFolder = ['E:/Light field dataset/TestData_5x5_', num2str(factor), 'xSR/'];

% resultsFolder = 'D:/gitclone/LF-DFnet/code/Results_2x_C2R_DF_ablation/';
% inputFolder = ['E:/Light field dataset/SubtleTestData_5x5_', ...
%     num2str(factor), 'xSR_RGB_synthetic/'];

savePath = './SRimages/SRimage_4x_LFT/'; % Super-resolved LF images (.png) will be saved here 
 
scenes = dir(resultsFolder);
scenes(1:2) = [];

sceneNum  = 25;
% sceneNum = 8;
% inputs = dir(inputFolder);
% inputs(1:2) = [];

p = zeros(sceneNum, 1);
s = zeros(sceneNum, 1);

ps = zeros(5, 5);
ss = zeros(5, 5);

for idxScene = 1:sceneNum
% for idxScene = 12
% for idxScene = 16
% for idxScene = 2
% for idxScene = 22
% for idxScene = 25
    sceneName = [num2str(idxScene), '_result.mat'];
    sceneName(end-3:end) = [];
    data = load([resultsFolder, num2str(idxScene), '_results.mat']);
%     data = load([resultsFolder, num2str(idxScene), '_results_trans_32.mat']);
    gt = load([inputFolder, num2str(idxScene), '.mat']);
    LFgt = double(gt.label);    
    [angRes, ~, H, W, ~] = size(LFgt);    
    LFsr = zeros(size(LFgt));
    LFb = zeros(size(LFgt));
    
    for u = 1 : angRes
        for v = 1 : angRes
            imgHR_rgb = squeeze(LFgt(u, v, :, :, :));
            imgLR_rgb = imresize(imgHR_rgb, 1 / factor);
            imgLR_ycbcr = rgb2ycbcr(imgLR_rgb);
            imgSR_ycbcr = imresize(imgLR_ycbcr, factor);    
            
            imgB_rgb = ycbcr2rgb(imgSR_ycbcr);
            LFb(u, v, :, :, :) = imgB_rgb;
               
            if length(size(data.LF)) == 4
                imgSR_ycbcr(:, :, 1) = data.LF(1, 1, (u - 1) * H + 1:u * H, (v - 1) * W + 1:v * W);
            else
                imgSR_ycbcr(:, :, 1) = data.LF((u - 1) * H + 1:u * H, (v - 1) * W + 1:v * W);
            end
            imgSR_rgb = ycbcr2rgb(imgSR_ycbcr);
            LFsr(u, v, :, :, :) = imgSR_rgb;   
            
            if exist([savePath, sceneName], 'dir')==0
                mkdir([savePath, sceneName]);
            end
            imwrite(uint8(255 * imgSR_rgb), [savePath, sceneName, '/',...
                num2str(u, '%02d'), '_', num2str(v, '%02d'), '.png' ]);

        end
    end
    % Calculate PSNR and SSIM values of each view
    boundary = 0; % Crop image boundaries for evaluation
    [PSNR, SSIM] = cal_metrics(LFgt, LFsr, boundary); 
    
    ps = ps + PSNR;
    ss = ss + SSIM;
    
    p(idxScene) = mean(PSNR(:));
%     p(idxScene) = PSNR(3, 3);
    s(idxScene) = mean(SSIM(:));
    
%     [PSNR, SSIM] = cal_metrics(LFgt, LFb, boundary);  
%     pb(idxScene) = mean(PSNR(:));
%     sb(idxScene) = mean(SSIM(:));
    % Maximum, average, and minimum scores are reported
    fprintf([sceneName, ': maxPSNR=%.2f; avgPSNR=%.2f; minPSNR=%.2f; ' ...
        'maxSSIM=%.3f; avgSSIM=%.3f; minSSIM=%.3f; \n'],...
            max(PSNR(:)), mean(PSNR(:)), min(PSNR(:)),...
            max(SSIM(:)), mean(SSIM(:)), min(SSIM(:)));        
end

% mean(p(1:15))
% mean(p(16:20))
% mean(p(21:25))
% 
% mean(s(1:15))
% mean(s(16:20))
% mean(s(21:25))

% mean(pb(1:15))
% mean(pb(16:20))
% mean(pb(21:25))
% 
% mean(sb(1:15))
% mean(sb(16:20))
% mean(sb(21:25))

%                           PSNR(1-15)    SSIM(1-15)     PSNR(16-20)    SSIM(16-20)     PSNR(21-25)   SSIM(21-25)
% (Ablation Study)
% (Raw data)
% 20_DF_1_color      35.7580           0.9733             37.0372            0.9817             37.8344          0.9785
% 20_DF_1_raw        35.6599           0.9775             36.4715            0.9848             37.9082          0.9827


% (MPI module)
% 20_DF_1_2D          36.8859           0.9787             38.2597            0.9857             39.0148          0.9838
% 20_DF_1_SISR       35.1447           0.9709             36.1657            0.9818             36.8794          0.9780
% 20_DF_1_sele        36.6343           0.9784             38.2257            0.9856             38.9520          0.9836
% 20_DF_1_tower3   36.8853           0.9788             38.3377            0.9857             39.0534          0.9839
% 20_DF_1_tower9   36.9089           0.9788             38.2777            0.9858             39.0566          0.9839

% (DF module)
% 20_DF_1_ssim       36.8159           0.9791             38.3854            0.9859             39.1320          0.9840
% 20_DF_2_ssim       36.8142           0.9792             38.4263            0.9860             39.1376          0.9841
% 20_DF_3_ssim       36.8683           0.9794             38.4999            0.9861             39.1942          0.9842
% 20_no_DC_ssim    36.4303           0.9774             37.9049            0.9847             38.6993          0.9830

% (Loss function of SSIM)
% 20_DF_1                36.8365           0.9777             38.2149            0.9851             38.9692          0.9833

% (Comparisons 2x)
% 50_DF_color       35.3295           0.9712             36.3269            0.9788             37.2717           0.9771
% 50_DF_raw         36.2048           0.9768             37.5085            0.9859             38.1209           0.9822

% 40_Inter_color    35.0573           0.9719             35.9396            0.9816             36.9106           0.9775
% 40_Inter_raw      35.6560           0.9746             36.6670            0.9843             37.3948           0.9801

% (Comparison 4x)
% 20_DF_1_ssim     31.4205           0.9344             32.6932            0.9588             33.1674          0.9498

% 50_DF_raw          30.6655           0.9248             31.8237            0.9547             32.1270          0.9414
% 50_DF_color        30.4434           0.9225             31.5756            0.9521             31.8977          0.9384

% 50_Inter_raw       30.7723           0.9230             31.9649            0.9524             32.0638          0.9389
% 50_Inter_color     30.5554           0.9209             31.8105            0.9511             31.9123          0.9368








