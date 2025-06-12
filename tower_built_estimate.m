clear all;
clc;
close all;

angRes = 5; % Angular resolution of LFs in ./results and ./input
factor = 2; % upscaling factor
resultsFolder = './tower_built/';
inputFolder = ['E:/Light field dataset/TestData_5x5_', ...
    num2str(factor), 'xSR/'];
savePath = './SRimage/'; % Super-resolved LF images (.png) will be saved here 

sceneNum = 25;
inputs = dir(inputFolder);
inputs(1:2) = [];

for idxScene = 2:sceneNum
    data = load([resultsFolder, num2str(idxScene), '_result.mat']);
    raw = squeeze(data.LF);
    gt = load([inputFolder, num2str(idxScene), '.mat']);
    LFgt = gt.raw_refocus;
    LFgt = LFgt(1:2:end, :, :, :);
    
    for k = 1:size(LFgt, 1)
        for u = 1 : size(LFgt, 2)
            imgHR_rgb = squeeze(LFgt(k, u, 10:end-10, 10:end-10));
            imgSR_rgb = squeeze(raw(k, u, 10:end-10, 10:end-10));
            
            err = abs(imgHR_rgb - imgSR_rgb);
            figure(1); imshow(imgHR_rgb);
            figure(2); imshow(imgSR_rgb);
            
        end
    end   
    
end