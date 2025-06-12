factor = 4;
file_path = ['E:\Light field dataset\TestData_5x5_',...
    num2str(factor), 'xSR\'];
file_dir = dir(file_path);
file_dir(1:2) = [];

for i = 1:length(file_dir)
    load([file_path, file_dir(i).name]);
    
    save(['E:\Light field dataset\SubtleTestData_5x5_', num2str(factor), 'xSR\', file_dir(i).name], ...
        '-v7.3', 'label', 'data', 'raw');
end