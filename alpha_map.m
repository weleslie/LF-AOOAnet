%% alpha map
load('D:\gitclone\LF-AOOAnet\Results_alpha_map\3_2D_alpha_map.mat');
m = squeeze(mask);

for i = 1:6
    m_temp = squeeze(m(i, :, :));
    figure(i); imagesc(m_temp, [0, 1]);
    axis off
    colormap(jet(10))
    colorbar('FontSize', 20, 'FontName', 'Elephant')
	saveas(gcf, ['2D_alpha_', num2str(i), '.png']);
end

%% color map
load('D:\gitclone\LF-AOOAnet\Results_alpha_map\3_3D_color_map.mat');
m = squeeze(mask);

for i = 1:6
    m_temp = squeeze(m(i, :, :));
    figure(i); imshow(m_temp, [])
	saveas(gcf, ['3D_color_', num2str(i), '.png']);
end

%% summation
load('D:\gitclone\LF-AOOAnet\Results_alpha_map\3_3D_alpha_map.mat');
m = squeeze(mask);
load('D:\gitclone\LF-AOOAnet\Results_alpha_map\3_3D_color_map.mat');
c = squeeze(mask);

%% 
load('D:\gitclone\LF-AOOAnet\Results_alpha_map\3_3D_out.mat');
out = squeeze(LF);
figure; imshow(out)
saveas(gcf, '3D_out_.png');