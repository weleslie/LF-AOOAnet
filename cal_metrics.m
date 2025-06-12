%% Functions
function [PSNR, SSIM] = cal_metrics(LF, LFout, boundary)
[U, V, H, W, ~] = size(LF);
PSNR = zeros(U, V);
SSIM = zeros(U, V);
for u = 1 : U
    for v = 1 : V
        Ir = squeeze(LFout(u, v, boundary+1:end-boundary, boundary+1:end-boundary, :));
        Is = squeeze(LF(u, v, boundary+1:end-boundary, boundary+1:end-boundary, :));
        Ir_ycbcr = rgb2ycbcr(Ir);
        Ir_y = Ir_ycbcr(:,:,1);
        Is_ycbcr = rgb2ycbcr(Is);
        Is_y = Is_ycbcr(:,:,1);
        temp = (Ir_y-Is_y).^2;
        mse = sum(temp(:))/(H*W);
        PSNR(u,v) = 10*log10(1/mse);
        SSIM(u,v) = cal_ssim(Ir_y, Is_y, 0, 0);
    end
end
end

function ssim = cal_ssim( im1, im2, b_row, b_col)

[h, w, ch] = size( im1 );
ssim = 0;
if (ch == 1)
    ssim = ssim_index ( im1(b_row+1:h-b_row, b_col+1:w-b_col), im2(b_row+1:h-b_row,b_col+1:w-b_col));
else
    for i = 1:ch
        ssim = ssim + ssim_index ( im1(b_row+1:h-b_row, b_col+1:w-b_col, i), im2(b_row+1:h-b_row,b_col+1:w-b_col, i));
    end
    ssim = ssim/3;
end
end

function [mssim, ssim_map] = ssim_index(img1, img2, K, window, L)

if (nargin < 2 || nargin > 5)
    mssim = -Inf;
    ssim_map = -Inf;
    return;
end

if (size(img1) ~= size(img2))
    mssim = -Inf;
    ssim_map = -Inf;
    return;
end

[M N] = size(img1);

if (nargin == 2)
    if ((M < 11) || (N < 11))
        mssim = -Inf;
        ssim_map = -Inf;
        return
    end
    window = fspecial('gaussian', 11, 1.5);	%
    K(1) = 0.01;					% default settings
    K(2) = 0.03;					%
    L = 2;                                     %
end

img1 = double(img1);
img2 = double(img2);

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));

mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;

if (C1 > 0 && C2 > 0)
    ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
else
    numerator1 = 2*mu1_mu2 + C1;
    numerator2 = 2*sigma12 + C2;
    denominator1 = mu1_sq + mu2_sq + C1;
    denominator2 = sigma1_sq + sigma2_sq + C2;
    ssim_map = ones(size(mu1));
    index = (denominator1.*denominator2 > 0);
    ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
    index = (denominator1 ~= 0) & (denominator2 == 0);
    ssim_map(index) = numerator1(index)./denominator1(index);
end

mssim = mean2(ssim_map);

end