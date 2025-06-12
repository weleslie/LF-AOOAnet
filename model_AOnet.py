import torch
import torch.nn as nn
import torch.nn.functional as F
from volume_transformer import PositionEncoding, VolumeTrans
from einops import rearrange

## MPI module
class Net(nn.Module):
    def __init__(self, angRes=5, factor=2, mpi_channel=25):
        super(Net, self).__init__()
        ato_channel = 64
        self.factor = factor
        self.angRes = angRes
        self.ato = ATO(self.factor, mpi_channel, ato_channel)
        self.beta = SISR(self.factor, 1, 32)

    def forward(self, hybrid, x, x_refocus, state):
        _, hybrid_cv = LFsplit(hybrid, self.angRes, hybrid.shape[1])
        hybrid_upscale = F.interpolate(hybrid, scale_factor=self.factor, mode='bicubic', align_corners=False)
        _, hybrid_cv_up = LFsplit(hybrid_upscale, self.angRes, hybrid_upscale.shape[1])

        x_upscale = F.interpolate(x, scale_factor=self.factor, mode='bicubic', align_corners=False)
        _, x_cv_up = LFsplit(x_upscale, self.angRes, x_upscale.shape[1])

        out_c, mask, color = self.ato(x_refocus, x_cv_up)

        hybrid_res = hybrid_cv[:, 0:1, :, :] - hybrid_cv[:, 1:2, :, :]
        hybrid_res_up = hybrid_cv_up[:, 0:1, :, :] - hybrid_cv_up[:, 1:2, :, :]
        hybrid_res = hybrid_res.unsqueeze(1)

        ## with SISR in Luminance Compensation
        hybrid_res_up = hybrid_res_up.unsqueeze(1)
        factor = self.beta(hybrid_res, hybrid_res_up)
        factor = factor.squeeze(1)
        ## without SISR in Luminance Compensation
        # factor = hybrid_res_up

        out_c = out_c + factor

        if state == 'train AO':
            return out_c
        elif state == 'test AO':
            return out_c, mask, color
            # return out_c


## SPI module (Ablation study)
# class Net(nn.Module):
#     def __init__(self, angRes=5, factor=2, spi_channel=1, mpi_channel=25):
#         super(Net, self).__init__()
#         self.factor = factor
#         self.angRes = angRes
#         self.ato = SISR(self.factor, spi_channel, 64)
#
#     def forward(self, hybrid, x, state):
#         _, hybrid_cv = LFsplit(hybrid, self.angRes, hybrid.shape[1])
#         hybrid_upscale = F.interpolate(hybrid, scale_factor=self.factor, mode='bicubic', align_corners=False)
#         _, hybrid_cv_up = LFsplit(hybrid_upscale, self.angRes, hybrid_upscale.shape[1])
#
#         x_upscale = F.interpolate(x, scale_factor=self.factor, mode='bicubic', align_corners=False)
#         _, x_cv = LFsplit(x, self.angRes, x.shape[1])
#         _, x_cv_up = LFsplit(x_upscale, self.angRes, x_upscale.shape[1])
#         x_cv = x_cv.unsqueeze(1)
#         x_cv_up = x_cv_up.unsqueeze(1)
#
#         out_c = self.ato(x_cv, x_cv_up)
#         out_c = out_c.squeeze(1)
#
#         hybrid_res = hybrid_cv[:, 0:1, :, :] - hybrid_cv[:, 1:2, :, :]
#         hybrid_res_up = hybrid_cv_up[:, 0:1, :, :] - hybrid_cv_up[:, 1:2, :, :]
#         hybrid_res = hybrid_res.unsqueeze(1)
#         hybrid_res_up = hybrid_res_up.unsqueeze(1)
#         factor = self.ato(hybrid_res, hybrid_res_up)
#         factor = factor.squeeze(1)
#
#         out_c = out_c + factor
#
#         return out_c


class ATO(nn.Module):
    def __init__(self, factor, in_channel, out_channel):
        super(ATO, self).__init__()
        self.factor = factor
        self.out_channel = out_channel
        self.FeaExtract = FeaExtract(in_channel, self.out_channel)
        self.Recons = RDG(G0=self.out_channel, C=4, G=32, n_RDB=4)

        # self.pos_encoding = PositionEncoding(temperature=10000)
        # self.MHSA_params = {}
        # self.MHSA_params['num_heads'] = 8
        # self.MHSA_params['dropout'] = 0.
        # self.layer_transformer1 = VolumeTrans(self.out_channel, self.MHSA_params)
        # self.layer_transformer2 = VolumeTrans(self.out_channel, self.MHSA_params)
        # self.layer_transformer3 = VolumeTrans(self.out_channel, self.MHSA_params)
        # self.layer_transformer4 = VolumeTrans(self.out_channel, self.MHSA_params)

        self.upsample = Upsample_MPI(self.out_channel, self.factor)

    def forward(self, x_refocus, x_upscale):
        b, n, c, h, w = x_refocus.shape

        ## Without MPI representation
        # x_refocus = x_refocus.contiguous().view(b, -1, n*c, h, w)

        x = self.FeaExtract(x_refocus)
        x = self.Recons(x)

        # ## w/ transformer
        # x = rearrange(x, '(b n) c h w -> b c n h w', b=b)
        # ang_position = self.pos_encoding(x, dim=[2], token_dim=self.out_channel)
        # x = self.layer_transformer1(x, ang_position)
        # x = self.layer_transformer2(x, ang_position)
        # x = self.layer_transformer3(x, ang_position)
        # x = self.layer_transformer4(x, ang_position)
        # x = rearrange(x, 'b c n h w -> (b n) c h w')

        out, mask, color, _ = self.upsample(x, b, n)

        out = out + x_upscale

        return out, mask, color


class SISR(nn.Module):
    def __init__(self, factor, in_channel, out_channel):
        super(SISR, self).__init__()
        self.factor = factor
        self.out_channel = out_channel
        self.FeaExtract = FeaExtract(in_channel, self.out_channel)
        self.Recons = RDG(G0=self.out_channel, C=4, G=24, n_RDB=4)
        self.upsample = Upsample_SPI(self.out_channel, self.factor)

    def forward(self, x_sv, x_sv_upscale):
        b, n, _, _, _ = x_sv.shape
        x = self.FeaExtract(x_sv)
        x1 = self.Recons(x)
        out = self.upsample(x1, b, n)

        out = out + x_sv_upscale

        return out


class one_conv(nn.Module):
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)


class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x


class RDG(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
        return out


class Upsample_SPI(nn.Module):
    def __init__(self, channel, factor):
        super(Upsample_SPI, self).__init__()
        self.upsp = nn.Sequential(
            nn.Conv2d(channel, channel * factor * factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(factor),
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x, b, n):
        out = self.upsp(x)
        _, _, H, W = out.shape
        out = out.contiguous().view(b, n, -1, H, W)

        return out


class Upsample_MPI(nn.Module):
    def __init__(self, channel, factor):
        super(Upsample_MPI, self).__init__()

        # self.upsp_color_1 = nn.Sequential(
        #     nn.Conv2d(channel, channel * factor * factor, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.PixelShuffle(factor)
        # )
        self.upsp_color_2 = nn.Sequential(
            nn.Conv2d(6*channel, channel, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 1, 1, 0, bias=False)
        )

        ## 2D convolution
        # self.upsp_alpha_1 = nn.Sequential(
        #     nn.Conv2d(channel, channel * factor * factor, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.PixelShuffle(factor)
        # )
        # self.upsp_alpha_2 = nn.Sequential(
        #     nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(channel, 1, 1, 1, 0, bias=False)
        # )

        self.upsp = nn.Sequential(
            nn.Conv2d(channel, channel * factor * factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(factor),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )
        ## 3D convolution
        # self.upsp_alpha_1 = nn.Sequential(
        #     nn.Conv2d(channel, channel * factor * factor, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.PixelShuffle(factor))
        # self.upsp_alpha_2 = nn.Sequential(
        #     nn.Conv3d(channel, channel, 3, 1, 1, bias=False),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv3d(channel, channel, 3, 1, 1, bias=False),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv3d(channel, channel, 3, 1, 1, bias=False),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv3d(channel, 1, 1, 1, 0, bias=False)
        # )

        self.softmax = nn.Softmax(1)

    def forward(self, x, b, n):
        # color = self.upsp_color_1(x)
        # _, _, H, W = color.shape
        # ## 2D convolution
        # color = self.upsp_color_2(color)
        # color = color.contiguous().view(b, n, -1, H, W)
        # ## 3D convolution
        # # color = color.contiguous().view(b, n, -1, H, W).permute(0, 2, 1, 3, 4)
        # # color = self.upsp_color_2(color).permute(0, 2, 1, 3, 4)
        #
        # ## 3D convolution
        # # mask = self.upsp_alpha_1(x)
        # # mask = mask.contiguous().view(b, n, -1, H, W).permute(0, 2, 1, 3, 4)
        # # mask = self.upsp_alpha_2(mask)
        # # mask = mask.squeeze(1)
        # ## 2D convolution
        # mask = self.upsp_alpha_1(x)
        # mask = self.upsp_alpha_2(mask)
        #
        # mask = mask.contiguous().view(b, -1, H * W)
        # mask = self.softmax(mask)
        # mask = mask.contiguous().view(b, n, -1, H, W)
        #
        # out = torch.sum(color * mask, dim=1)
        #
        # return out, mask, color
        #
        # ## Without MPI representation
        # # color = self.upsp_color_1(x)
        # # color = self.upsp_color_2(color)
        # # return color, None, None

        _, c, H, W = x.shape
        x = x.contiguous().view(b, n*c, H, W)
        color = self.upsp_color_2(x)
        out = self.upsp(color)

        return out, out, color, out


class FeaExtract(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeaExtract, self).__init__()
        self.FEconv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.FERB_1 = ResASPP(out_channel)
        self.FERB_2 = RB(out_channel)
        self.FERB_3 = ResASPP(out_channel)
        self.FERB_4 = RB(out_channel)

    def forward(self, x):
        b, n, _, h, w = x.shape
        x = x.contiguous().view(b*n, -1, h, w)
        buffer = self.FEconv(x)
        buffer = self.FERB_1(buffer)
        buffer = self.FERB_2(buffer)
        buffer = self.FERB_3(buffer)
        buffer = self.FERB_4(buffer)

        return buffer


class RB(nn.Module):
    def __init__(self, channel):
        super(RB, self).__init__()
        self.conv01 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv02 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        buffer = self.conv01(x)
        buffer = self.lrelu(buffer)
        buffer = self.conv02(buffer)
        return buffer + x


class ResASPP(nn.Module):
    def __init__(self, channel):
        super(ResASPP, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1,
                                              dilation=1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=2,
                                              dilation=2, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=4,
                                              dilation=4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_t = nn.Conv2d(channel*3, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv_1(x))
        buffer_1.append(self.conv_2(x))
        buffer_1.append(self.conv_3(x))
        buffer_1 = self.conv_t(torch.cat(buffer_1, 1))
        return x + buffer_1


def LFsplit(data, angRes, channel):
    b, _, H, W = data.shape
    h = int(H / angRes)
    w = int(W / angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            k = u*angRes + v
            if k == (angRes*angRes - 1)/2:
                data_cv = data[:, :, u * h:(u + 1) * h, v * w:(v + 1) * w]

            temp = data[:, :, u*h:(u+1)*h, v*w:(v+1)*w].contiguous().view(b, -1, channel, h, w)
            data_sv.append(temp)

    data_sv = torch.cat(data_sv, dim=1)
    return data_sv, data_cv


if __name__ == "__main__":
    net = Net(5, 4).cuda()
    from thop import profile
    input1 = torch.randn(1, 2, 160, 160).cuda()
    input2 = torch.randn(1, 1, 160, 160).cuda()
    input3 = torch.randn(1, 6, 25, 32, 32).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input1, input2, input3, 'train OA',))
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))
