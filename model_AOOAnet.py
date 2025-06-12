import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from dcn.modules.deform_conv import DeformConv
from modules import BinocularEncoder, MultiscaleDecoder, FlowWarp
from volume_transformer import PositionEncoding, VolumeTrans, ParallaxTrans
from einops import rearrange


class Net(nn.Module):
    def __init__(self, angRes=5, factor=2, spi_channel=2, mpi_channel=25):
        super(Net, self).__init__()
        ato_channel = 64
        channel = 32
        self.factor = factor
        self.angRes = angRes
        ## MPI module
        self.ato = ATO(self.factor, mpi_channel, ato_channel)
        ## SPI module
        # self.ato = SISR(self.factor, spi_channel, channel)

        self.beta = SISR(self.factor, spi_channel, channel)

        self.fea_extra = nn.Sequential(
            nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            ResASPP(channel),
            RB(channel),
            ResASPP(channel),
            RB(channel),
            # RDG(G0=channel, C=4, G=24, n_RDB=4)
        )

        # self.dfnet1 = DFNet(channel, channel)
        # self.dfnet2 = DFNet(channel, channel)
        self.dfnet3 = DFNet(channel, 1)

        self.OTA = OTA(self.factor, channel)

    def forward(self, hybrid, x, x_refocus, state):
        x_upscale = F.interpolate(x, scale_factor=self.factor, mode='bicubic', align_corners=False)
        x_sv, x_cv = LFsplit(x, self.angRes, x.shape[1])
        x_sv_up, x_cv_up = LFsplit(x_upscale, self.angRes, x_upscale.shape[1])

        hybrid_upscale = F.interpolate(hybrid, scale_factor=self.factor, mode='bicubic', align_corners=False)
        hybrid_sv, hybrid_cv = LFsplit(hybrid, self.angRes, hybrid.shape[1])
        hybrid_sv_up, hybrid_cv_up = LFsplit(hybrid_upscale, self.angRes, hybrid_upscale.shape[1])

        ## MPI module
        out_c, mask, color = self.ato(x_refocus, x_cv_up)
        ## SPI module
        # x_cv = x_cv.unsqueeze(1)
        # x_cv_up = x_cv_up.unsqueeze(1)
        # out_c = self.ato(x_cv, x_cv_up)
        # out_c = out_c.squeeze(1)

        ## luminance compensation for center view
        # hybrid_res = hybrid_cv[:, 0:1, :, :] - hybrid_cv[:, 1:2, :, :]
        # hybrid_res_up = hybrid_cv_up[:, 0:1, :, :] - hybrid_cv_up[:, 1:2, :, :]
        # hybrid_res = hybrid_res.unsqueeze(1)

        ## with SISR in luminance compensation
        # hybrid_res_up = hybrid_res_up.unsqueeze(1)
        # factor_cv = self.beta(hybrid_res, hybrid_res_up)
        # factor_cv = factor_cv.squeeze(1)
        # out_c = out_c + factor_cv
        ## without SISR in luminance compensation
        # factor_cv = hybrid_res_up
        # out_c = out_c + factor_cv

        # ## with SISR in residual learning
        # out_s0 = self.beta(x_sv, x_sv_up)
        # # out_sisr = FormOutput(out_s0)
        # ## without SISR in residual learning
        # # out_s0 = x_sv_up
        # # out_sisr = FormOutput(out_s0)

        # fea_cv = self.fea_extra(out_c)
        # b, n, c, h, w = out_s0.shape
        # out_s0 = out_s0.contiguous().view(b * n, -1, h, w)
        # fea_sv = self.fea_extra(out_s0)
        # fea_sv = fea_sv.contiguous().view(b, n, -1, h, w)
        # fea_s = self.dfnet3(fea_sv, fea_cv)

        fea_cv = self.fea_extra(out_c)
        b, n, c, h, w = x_sv.shape
        fea_sv = x_sv.contiguous().view(b * n, -1, h, w)
        fea_sv = self.fea_extra(fea_sv)
        fea_sv = fea_sv.contiguous().view(b, n, -1, h, w)
        fea_s = self.OTA(fea_sv, fea_cv, x_sv_up)

        ## luminance compensation for all views
        hybrid_res = hybrid_sv[:, :, 0:1, :, :] - hybrid_sv[:, :, 1:2, :, :]
        hybrid_res_up = hybrid_sv_up[:, :, 0:1, :, :] - hybrid_sv_up[:, :, 1:2, :, :]

        ## with SISR in luminance compensation
        factor_sv = self.beta(hybrid_res, hybrid_res_up)
        ## without SISR in luminance compensation
        # factor_sv = hybrid_res_up

        ## with SISR in Residual learning
        # out_s = fea_s + x_sv_up + factor_sv
        ## without SISR in Residual learning
        out_s = fea_s + factor_sv

        out = FormOutput(out_s)

        if state == 'train OA':
            return out
        elif state == 'test OA':
            ## MPI module
            # return out, out_c, out_sisr
            return out, out_c
            ## SPI module
            # return out


class ATO(nn.Module):
    def __init__(self, factor, in_channel, out_channel):
        super(ATO, self).__init__()
        self.factor = factor
        self.out_channel = out_channel
        self.FeaExtract = FeaExtract(in_channel, self.out_channel)
        self.Recons = RDG(G0=self.out_channel, C=4, G=32, n_RDB=4)

        self.pos_encoding = PositionEncoding(temperature=10000)
        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = 8
        self.MHSA_params['dropout'] = 0.
        self.layer_transformer1 = VolumeTrans(self.out_channel, self.MHSA_params)
        # self.layer_transformer2 = VolumeTrans(self.out_channel, self.MHSA_params)
        # self.layer_transformer3 = VolumeTrans(self.out_channel, self.MHSA_params)
        # self.layer_transformer4 = VolumeTrans(self.out_channel, self.MHSA_params)

        self.upsample = Upsample_MPI(self.out_channel, self.factor)

        self.softmax = nn.Softmax(1)

    def forward(self, x_refocus, x_upscale):
        b, n, c, h, w = x_refocus.shape

        ## Without MPI representation
        # x_refocus = x_refocus.contiguous().view(b, -1, n * c, h, w)

        x = self.FeaExtract(x_refocus)
        x = self.Recons(x)

        x = rearrange(x, '(b n) c h w -> b c n h w', b=b)
        ang_position = self.pos_encoding(x, dim=[2], token_dim=self.out_channel)
        x = self.layer_transformer1(x, ang_position)
        # x = self.layer_transformer2(x, ang_position)
        # x = self.layer_transformer3(x, ang_position)
        # x = self.layer_transformer4(x, ang_position)
        x1 = rearrange(x, 'b c n h w -> (b n) c h w')

        out, mask, color = self.upsample(x1, b, n)

        out = out + x_upscale

        return out, mask, color


class OTA(nn.Module):
    def __init__(self, factor, channel):
        super(OTA, self).__init__()
        self.factor = factor
        self.out_channel = channel

        self.FeaExtract = FeaExtract(1, self.out_channel)

        self.pos_encoding = PositionEncoding(temperature=10000)
        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = 8
        self.MHSA_params['dropout'] = 0.
        self.parallax_transformer1 = ParallaxTrans(self.out_channel, self.factor, self.MHSA_params)
        # self.layer_transformer2 = VolumeTrans(self.out_channel, self.MHSA_params)
        # self.layer_transformer3 = VolumeTrans(self.out_channel, self.MHSA_params)
        # self.layer_transformer4 = VolumeTrans(self.out_channel, self.MHSA_params)

        self.upsample = Upsample_SPI(self.out_channel, self.factor)

    def forward(self, fea_sv, fea_cv, x_upscale):
        b, n, c, h, w = fea_sv.shape

        # fea_sv = self.FeaExtract(fea_sv)
        # fea_sv = rearrange(fea_sv, '(b n) c h w -> b n c h w', b=b)

        fea_cv = fea_cv.unsqueeze(1)
        fea_cv = fea_cv.repeat(1, 25, 1, 1, 1)

        x_sv = rearrange(fea_sv, 'b n c h w -> b c n h w')
        x_cv = rearrange(fea_cv, 'b n c h w -> b c n h w')
        ang_position = self.pos_encoding(x_sv, dim=[3, 4], token_dim=self.out_channel)
        x = self.parallax_transformer1(x_sv, x_cv, ang_position)

        x1 = rearrange(x, 'b c n h w -> (b n) c h w')

        out = self.upsample(x1, b, n) + x_upscale
        # out = x1 + x_upscale

        return out


class SISR(nn.Module):
    def __init__(self, factor, in_channel, out_channel):
        super(SISR, self).__init__()
        self.factor = factor
        self.out_channel = out_channel
        self.FeaExtract = FeaExtract(in_channel, self.out_channel)
        self.Recons = RDG(G0=self.out_channel, C=4, G=24, n_RDB=4)

        # self.pos_encoding = PositionEncoding(temperature=10000)
        # self.MHSA_params = {}
        # self.MHSA_params['num_heads'] = 8
        # self.MHSA_params['dropout'] = 0.
        # self.parallax_transformer1 = ParallaxTrans(self.out_channel, self.factor, self.MHSA_params)

        self.upsample = Upsample_SPI(self.out_channel, self.factor)

    def forward(self, x_sv, x_sv_upscale):
        b, n, _, _, _ = x_sv.shape
        x = self.FeaExtract(x_sv)
        x1 = self.Recons(x)

        # x_sv = rearrange(x1, '(b n) c h w -> b c n h w', b=b)
        # spa_position = self.pos_encoding(x_sv, dim=[3, 4], token_dim=self.out_channel)
        # x = self.parallax_transformer1(x_sv, spa_position)
        # x1 = rearrange(x, 'b c n h w -> (b n) c h w')

        out = self.upsample(x1, b, n)

        out = out + x_sv_upscale

        return out


class DFNet(nn.Module):
    def __init__(self, channel, out_channel):
        super(DFNet, self).__init__()

        ## DC module
        self.conv_1 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.ASPP = ResASPP(channel)
        self.conv_off = nn.Conv2d(channel, 2 * 9, kernel_size=1, stride=1, padding=0)
        self.conv_off.lr_mult = 0.1
        self.init_offset()
        self.dcn = DeformConv(channel, channel, 3, 1, 1, deformable_groups=1)

        ## non-DC module
        # self.conv_DC = nn.Conv2d(channel, channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_f1 = nn.Sequential(
            nn.Conv2d(2 * channel, channel, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channel, out_channel, kernel_size=1, stride=1, padding=0)
        )

    def init_offset(self):
        self.conv_off.weight.data.zero_()
        self.conv_off.bias.data.zero_()

    def forward(self, fea_sv, fea_cv):
        out_sv = []
        for i in range(fea_sv.shape[1]):
            current_sv = fea_sv[:, i, :, :, :].contiguous()

            ## DC module
            buffer = torch.cat((fea_cv, current_sv), dim=1)  # B * 2C * H * W
            buffer = self.lrelu(self.conv_1(buffer))
            buffer = self.ASPP(buffer)
            offset = self.conv_off(buffer)  # B, 18, H, W
            dist_fea = self.lrelu(self.dcn(fea_cv, offset))  # B, C, H, W

            ## non-DC module
            # dist_fea = self.conv_DC(fea_cv)

            current_sv = torch.cat((current_sv, dist_fea), dim=1)
            current_sv = self.conv_f1(current_sv)
            out_sv.append(current_sv)
        out_sv = torch.stack(out_sv, dim=1)  # B, N, C, H, W

        return out_sv


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

        self.upsp_color_1 = nn.Sequential(
            nn.Conv2d(channel, channel * factor * factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(factor)
        )
        self.upsp_color_2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, 1, 1, 1, 0, bias=False)
        )

        ## 2D convolution
        self.upsp_alpha_1 = nn.Sequential(
            nn.Conv2d(channel, channel * factor * factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(factor)
        )
        self.upsp_alpha_2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, 1, 1, 1, 0, bias=False)
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
        color = self.upsp_color_1(x)
        color = self.upsp_color_2(color)
        _, _, H, W = color.shape
        color = color.contiguous().view(b, n, -1, H, W)

        ## 3D convolution
        # mask = self.upsp_alpha_1(x)
        # mask = mask.contiguous().view(b, n, -1, H, W).permute(0, 2, 1, 3, 4)
        # mask = self.upsp_alpha_2(mask)
        # mask = mask.squeeze(1)
        ## 2D convolution
        mask = self.upsp_alpha_1(x)
        mask = self.upsp_alpha_2(mask)

        mask = mask.contiguous().view(b, -1, H * W)
        mask = self.softmax(mask)
        mask = mask.contiguous().view(b, n, -1, H, W)

        out = torch.sum(color * mask, dim=1)

        return out, mask, color

        ## Without MPI representation
        # color = self.upsp_color_1(x)
        # color = self.upsp_color_2(color)
        # return color, None, None


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
        buffer_0 = self.FEconv(x)
        buffer = self.FERB_1(buffer_0)
        buffer = self.FERB_2(buffer)
        buffer = self.FERB_3(buffer)
        buffer = self.FERB_4(buffer)
        # _, c, h, w = buffer.shape
        # # buffer:  B, N, C, H, W
        # buffer = buffer.unsqueeze(1).contiguous().view(b, -1, c, h, w)

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


def FormOutput(x_sv):
    b, n, c, h, w = x_sv.shape
    angRes = int(sqrt(n+1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(x_sv[:, kk, :, :, :])
            kk = kk+1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out


if __name__ == "__main__":
    net = Net(5, 4, spi_channel=1, mpi_channel=25).cuda()
    from thop import profile
    input1 = torch.randn(1, 2, 80, 80).cuda()
    input2 = torch.randn(1, 1, 80, 80).cuda()
    input3 = torch.randn(1, 6, 25, 16, 16).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input1, input2, input3, 'train OA',))
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))
