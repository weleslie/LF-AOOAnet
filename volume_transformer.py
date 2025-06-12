from einops import rearrange
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionEncoding(nn.Module):
    def __init__(self, temperature):
        super(PositionEncoding, self).__init__()
        self.temperature = temperature

    def forward(self, x, dim: list, token_dim):
        self.token_dim = token_dim
        assert len(x.size()) == 5, 'the object of position encoding requires 5-dim tensor! '
        grid_dim = torch.linspace(0, self.token_dim - 1, self.token_dim, dtype=torch.float32)
        grid_dim = 2 * (grid_dim // 2) / self.token_dim
        grid_dim = self.temperature ** grid_dim
        position = None
        for index in range(len(dim)):
            pos_size = [1, 1, 1, 1, 1, self.token_dim]
            length = x.size(dim[index])
            pos_size[dim[index]] = length

            pos_dim = (torch.linspace(0, length - 1, length, dtype=torch.float32).view(-1, 1) / grid_dim).to(x.device)
            pos_dim = torch.cat([pos_dim[:, 0::2].sin(), pos_dim[:, 1::2].cos()], dim=1)
            pos_dim = pos_dim.view(pos_size)

            if position is None:
                position = pos_dim
            else:
                position = position + pos_dim
            pass

        position = rearrange(position, 'b 1 a h w dim -> b dim a h w')

        return position / len(dim)


class VolumeTrans(nn.Module):
    def __init__(self, channels, MHSA_params, layerNum=6):
        super(VolumeTrans, self).__init__()
        self.layerNum = layerNum
        self.ang_dim = channels
        self.norm = nn.LayerNorm(self.ang_dim)
        self.attention = nn.MultiheadAttention(self.ang_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.ang_dim),
            nn.Linear(self.ang_dim, self.ang_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(self.ang_dim * 2, self.ang_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )

    @staticmethod
    def SAI2Token(buffer):
        buffer_token = rearrange(buffer, 'b c n h w -> n (b h w) c')
        return buffer_token

    def Token2SAI(self, buffer_token, h, w):
        buffer = rearrange(buffer_token, '(n) (b h w) (c) -> b c n h w', n=self.layerNum, h=h, w=w)
        return buffer

    def forward(self, buffer, ang_position):
        _, _, _, h, w = buffer.shape
        ang_token = self.SAI2Token(buffer)
        ang_PE = self.SAI2Token(ang_position)
        ang_token_norm = self.norm(ang_token + ang_PE)

        ang_token = self.attention(query=ang_token_norm,
                                   key=ang_token_norm,
                                   value=ang_token,
                                   need_weights=False)[0] + ang_token

        ang_token = self.feed_forward(ang_token) + ang_token
        buffer = self.Token2SAI(ang_token, h, w)

        return buffer


# class ParallaxTrans(nn.Module):
#     def __init__(self, channels, scale_factor, MHSA_params):
#         super(ParallaxTrans, self).__init__()
#         self.factor = scale_factor
#         self.kernel_field_cv = 3
#         self.kernel_field_sv = 3
#         self.kernel_search = 5
#         self.spa_dim = channels
#         self.MLP_sv = nn.Linear(channels * self.kernel_field_sv ** 2, self.spa_dim, bias=False)
#         self.MLP_cv = nn.Linear(channels * self.kernel_field_cv ** 2, self.spa_dim, bias=False)
#
#         self.norm = nn.LayerNorm(self.spa_dim)
#         self.attention = nn.MultiheadAttention(self.spa_dim,
#                                                MHSA_params['num_heads'],
#                                                MHSA_params['dropout'],
#                                                bias=False)
#         nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
#         self.attention.out_proj.bias = None
#
#         self.feed_forward = nn.Sequential(
#             nn.LayerNorm(self.spa_dim),
#             nn.Linear(self.spa_dim, self.spa_dim*2, bias=False),
#             nn.ReLU(True),
#             nn.Dropout(MHSA_params['dropout']),
#             nn.Linear(self.spa_dim*2, self.spa_dim, bias=False),
#             nn.Dropout(MHSA_params['dropout'])
#         )
#
#         self.linear = nn.Sequential(
#             nn.Conv3d(self.spa_dim, channels, kernel_size=(1, 1, 1), padding=(0, 0, 0), dilation=1, bias=False),
#         )
#
#     def SAI2Token_cv(self, buffer):
#         buffer = rearrange(buffer, 'b c a h w -> (b a) c h w')
#         # local feature embedding
#         spa_token = F.unfold(buffer, kernel_size=self.kernel_field_cv, padding=self.kernel_field_cv // 2).permute(2, 0, 1)
#         spa_token = self.MLP_cv(spa_token)
#
#         return spa_token
#
#     def SAI2Token_sv(self, buffer):
#         buffer = rearrange(buffer, 'b c a h w -> (b a) c h w')
#         # local feature embedding
#         spa_token = F.unfold(buffer, kernel_size=self.kernel_field_sv, padding=self.kernel_field_sv // 2).permute(2, 0, 1)
#         spa_token = self.MLP_sv(spa_token)
#
#         return spa_token
#
#     def Token2SAI(self, buffer_token_spa, h, w, angRes):
#         buffer = rearrange(buffer_token_spa, '(h w) (b a) c -> b c a h w', h=h, w=w, a=angRes)
#         return buffer
#
#     def forward(self, buffer_sv, buffer_cv, spa_position):
#         _, _, a, h, w = buffer_sv.shape
#
#         spa_token_sv = self.SAI2Token_sv(buffer_sv)
#         spa_token_cv = self.SAI2Token_cv(buffer_cv)
#         spa_PE = self.SAI2Token_sv(spa_position)
#         spa_token_norm_sv = self.norm(spa_token_sv + spa_PE)
#         spa_token_norm_cv = self.norm(spa_token_cv + spa_PE)
#
#         # spa_token_norm_sv = spa_token_norm_sv.permute(1, 0, 2)
#         # spa_token_norm_cv = spa_token_norm_cv.permute(1, 2, 0)
#         # spa_token_sv = spa_token_sv.permute(1, 0, 2)
#         # spa_Q_K = torch.bmm(spa_token_norm_sv, spa_token_norm_cv)
#         # spa_Q_K = self.softmax(spa_Q_K)
#         # spa_token = torch.bmm(spa_Q_K, spa_token_sv) + spa_token_sv
#         # spa_token = spa_token.permute(1, 0, 2)
#
#         spa_token = self.attention(query=spa_token_norm_sv,
#                                    key=spa_token_norm_cv,
#                                    value=spa_token_cv,
#                                    need_weights=False)[0] + spa_token_sv
#         spa_token = self.feed_forward(spa_token) + spa_token
#         buffer = self.Token2SAI(spa_token, h, w, a)
#
#         return buffer

class ParallaxTrans(nn.Module):
    def __init__(self, channels, scale_factor, MHSA_params):
        super(ParallaxTrans, self).__init__()
        self.factor = scale_factor
        self.kernel_field_cv = 3
        self.kernel_field_sv = 3
        self.kernel_search = 5
        self.spa_dim = channels
        self.MLP_sv = nn.Linear(channels * self.kernel_field_sv ** 2, self.spa_dim, bias=False)
        self.MLP_cv = nn.Linear(channels * self.kernel_field_cv ** 2, self.spa_dim, bias=False)

        self.norm = nn.LayerNorm(self.spa_dim)
        self.attention = nn.MultiheadAttention(self.spa_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.spa_dim),
            nn.Linear(self.spa_dim, self.spa_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(self.spa_dim*2, self.spa_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )

        self.linear = nn.Sequential(
            nn.Conv3d(self.spa_dim, channels, kernel_size=(1, 1, 1), padding=(0, 0, 0), dilation=1, bias=False),
        )

    def SAI2Token_cv(self, buffer):
        buffer = rearrange(buffer, 'b c a h w -> (b a) c h w')
        # local feature embedding
        spa_token = F.unfold(buffer, kernel_size=self.kernel_field_cv, padding=self.kernel_field_cv // 2).permute(2, 0, 1)
        spa_token = self.MLP_cv(spa_token)

        return spa_token

    def SAI2Token_sv(self, buffer):
        buffer = rearrange(buffer, 'b c a h w -> (b a) c h w')
        # local feature embedding
        spa_token = F.unfold(buffer, kernel_size=self.kernel_field_sv, padding=self.kernel_field_sv // 2).permute(2, 0, 1)
        spa_token = self.MLP_sv(spa_token)

        return spa_token

    def Token2SAI(self, buffer_token_spa, h, w, angRes):
        buffer = rearrange(buffer_token_spa, '(h w) (b a) c -> b c a h w', h=h, w=w, a=angRes)
        return buffer

    def forward(self, buffer_sv, buffer_cv, spa_position):
        _, _, a, h, w = buffer_sv.shape

        spa_token_sv = self.SAI2Token_sv(buffer_sv)
        spa_token_cv = self.SAI2Token_cv(buffer_cv)
        spa_PE = self.SAI2Token_sv(spa_position)
        spa_token_norm_sv = self.norm(spa_token_sv + spa_PE)
        spa_token_norm_cv = self.norm(spa_token_cv + spa_PE)

        # spa_token_norm_sv = spa_token_norm_sv.permute(1, 0, 2)
        # spa_token_norm_cv = spa_token_norm_cv.permute(1, 2, 0)
        # spa_token_sv = spa_token_sv.permute(1, 0, 2)
        # spa_Q_K = torch.bmm(spa_token_norm_sv, spa_token_norm_cv)
        # spa_Q_K = self.softmax(spa_Q_K)
        # spa_token = torch.bmm(spa_Q_K, spa_token_sv) + spa_token_sv
        # spa_token = spa_token.permute(1, 0, 2)

        spa_token = self.attention(query=spa_token_norm_sv,
                                   key=spa_token_norm_cv,
                                   value=spa_token_cv,
                                   need_weights=False)[0] + spa_token_sv
        spa_token = self.feed_forward(spa_token) + spa_token
        buffer = self.Token2SAI(spa_token, h, w, a)

        return buffer