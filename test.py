import time
import argparse
import scipy.misc
import torch.backends.cudnn as cudnn
from utils import *
# from model_AOOAnet import Net
from model_AFOnet import Net
import matplotlib.pyplot as plt
import scipy
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--upscale_factor", type=int, default=4, help="upscale factor")
    parser.add_argument('--testset_dir', type=str, default='E:/Light field dataset/TestData_5x5_4xSR/')

    parser.add_argument("--patchsize", type=int, default=64, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--stride", type=int, default=32, help="The stride between two test patches is set to patchsize/2")

    # parser.add_argument('--model_path', type=str, default='./log/OAnet_2xSR_5x5_ours_2D.pth.tar')
    parser.add_argument('--model_path', type=str, default='./log/OFAnet_ParaTrans_4_channel_32_4xSR_5x5.pth.tar')
    parser.add_argument('--save_path', type=str, default='./Results_4x_RawSR/')
    # parser.add_argument('--save_path', type=str, default='./Ablation_Study/')

    return parser.parse_args()


def test(cfg):
    net = Net(cfg.angRes, cfg.upscale_factor, spi_channel=1, mpi_channel=25)
    net.to(cfg.device)
    cudnn.benchmark = True

    if os.path.isfile(cfg.model_path):
        model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
        pretrained_dict = model['state_dict']
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    else:
        print("=> no model found at '{}'".format(cfg.load_model))

    for root, dirs, files in os.walk(cfg.testset_dir):
        if len(files) == 0:
            break
        for file_name in files:
            file_path = cfg.testset_dir + file_name

            data = loadmat(file_path, verify_compressed_data_integrity=False)
            lr_color = np.expand_dims(data['data'], axis=0)
            lr_color = np.expand_dims(lr_color, axis=0)
            lr_color = torch.from_numpy(lr_color.copy()).cuda()

            lr_raw = np.expand_dims(data['raw'], axis=0)
            lr_raw = np.expand_dims(lr_raw, axis=0)
            lr_raw = torch.from_numpy(lr_raw.copy()).cuda()
            lr_color = torch.cat((lr_color, lr_raw), dim=1)

            raw_refocus = np.expand_dims(data['raw_refocus'], axis=0)
            raw_refocus = torch.from_numpy(raw_refocus.copy()).cuda()
            raw_refocus = raw_refocus[:, ::2, :, :, :]

            # lr_raw = lr_raw.squeeze()
            # lr_color = lr_color.squeeze()
            # raw_refocus = raw_refocus.squeeze()
            #
            # subLF_color = LFdivide(lr_color, cfg.angRes, cfg.patchsize, cfg.stride)
            # subLF_raw = LFdivide(lr_raw, cfg.angRes, cfg.patchsize, cfg.stride)
            # subLF_PSV = LFdivide_PSV(raw_refocus, cfg.angRes, cfg.patchsize, cfg.stride)
            #
            # uh, vw = lr_color.shape
            # h0, w0 = int(uh // cfg.angRes), int(vw // cfg.angRes)
            # numU, numV, H, W = subLF_color.size()
            # subLFout = torch.zeros(numU, numV, cfg.angRes * cfg.patchsize * cfg.upscale_factor,
            #                        cfg.angRes * cfg.patchsize * cfg.upscale_factor)
            #
            # for u in range(numU):
            #     for v in range(numV):
            #         tmp_color = subLF_color[u:u + 1, v:v + 1, :, :]
            #         tmp_raw = subLF_raw[u:u + 1, v:v + 1, :, :]
            #         tmp_PSV = subLF_PSV[:, :, u, v, :, :]
            #         tmp_PSV = tmp_PSV.unsqueeze(0)
            #         tmp_color = torch.cat((tmp_color, tmp_raw), dim=1)
            #
            #         with torch.no_grad():
            #             net.eval()
            #             torch.cuda.empty_cache()
            #             out, _ = net(tmp_color.cuda(), tmp_raw.cuda(), tmp_PSV.cuda(), 'test OA')
            #             subLFout[u:u + 1, v:v + 1, :, :] = out.squeeze()
            #
            # Sr_4D_y = LFintegrate(subLFout, cfg.angRes, cfg.patchsize * cfg.upscale_factor,
            #                       cfg.stride * cfg.upscale_factor, h0 * cfg.upscale_factor,
            #                       w0 * cfg.upscale_factor)
            # Sr_SAI_y = Sr_4D_y.permute(0, 2, 1, 3).reshape((h0 * cfg.angRes * cfg.upscale_factor,
            #                                                 w0 * cfg.angRes * cfg.upscale_factor))
            #
            # out = Sr_SAI_y.cpu().numpy()
            # scipy.io.savemat(cfg.save_path + file_name[0:-4] + '_results_trans2.mat', {'LF': out})

            with torch.no_grad():
                torch.cuda.empty_cache()

                # lr_color1 = lr_color[:, :, :h // 2, :w // 2]
                # lr_raw1 = lr_raw[:, :, :h // 2, :w // 2]
                # raw_refocus1 = raw_refocus[:, :, :, :h // 10, :w // 10]

                # s = time.time()
                out, out_c = net(lr_color, lr_raw, raw_refocus, 'test OA')
                # e = time.time()

                # print(e - s)
                # mask = mask.squeeze(2)
                # color = color.squeeze(2)

                # f = flow[0, :, :, :].cpu().numpy()
                # o = out[0, 0, :, :].cpu().numpy()
                # m = mask[0, 0, :, :].cpu().numpy()
                # c = color[0, 0, :, :].cpu().numpy()

                # plt.subplot(1, 3, 1)
                # plt.imshow(f[0, :, :])
                # plt.subplot(1, 3, 2)
                # plt.imshow(f[1, :, :])
                # plt.subplot(1, 3, 3)
                # plt.imshow(o)
                # plt.show()

                # for i in range(weight.shape[1]):
                #     w1 = weight[0, i, 0, :, :].cpu().numpy()
                #     w2 = weight[0, i, 1, :, :].cpu().numpy()
                #     # m = mask[0, i, :, :].cpu().numpy()
                #     # c = color[0, i, :, :].cpu().numpy()
                #
                #     plt.subplot(2, 1, 1)
                #     plt.imshow(w1)
                #     plt.subplot(2, 1, 2)
                #     plt.imshow(w2)
                #     plt.show()

                out = out.cpu().numpy()
                # img = out[0, 0, :, :]
                # plt.imshow(img)
                # plt.show()
                out_c = out_c.cpu().numpy()
                # out_sisr = out_sisr.cpu().numpy()
                scipy.io.savemat(cfg.save_path + file_name[0:-4] + '_results_trans_32_2.mat', {'LF': out, 'LF_c': out_c})
                # scipy.io.savemat(cfg.save_path + file_name[0:-4] + '_out_c_SISR.mat', {'LF': out_c})
                # scipy.io.savemat(cfg.save_path + file_name[0:-4] + '_out_sisr_SISR.mat', {'LF': out_sisr})

def main(cfg):
    test(cfg)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
