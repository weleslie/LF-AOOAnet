import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage import metrics
from scipy.io import loadmat
import h5py


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        file_list = os.listdir(dataset_dir)
        item_num = len(file_list)
        self.item_num = item_num
        # self.item_num = 2000

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        index = index + 1

        file_name = dataset_dir + '/%d' % index + '.mat'
        # with h5py.File(file_name, 'r') as hf:
        #     lr_color = np.array(hf.get('data'))
        #     label = np.array(hf.get('label'))
        #     lr_raw = np.array(hf.get('raw'))
        #     color_refocus = np.array(hf.get('data_refocus'))
        #     raw_refocus = np.array(hf.get('raw_refocus'))
        data = loadmat(file_name, verify_compressed_data_integrity=False)
        lr_color = data['data']
        lr_raw = data['raw']
        raw_refocus = data['raw_refocus']
        label = data['label']

        lr_color, lr_raw, raw_refocus, label =\
            augmentation(lr_color, lr_raw, raw_refocus, label)

        lr_color = ToTensor()(lr_color.copy())
        lr_raw = ToTensor()(lr_raw.copy())
        raw_refocus = torch.from_numpy(raw_refocus.copy()).contiguous()
        label = ToTensor()(label.copy())

        return lr_color, lr_raw, raw_refocus, label

    def __len__(self):
        return self.item_num


class TrainSetLoader_subtle(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader_subtle, self).__init__()
        self.dataset_dir = dataset_dir
        file_list = os.listdir(dataset_dir)
        item_num = len(file_list)
        # self.item_num = item_num
        self.item_num = 28984

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        index = index + 1

        file_name = dataset_dir + '/%d' % index + '.mat'
        # with h5py.File(file_name, 'r') as hf:
        #     lr_color = np.array(hf.get('data'))
        #     label = np.array(hf.get('label'))
        #     lr_raw = np.array(hf.get('raw'))
        data = loadmat(file_name, verify_compressed_data_integrity=False)
        lr_color = data['data']
        lr_raw = data['raw']
        hr_raw = data['raw_hr']
        label = data['label']

        lr_color = ToTensor()(lr_color.copy())
        lr_raw = ToTensor()(lr_raw.copy())
        hr_raw = ToTensor()(hr_raw.copy())
        label = ToTensor()(label.copy())

        return lr_color, lr_raw, hr_raw, label

    def __len__(self):
        return self.item_num


def augmentation(data, raw, raw_refocus, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        raw = raw[:, ::-1]
        raw_refocus = raw_refocus[:, :, :, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        raw = raw[::-1, :]
        raw_refocus = raw_refocus[:, :, ::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5: # transpose between U-V and H-W
        data = data.transpose(1, 0)
        raw = raw.transpose(1, 0)
        raw_refocus = raw_refocus.transpose(0, 1, 3, 2)
        label = label.transpose(1, 0)
    return data, raw, raw_refocus, label


def augmentation_subtle(data, raw, raw_refocus, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = torch.flip(data, [3])
        raw = torch.flip(raw, [3])
        raw_refocus = torch.flip(raw_refocus, [4])
        label = torch.flip(label, [3])
    if random.random() < 0.5:  # flip along W-V direction
        data = torch.flip(data, [2])
        raw = torch.flip(raw, [2])
        raw_refocus = torch.flip(raw_refocus, [3])
        label = torch.flip(label, [2])
    if random.random() < 0.5:   # transpose between U-V and H-W
        data = data.permute(0, 1, 3, 2)
        raw = raw.permute(0, 1, 3, 2)
        raw_refocus = raw_refocus.permute(0, 1, 2, 4, 3)
        label = label.permute(0, 1, 3, 2)
    return data, raw, raw_refocus, label


def cal_psnr(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.peak_signal_noise_ratio(img1_np, img2_np)


def cal_ssim(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.structural_similarity(img1_np, img2_np, gaussian_weights=True)


def cal_metrics(img1, img2, angRes):
    if len(img1.size())==2:
        [H, W] = img1.size()
        img1 = img1.view(angRes, H // angRes, angRes, W // angRes).permute(0,2,1,3)
    if len(img2.size())==2:
        [H, W] = img2.size()
        img2 = img2.view(angRes, H // angRes, angRes, W // angRes).permute(0,2,1,3)

    [U, V, h, w] = img1.size()
    PSNR = np.zeros(shape=(U, V), dtype='float32')
    SSIM = np.zeros(shape=(U, V), dtype='float32')

    for u in range(U):
        for v in range(V):
            PSNR[u, v] = cal_psnr(img1[u, v, :, :], img2[u, v, :, :])
            SSIM[u, v] = cal_ssim(img1[u, v, :, :], img2[u, v, :, :])
            pass
        pass

    psnr_mean = PSNR.sum() / np.sum(PSNR > 0)
    ssim_mean = SSIM.sum() / np.sum(SSIM > 0)

    return psnr_mean, ssim_mean


def tower_built(cfg, img):
    b, n, h, w = img.shape
    slopes = np.array([-1, -0.6, -0.2, 0.2, 0.6, 1])
    U = np.linspace(-0.5, 0.5, cfg.angRes) * cfg.angRes
    V = np.linspace(-0.5, 0.5, cfg.angRes) * cfg.angRes

    towers = []
    for slope in slopes:
        temp = []
        U1 = U * slope
        V1 = V * slope
        for j in range(cfg.angRes):
            U_temp = U1[j]
            for i in range(cfg.angRes):
                V_temp = V1[i]
                temp_img = img[:, :,
                           i * h // cfg.angRes:(i + 1) * h // cfg.angRes,
                           j * w // cfg.angRes:(j + 1) * w // cfg.angRes]
                out = plane_sweep(temp_img, U_temp, V_temp,
                                  b, h // cfg.angRes, w // cfg.angRes)

                temp.append(out)

        tower = torch.cat(temp, dim=1)
        towers.append(tower)

    towers = torch.stack(towers, dim=1)

    return towers


def plane_sweep(imgr, U, V, b, h, w):
    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, w).repeat(b, h, 1).type_as(imgr)
    y_base = torch.linspace(0, 1, h).repeat(b, w, 1).transpose(1, 2).type_as(imgr)

    # Apply shift in X direction
    x_shifts = U / w  # Normalize the U dimension
    y_shifts = V / h  # Normalize the V dimension
    flow_field = torch.stack((x_base + x_shifts, y_base + y_shifts), dim=3) # [B, H, W, 2]
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(imgr, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros', align_corners=False)

    return output


def LFdivide(data, angRes, patch_size, stride):
    uh, vw = data.shape
    h0 = uh // angRes
    w0 = vw // angRes
    bdr = (patch_size - stride) // 2
    h = h0 + 2 * bdr
    w = w0 + 2 * bdr
    if (h - patch_size) % stride:
        numU = (h - patch_size)//stride + 2
    else:
        numU = (h - patch_size)//stride + 1
    if (w - patch_size) % stride:
        numV = (w - patch_size)//stride + 2
    else:
        numV = (w - patch_size)//stride + 1
    hE = stride * (numU-1) + patch_size
    wE = stride * (numV-1) + patch_size

    dataE = torch.zeros(hE*angRes, wE*angRes)
    for u in range(angRes):
        for v in range(angRes):
            Im = data[u*h0:(u+1)*h0, v*w0:(v+1)*w0]
            dataE[u*hE:u*hE+h, v*wE:v*wE+w] = ImageExtend(Im, bdr)
    subLF = torch.zeros(numU, numV, patch_size*angRes, patch_size*angRes)
    for kh in range(numU):
        for kw in range(numV):
            for u in range(angRes):
                for v in range(angRes):
                    uu = u*hE + kh*stride
                    vv = v*wE + kw*stride
                    subLF[kh, kw, u*patch_size:(u+1)*patch_size, v*patch_size:(v+1)*patch_size] = \
                        dataE[uu:uu+patch_size, vv:vv+patch_size]
    return subLF


def LFdivide_PSV(data, angRes, patch_size, stride):
    n1, n2, uh, vw = data.shape
    h0 = uh
    w0 = vw
    bdr = (patch_size - stride) // 2
    h = h0 + 2 * bdr
    w = w0 + 2 * bdr
    if (h - patch_size) % stride:
        numU = (h - patch_size)//stride + 2
    else:
        numU = (h - patch_size)//stride + 1
    if (w - patch_size) % stride:
        numV = (w - patch_size)//stride + 2
    else:
        numV = (w - patch_size)//stride + 1
    hE = stride * (numU - 1) + patch_size
    wE = stride * (numV - 1) + patch_size

    dataE = torch.zeros(n1, n2, hE, wE)
    temp = ImageExtendPSV(data, bdr)
    dataE[:, :, :h, :w] = temp
    subLF = torch.zeros(n1, n2, numU, numV, patch_size, patch_size)
    for kh in range(numU):
        for kw in range(numV):
            uu = kh * stride
            vv = kw * stride
            subLF[:, :, kh, kw, :, :] = \
                dataE[:, :, uu:uu+patch_size, vv:vv+patch_size]
    return subLF


def ImageExtend(Im, bdr):
    h, w = Im.shape
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[h - bdr: 2 * h + bdr, w - bdr: 2 * w + bdr]

    return Im_out


def ImageExtendPSV(Im, bdr):
    n1, n2, h, w = Im.shape
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr: 2 * h + bdr, w - bdr: 2 * w + bdr]

    return Im_out


def LFintegrate(subLF, angRes, pz, stride, h0, w0):
    numU, numV, pH, pW = subLF.shape
    # H, W = numU*pH, numV*pW
    ph, pw = pH // angRes, pW // angRes
    bdr = (pz - stride) // 2
    temp = torch.zeros(stride*numU, stride*numV)
    outLF = torch.zeros(angRes, angRes, h0, w0)
    for u in range(angRes):
        for v in range(angRes):
            for ku in range(numU):
                for kv in range(numV):
                    temp[ku*stride:(ku+1)*stride, kv*stride:(kv+1)*stride] = \
                        subLF[ku, kv, u*ph+bdr:u*ph+bdr+stride, v*pw+bdr:v*ph+bdr+stride]

            outLF[u, v, :, :] = temp[0:h0, 0:w0]

    return outLF


def LFsep(data, angRes, patch_size, bdr):
    uh, vw = data.shape
    h0 = uh // angRes
    w0 = vw // angRes

    pu, pv = patch_size
    puE, pvE = pu + bdr * 2, pv + bdr * 2
    numU, numV = h0 // pu, w0 // pv
    subLF = torch.zeros(numU, numV, puE*angRes, pvE*angRes)

    for kh in range(numU):
        for kw in range(numV):
            for u in range(angRes):
                for v in range(angRes):
                    lbdr, ubdr = -bdr, -bdr
                    if kh == 0:
                        ubdr = 0
                    elif kh == numU - 1:
                        ubdr = -bdr * 2

                    if kw == 0:
                        lbdr = 0
                    elif kw == numV - 1:
                        lbdr = -bdr * 2

                    uu = u*h0 + kh*pu + ubdr
                    vv = v*w0 + kw*pv + lbdr
                    subLF[kh, kw, u*puE:(u+1)*puE, v*pvE:(v+1)*pvE] = \
                        data[uu:uu+puE, vv:vv+pvE]

    return subLF


def LFsep_PSV(data, angRes, patch_size, bdr):
    n, c, h0, w0 = data.shape

    pu, pv = patch_size
    puE, pvE = pu + bdr * 2, pv + bdr * 2
    numU, numV = h0 // pu, w0 // pv
    subLF = torch.zeros(n, c, numU, numV, puE, pvE)

    for kh in range(numU):
        for kw in range(numV):
            lbdr, ubdr = -bdr, -bdr
            if kh == 0:
                ubdr = 0
            elif kh == numU - 1:
                ubdr = -bdr * 2

            if kw == 0:
                lbdr = 0
            elif kw == numV - 1:
                lbdr = -bdr * 2

            uu = kh * pu + ubdr
            vv = kw * pv + lbdr
            subLF[:, :, kh, kw, :, :] = data[:, :, uu:uu+puE, vv:vv+pvE]

    return subLF


def LFcom(subLF, angRes, bdr, h0, w0):
    numU, numV, pHE, pWE = subLF.shape
    # H, W = numU*pH, numV*pW
    phE, pwE = pHE // angRes, pWE // angRes
    ph, pw = phE - 2 * bdr, pwE - 2 * bdr
    temp = torch.zeros(ph*numU, pw*numV)
    outLF = torch.zeros(angRes, angRes, h0, w0)
    for u in range(angRes):
        for v in range(angRes):
            for ku in range(numU):
                for kv in range(numV):
                    lbdr, ubdr = bdr, bdr

                    if ku == 0:
                        ubdr = 0
                    elif ku == numU - 1:
                        ubdr = bdr * 2

                    if kv == 0:
                        lbdr = 0
                    elif kv == numV - 1:
                        lbdr = bdr * 2

                    uu = u * phE + ubdr
                    vv = v * pwE + lbdr

                    temp[ku*ph:(ku+1)*ph, kv*pw:(kv+1)*pw] = \
                        subLF[ku, kv, uu:uu+ph, vv:vv+pw]

            outLF[u, v, :, :] = temp

    return outLF