import time
import argparse
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils import *
import pytorch_ssim
from einops import rearrange

from model_AOnet import Net

# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--upscale_factor", type=int, default=4, help="upscale factor")
    parser.add_argument('--model_name', type=str, default='AO_transformer_net_3')
    parser.add_argument('--trainset_dir', type=str, default='E:/Light field dataset/TrainingData_5x5_4xSR')
    # parser.add_argument('--trainset_dir', type=str, default='E:/Light field dataset/SubtleTrainingData_5x5_4xSR')
    # parser.add_argument('--trainset_dir', type=str, default='E:/SubtleTrainingData_5x5_4xSR')

    parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--lr', type=float, default=4e-4, help='initial learning rate')

    parser.add_argument('--n_epochs', type=int, default=40, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=10, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')

    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='./log/AOnet_2xSR_5x5_ours.pth.tar')

    return parser.parse_args()


def train(cfg, train_loader):
    net = Net(cfg.angRes, cfg.upscale_factor, mpi_channel=25)
    net.to(cfg.device)
    cudnn.benchmark = True
    epoch_state = 0

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            net.load_state_dict(model['state_dict'])
            epoch_state = model["epoch"]
            print("Load Model!")
        else:
            print("=> no model found at '{}'".format(cfg.load_model))
    else:
        print("Not Load Model!")

    criterion_Loss = torch.nn.L1Loss().to(cfg.device)
    ssim_loss = pytorch_ssim.SSIM()
    optimizer = torch.optim.Adam([{'params': net.parameters(), 'initial_lr': 4e-4}], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps,
                                                gamma=cfg.gamma, last_epoch=epoch_state-1)
    scheduler._step_count = epoch_state
    loss_epoch = []
    loss_epoch1 = []
    loss_epoch2 = []
    loss_list = []

    for idx_epoch in range(epoch_state, cfg.n_epochs+1):
        print(optimizer.param_groups[0]['lr'])
        counter = 0

        for idx_iter, (data, raw, raw_refocus, label) \
                in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, raw, label, raw_refocus =\
                data.cuda(), raw.cuda(), label.cuda(), raw_refocus.cuda()

        # ## synthesis PSV
        # for idx_iter, (data, raw, raw_hr, label) \
        #         in tqdm(enumerate(train_loader), total=len(train_loader)):
        #     data, raw, raw_hr, label = data.cuda(), raw.cuda(), raw_hr.cuda(), label.cuda()
        #
        #     raw_refocus = tower_built(cfg, raw_hr)
        #     data = rearrange(data, 'b c (a1 h) (a2 w) -> b c (a1 a2) h w', a1=cfg.angRes, a2=cfg.angRes)
        #     data = data[:, :, :, 5:-5, 5:-5]
        #     data = rearrange(data, 'b c (a1 a2) h w-> b c (a1 h) (a2 w)', a1=cfg.angRes, a2=cfg.angRes)
        #     label = rearrange(label, 'b c (a1 h) (a2 w) -> b c (a1 a2) h w', a1=cfg.angRes, a2=cfg.angRes)
        #     label = label[:, :, :, 20:-20, 20:-20]
        #     label = rearrange(label, 'b c (a1 a2) h w-> b c (a1 h) (a2 w)', a1=cfg.angRes, a2=cfg.angRes)
        #     raw = rearrange(raw, 'b c (a1 h) (a2 w) -> b c (a1 a2) h w', a1=cfg.angRes, a2=cfg.angRes)
        #     raw = raw[:, :, :, 5:-5, 5:-5]
        #     raw = rearrange(raw, 'b c (a1 a2) h w-> b c (a1 h) (a2 w)', a1=cfg.angRes, a2=cfg.angRes)
        #     raw_refocus = rearrange(raw_refocus, 'b n c h w-> (b n) c h w')
        #     raw_refocus = F.interpolate(raw_refocus, scale_factor=0.25, mode='bicubic', align_corners=False)
        #     # raw_refocus = F.interpolate(raw_refocus, scale_factor=0.25, mode='nearest')
        #     raw_refocus = rearrange(raw_refocus, '(b n) c h w-> b n c h w', b=cfg.batch_size)
        #     raw_refocus = raw_refocus[:, :, :, 5:-5, 5:-5]
        #     data, raw, raw_refocus, label = augmentation_subtle(data, raw, raw_refocus, label)

            raw_refocus = raw_refocus[:, ::2, :, :, :]

            data = torch.cat((data, raw), dim=1)

            # MPI module
            out = net(data, raw, raw_refocus, 'train AO')
            # SPI module (Ablation study)
            # out = net(data, raw, 'train AO')

            b, n, h, w = label.shape
            label = label[:, :, 2 * h // 5:3 * h // 5, 2 * w // 5:3 * w // 5]
            loss1 = criterion_Loss(out, label)
            loss2 = 0.5 * (1 - ssim_loss(out, label))
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())
            loss_epoch1.append(loss1.data.cpu())
            loss_epoch2.append(loss2.data.cpu())

            if counter % 500 == 0:
                print(np.mean(loss_epoch), np.mean(loss_epoch1), np.mean(loss_epoch2))

            counter += 1

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'loss': loss_list},
                save_path='./log/', filename=cfg.model_name + '_' + str(cfg.upscale_factor) + 'xSR_' + str(cfg.angRes) +
                            'x' + str(cfg.angRes) + '_epoch_' + str(idx_epoch + 1) + '.pth.tar')
            loss_epoch = []
            loss_epoch1 = []
            loss_epoch2 = []

        scheduler.step()


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))


def main(cfg):
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir)
    # train_set = TrainSetLoader_subtle(dataset_dir=cfg.trainset_dir)
    train_loader = DataLoader(dataset=train_set, num_workers=4, pin_memory=True, batch_size=cfg.batch_size, shuffle=True)
    train(cfg, train_loader)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
