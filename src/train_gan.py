import os

import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
import random
from tqdm import tqdm

import constants as C
import GAN.fastGAN as V1
import GAN.fastGAN_v2 as V2
from GAN.operation import copy_G_params, load_params, get_dir
from GAN.operation import InfiniteSamplerWrapper
from GAN.diffaug import DiffAugment
from dataloader import DataRealGAN, DataCodeGAN
from torch.utils.tensorboard import SummaryWriter

from models.condition_model import ConditionCA

# policy = 'color,translation'
policy = 'translation'
from GAN import lpips_v1 as lpips

torch.manual_seed(330)
np.random.seed(330)

# torch.backends.cudnn.benchmark = True


def crop_image_by_part(image, part):
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :, :hw, :hw]
    if part == 1:
        return image[:, :, :hw, hw:]
    if part == 2:
        return image[:, :, hw:, :hw]
    if part == 3:
        return image[:, :, hw:, hw:]


def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=0)
    return F.cross_entropy(input, labels, ignore_index=-1)


def train_d(net, data, y, percept,type="real"):
    """Train function of discriminator"""
    if type == "real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, y, type, part=part)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() + \
              percept(rec_all, F.interpolate(data, rec_all.shape[2])).sum() + \
              percept(rec_small, F.interpolate(data, rec_small.shape[2])).sum() + \
              percept(rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2])).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, y, type)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()


def train(args):
    file_root = args.file_root
    img_root = args.img_root
    total_iterations = args.iter
    checkpoint = args.ckpt
    cls_ckpt = args.class_ckpt
    batch_size = args.batch_size
    img_size = args.img_size
    use_cuda = True
    multi_gpu = False
    dataloader_workers = 6
    current_iteration = args.start_iter
    save_interval = 100
    saved_model_folder, saved_image_folder = get_dir(args)

    percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True, gpu_ids=[args.gpu_id])

    writer = SummaryWriter(os.path.join(args.visual_dir, 'tensorboard/gan_log/' + args.name))

    device_id = args.gpu_id

    device = torch.device('cuda', device_id)

    transform_list = [
        transforms.Resize((int(img_size), int(img_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    trans = transforms.Compose(transform_list)

    print("Loading dataset...")

    # realset = DataRealGAN(file_root, img_root, trans)
    codeset = DataCodeGAN(file_root,img_root, trans)

    # real_dataloader = iter(DataLoader(realset, batch_size=batch_size, shuffle=False,
    #                                   sampler=InfiniteSamplerWrapper(realset), num_workers=dataloader_workers,
    #                                   pin_memory=True))
    code_dataloader = iter(DataLoader(codeset, batch_size=batch_size, shuffle=False,
                                      sampler=InfiniteSamplerWrapper(codeset), num_workers=dataloader_workers,
                                      pin_memory=True))

    attr_num = codeset.attr_num
    ndf = args.ndf
    ngf = args.ngf
    nz = 256
    nlr = args.lr
    # 目前最好的nbeta时0.9
    nbeta1 = 0.9

    # from model_s import Generator, Discriminator
    # netG = Generator_v2(ngf=ngf,nz=nz, im_size=img_size, attr_num=attr_num,init_condition_dim=nz) #gan_v2
    if args.version == 'v1':
        netG = V1.Generator_v2(nz=nz, im_size=img_size, attr_num=attr_num) #gan_v1
        netG.apply(V1.weights_init)

        netD = V1.Discriminator_v2(ndf=ndf, im_size=img_size, attr_num=attr_num)
        netD.apply(V1.weights_init)

        netC = V1.Classifier(attr_num, backbone=args.class_backbone)
    elif args.version == 'v2':
        netG = V2.Generator_v2(ngf=ngf, nz=nz, im_size=img_size, attr_num=attr_num, init_condition_dim=nz) #gan_v2
        netG.apply(V2.weights_init)

        netD = V2.Discriminator_v2(ndf=ndf, im_size=img_size, attr_num=attr_num)
        netD.apply(V2.weights_init)

        netC = V2.Classifier(attr_num, backbone=args.class_backbone)
    else:
        netG = V2.Generator_v2(ngf=ngf, nz=nz, im_size=img_size, attr_num=attr_num, init_condition_dim=nz)  # gan_v2
        netG.apply(V2.weights_init)

        netD = V2.Discriminator_v2(ndf=ndf, im_size=img_size, attr_num=attr_num)
        netD.apply(V2.weights_init)

        netC = ConditionCA(attr_nums=attr_num, mode='singleton')

    netG.to(device)
    netD.to(device)

    netC.to(device)

    avg_param_G = copy_G_params(netG)

    _, _, fixed_label = next(code_dataloader)
    fixed_code = torch.randn(fixed_label.shape[0],nz).to(device)
    fixed_label = fixed_label.to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999)) # 最好的是使用0.0001
    if args.trainable_cls:
        optimizerC = optim.Adam(netC.parameters(), lr=nlr, betas=(0.9, 0.999))
    MSE = nn.MSELoss(reduction='mean')

    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['g'].items()})
        netD.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['d'].items()})
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt

    if cls_ckpt != 'None':
        ckpt = torch.load(cls_ckpt, map_location=lambda storage, loc: storage.cuda(args.gpu_id))
        netC.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})

    for param in netC.parameters():
        param.requires_grad = False

    if multi_gpu:
        netG = nn.DataParallel(netG,device_ids=[0,2,3])
        netD = nn.DataParallel(netD,device_ids=[0,2,3])
        netC = nn.DataParallel(netC,device_ids=[0,2,3])

    total = 0
    hit = 0
    cls_total = 0
    cls_hit = 0
    for iteration in tqdm(range(current_iteration, total_iterations + 1)):
        # real_image, _, real_label = next(real_dataloader)
        # real_image = real_image.to(device)
        # real_label = real_label.to(device)

        target_img, _, label = next(code_dataloader)
        code = torch.randn(label.shape[0],nz).to(device)
        label = label.to(device)
        target_img = target_img.to(device)

        fake_images = netG(code, label)

        real_image = DiffAugment(target_img, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]

        ## 2. train Discriminator
        netD.zero_grad()

        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image, label, percept, type="real")
        train_d(netD, [fi.detach() for fi in fake_images], label, percept, type="fake")
        optimizerD.step()

        ## 3. train Generator
        netG.zero_grad()
        netC.eval()
        pred_g = netD(fake_images, label,"fake")

        big_cls_loss = 0
        small_cls_loss = 0
        fake_feat_b, cls_out = netC(F.interpolate(fake_images[0], 224))
        for j in range(len(attr_num)):
            loss = F.cross_entropy(cls_out[j], label[:, j], ignore_index=-1)
            if not torch.isnan(loss):
                big_cls_loss += loss
        fake_feat_s, s_cls_out = netC(F.interpolate(fake_images[1], 224))
        for j in range(len(attr_num)):
            loss = F.cross_entropy(s_cls_out[j], label[:, j], ignore_index=-1)
            if not torch.isnan(loss):
                small_cls_loss += loss
        fake_feat = 0.5 * torch.cat(fake_feat_b,dim=1) + 0.5 * torch.cat(fake_feat_s,dim=1)
        with torch.no_grad():
            target_feat, target_cls_out = netC(F.interpolate(target_img,224))

        # 计算属性预测准确率
        for j in range(len(attr_num)):
            for b in range(fake_images[0].shape[0]):
                gt = label[b, j]
                if gt != -1:
                    total += 1
                    pred = torch.argmax(cls_out[j][b])
                    if pred == gt:
                        hit += 1

        err_g = -pred_g.mean() +args.cls_weight * big_cls_loss + args.cls_weight * small_cls_loss + \
            MSE(fake_feat,torch.cat(target_feat,dim=1).detach())
        # err_g = -pred_g.mean()

        if args.percept:
            g_per = percept(fake_images[0], target_img).sum() + \
                    percept(fake_images[1], F.interpolate(target_img, fake_images[1].shape[2])).sum()
            err_g += g_per
        if args.reconstruct:
            g_rec= F.l1_loss(fake_images[0],target_img) + F.l1_loss(fake_images[1],F.interpolate(target_img, fake_images[1].shape[2]))
            err_g+=g_rec
        err_g.backward()
        optimizerG.step()

        if args.trainable_cls:
            netC.train()
            netC.zero_grad()
            target_cls_loss = 0
            _,target_cls_out = netC(F.interpolate(target_img,224))
            for j in range(len(attr_num)):
                loss = F.cross_entropy(target_cls_out[j], label[:, j], ignore_index=-1)
                if not torch.isnan(loss):
                    target_cls_loss += loss
            target_cls_loss.backward()
            optimizerC.step()

        # # 计算属性预测准确率
        # for j in range(len(attr_num)):
        #     for b in range(target_img.shape[0]):
        #         gt = label[b, j]
        #         if gt != -1:
        #             cls_total += 1
        #             pred = torch.argmax(target_cls_out[j][b])
        #             if pred == gt:
        #                 cls_hit += 1

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f    cls_acc: %.5f " % (err_dr, err_g.item(), hit / total))
            # print("GAN: loss d: %.5f    loss g: %.5f" % (err_dr, err_g.item()))
            writer.add_scalar('loss d', err_dr, iteration)
            writer.add_scalar('loss g', err_g.item(), iteration)
            writer.add_scalar('cls acc', hit / total, iteration)
            writer.flush()
            total = 0
            hit = 0

        if iteration % (save_interval * 10) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_code, fixed_label)[0].add(1).mul(0.5),
                                  saved_image_folder + '/%d.jpg' % iteration,
                                  nrow=4)
                vutils.save_image(torch.cat([
                    F.interpolate(real_image, 128),
                    rec_img_all, rec_img_small,
                    rec_img_part]).add(1).mul(0.5), saved_image_folder + '/rec_%d.jpg' % iteration)
            load_params(netG, backup_para)

        if iteration % (save_interval * 50) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            # torch.save({'g': netG.state_dict(), 'd': netD.state_dict(),'c':netC.state_dict()}, saved_model_folder + '/%d.pth' % iteration)
            torch.save({'g': netG.state_dict(), 'd': netD.state_dict()}, saved_model_folder + '/%d.pth' % iteration)
            load_params(netG, backup_para)
            # torch.save({'g': netG.state_dict(),
            #             'd': netD.state_dict(),
            #             'c':netC.state_dict(),
            #             'g_ema': avg_param_G,
            #             'opt_g': optimizerG.state_dict(),
            #             'opt_d': optimizerD.state_dict()}, saved_model_folder + '/all_%d.pth' % iteration)
            torch.save({'g': netG.state_dict(),
                        'd': netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder + '/all_%d.pth' % iteration)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')
    parser.add_argument('--file_root', type=str, required=True, help='Path for pre-processed files')
    parser.add_argument('--img_root', type=str, required=True, help='Path for raw images')
    parser.add_argument('--save_root', type=str, required=True, help='Path for ckpt')
    parser.add_argument('--gpu_id', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--img_size', type=int, default=256, help='image resolution')
    parser.add_argument('--ngf', type=int, default=64, help='channel of feature map')
    parser.add_argument('--ndf', type=int, default=64, help='channel of feature map')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--class_ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--class_backbone', type=str, default='alexnet',
                        help='backbone of classifier, alexnet resnet or vit')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--cls_weight', type=float, default=0.5)
    parser.add_argument('--percept', action='store_true', help='use perceptual loss')
    parser.add_argument('--reconstruct', action='store_true', help='use l1 loss')
    # TODO 添加mse的开关
    parser.add_argument('--trainable_cls', action='store_true', help='trainable classifier')
    parser.add_argument('--version', choices=['v1','v2','v2cca'], help='gan version')
    parser.add_argument('--visual_dir', type=str, required=False, help='Path for tensorboard')
    args = parser.parse_args()
    print(args)
    train(args)
