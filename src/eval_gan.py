import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils
import numpy as np
from PIL import Image,ImageEnhance

import os
import random
import argparse
from tqdm import tqdm

from GAN.fastGAN_v2 import Generator_v2
from utils import get_idx_label

torch.manual_seed(330)
np.random.seed(330)
random.seed(330)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def resize(img):
    return F.interpolate(img, size=256)


def batch_generate(zs, netG, batch=8):
    g_images = []
    with torch.no_grad():
        for i in range(len(zs) // batch):
            g_images.append(netG(zs[i * batch:(i + 1) * batch]).cpu())
        if len(zs) % batch > 0:
            g_images.append(netG(zs[-(len(zs) % batch):]).cpu())
    return torch.cat(g_images)


def batch_save(images, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for i, image in enumerate(images):
        vutils.save_image(image.add(1).mul(0.5), folder_name + '/%d.jpg' % i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate images'
    )
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--save_root', type=str, default=".", help='path to artifacts.')
    parser.add_argument('--name', type=str, default=".", help='experiment name')
    parser.add_argument('--file_root', type=str, required=True, help='Path for pre-processed files')
    parser.add_argument('--gpu_id', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--start_iter', type=int, default=3)
    parser.add_argument('--end_iter', type=int, default=3)

    parser.add_argument('--dist', type=str, default='.')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=7200)
    parser.add_argument('--big', action='store_true')
    parser.add_argument('--img_size', type=int, default=256)
    parser.set_defaults(big=False)
    args = parser.parse_args()

    noise_dim = 256
    ndf = 64
    ngf = 64
    img_size = args.img_size
    attr_num = np.loadtxt(os.path.join(args.file_root, "attr_num.txt"), dtype=int)
    label_data = np.loadtxt(os.path.join(args.file_root, "gt_test.txt"), dtype=int)
    device = torch.device('cuda:%d' % (args.gpu_id))

    # net_ig = Generator_v2(nz=noise_dim, im_size=img_size,attr_num=attr_num)
    net_ig = Generator_v2(ngf=ngf, nz=noise_dim, im_size=img_size, attr_num=attr_num, init_condition_dim=noise_dim)


    for epoch in [50000 * i for i in range(args.start_iter, args.end_iter + 1)]:
        ckpt = f"{args.save_root}/{args.name}/models/{epoch}.pth"
        checkpoint = torch.load(ckpt, map_location=lambda a, b: a)
        # Remove prefix `module`.
        checkpoint['g'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
        net_ig.load_state_dict(checkpoint['g'])
        # load_params(net_ig, checkpoint['g_ema'])

        net_ig.eval()
        print('load checkpoint success, epoch %d' % epoch)

        net_ig.to(device)

        del checkpoint

        dist = 'fid_%s' % (args.name)
        dist = os.path.join(dist, '%d' % epoch)
        os.makedirs(dist, exist_ok=True)

        with torch.no_grad():
            for i in tqdm(range(args.n_sample // args.batch_size)):
                noise = torch.randn(args.batch_size, noise_dim).to(device)
                label = np.stack(
                    [get_idx_label(l, attr_num) for l in
                     label_data[np.random.choice(len(label_data), args.batch_size)]],
                    axis=0)
                label = torch.tensor(label).to(device)
                g_imgs = net_ig(noise,label)[0]
                g_imgs = F.interpolate(g_imgs, 256)
                for j, g_img in enumerate(g_imgs):
                    vutils.save_image(g_img.add(1).mul(0.5).clip(0,1),
                                      os.path.join(dist,
                                                   '%d.png' % (i * args.batch_size + j)))  # , normalize=True, range=(-1,1))