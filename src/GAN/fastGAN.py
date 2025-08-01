import math

import torch
import torch.nn as nn
import torchvision
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

import random

from torchinfo import summary

seq = nn.Sequential


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)


def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))


class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                  conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                  conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
            convTranspose2d(nz, channel * 2, 4, 1, 0, bias=False),
            batchNorm2d(channel * 2), GLU())

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        batchNorm2d(out_planes * 2), GLU())
    return block


def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
        # convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes * 2), GLU(),
        conv2d(out_planes, out_planes * 2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes * 2), GLU()
    )
    return block


class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024):
        super(Generator, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])

        self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.feat_32 = UpBlockComp(nfc[16], nfc[32])
        self.feat_64 = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)

        if im_size > 256:
            self.feat_512 = UpBlockComp(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

    def forward(self, input):

        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)

        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))

        feat_128 = self.se_128(feat_8, self.feat_128(feat_64))

        feat_256 = self.se_256(feat_16, self.feat_256(feat_128))

        if self.im_size == 256:
            return [self.to_big(feat_256), self.to_128(feat_128)]

        feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
        if self.im_size == 512:
            return [self.to_big(feat_512), self.to_128(feat_128)]

        feat_1024 = self.feat_1024(feat_512)

        im_128 = torch.tanh(self.to_128(feat_128))
        im_1024 = torch.tanh(self.to_big(feat_1024))

        return [im_1024, im_128]


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2)
        )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ndf)

        if im_size == 1024:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                batchNorm2d(nfc[512]),
                nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 512:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 256:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True))

        self.down_4 = DownBlockComp(nfc[512], nfc[256])
        self.down_8 = DownBlockComp(nfc[256], nfc[128])
        self.down_16 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64], nfc[32])
        self.down_64 = DownBlockComp(nfc[32], nfc[16])

        self.rf_big = nn.Sequential(
            conv2d(nfc[16], nfc[8], 1, 1, 0, bias=False),
            batchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[8], 1, 4, 1, 0, bias=False))

        self.se_2_16 = SEBlock(nfc[512], nfc[64])
        self.se_4_32 = SEBlock(nfc[256], nfc[32])
        self.se_8_64 = SEBlock(nfc[128], nfc[16])

        self.down_from_small = nn.Sequential(
            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256], nfc[128]),
            DownBlock(nfc[128], nfc[64]),
            DownBlock(nfc[64], nfc[32]), )

        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)

    def forward(self, imgs, label, part=None):
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]

        feat_2 = self.down_from_big(imgs[0])
        feat_4 = self.down_4(feat_2)
        feat_8 = self.down_8(feat_4)

        feat_16 = self.down_16(feat_8)
        feat_16 = self.se_2_16(feat_2, feat_16)

        feat_32 = self.down_32(feat_16)
        feat_32 = self.se_4_32(feat_4, feat_32)

        feat_last = self.down_64(feat_32)
        feat_last = self.se_8_64(feat_8, feat_last)

        # rf_0 = torch.cat([self.rf_big_1(feat_last).view(-1),self.rf_big_2(feat_last).view(-1)])
        # rff_big = torch.sigmoid(self.rf_factor_big)
        rf_0 = self.rf_big(feat_last).view(-1)

        feat_small = self.down_from_small(imgs[1])
        # rf_1 = torch.cat([self.rf_small_1(feat_small).view(-1),self.rf_small_2(feat_small).view(-1)])
        rf_1 = self.rf_small(feat_small).view(-1)

        if label == 'real':
            rec_img_big = self.decoder_big(feat_last)
            rec_img_small = self.decoder_small(feat_small)

            assert part is not None
            rec_img_part = None
            if part == 0:
                rec_img_part = self.decoder_part(feat_32[:, :, :8, :8])
            if part == 1:
                rec_img_part = self.decoder_part(feat_32[:, :, :8, 8:])
            if part == 2:
                rec_img_part = self.decoder_part(feat_32[:, :, 8:, :8])
            if part == 3:
                rec_img_part = self.decoder_part(feat_32[:, :, 8:, 8:])

            return torch.cat([rf_0, rf_1]), [rec_img_big, rec_img_small, rec_img_part]

        return torch.cat([rf_0, rf_1])


class Discriminator_v2(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512, attr_num=None):
        super(Discriminator_v2, self).__init__()
        self.ndf = ndf
        self.im_size = im_size
        self.attr_num = attr_num

        nfc_multi = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ndf)

        if im_size == 1024:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                batchNorm2d(nfc[512]),
                nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 512:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 256:
            self.down_from_big = nn.Sequential(
                conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True))

        self.condition_2 = ConditionEncoder(attr_num, 256 ** 2, '2d',256)  # channel: 1 * len(attr_num)
        self.condition_4 = ConditionEncoder(attr_num, 128 ** 2 * 2, '2d',128)  # channel: 2 * len(attr_num)
        self.condition_8 = ConditionEncoder(attr_num, 64 ** 2 * 4, '2d',64)  # channel: 4 * len(attr_num)
        self.condition_16 = ConditionEncoder(attr_num, 32 ** 2 * 8, '2d',32)  # channel: 8 * len(attr_num)
        self.condition_32 = ConditionEncoder(attr_num, 16 ** 2 * 16, '2d',16)  # channel: 16 * len(attr_num)
        self.condition_64 = ConditionEncoder(attr_num, 8 ** 2 * 32, '2d',8)  # channel: 32 * len(attr_num)

        self.down_4 = DownBlockComp(nfc[512] + len(attr_num), nfc[256])
        self.down_8 = DownBlockComp(nfc[256] + 2 * len(attr_num), nfc[128])
        self.down_16 = DownBlockComp(nfc[128] + 4 * len(attr_num), nfc[64])
        self.down_32 = DownBlockComp(nfc[64] + 8 * len(attr_num), nfc[32])
        self.down_64 = DownBlockComp(nfc[32] + 16 * len(attr_num), nfc[16])

        self.rf_big = nn.Sequential(
            conv2d(nfc[16] + 32 * len(attr_num), nfc[8], 1, 1, 0, bias=False),
            batchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
            conv2d(nfc[8], 1, 4, 1, 0, bias=False))

        self.se_2_16 = SEBlock(nfc[512] + len(attr_num), nfc[64] + 8 * len(attr_num))
        self.se_4_32 = SEBlock(nfc[256] + 2 * len(attr_num), nfc[32] + 16 * len(attr_num))
        self.se_8_64 = SEBlock(nfc[128] + 4 * len(attr_num), nfc[16] + 32 * len(attr_num))

        self.down_from_small = nn.Sequential(
            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.condition_4s = ConditionEncoder(attr_num, 64 ** 2 * 2, '2d',64)
        self.condition_8s = ConditionEncoder(attr_num, 32 ** 2 * 4, '2d', 32)  # channel: 4 * len(attr_num)
        self.condition_16s = ConditionEncoder(attr_num, 16 ** 2 * 8, '2d', 16)  # channel: 8 * len(attr_num)
        self.condition_32s = ConditionEncoder(attr_num, 8 ** 2 * 16, '2d', 8)  # channel: 16 * len(attr_num)

        self.down_8s = DownBlock(nfc[256] + 2 * len(attr_num), nfc[128])
        self.down_16s = DownBlock(nfc[128] + 4 * len(attr_num), nfc[64])
        self.down_32s = DownBlock(nfc[64] + 8 * len(attr_num), nfc[32])

        self.rf_small = conv2d(nfc[32] + 16 * len(attr_num), 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(nfc[16] + 32 * len(attr_num), nc)
        self.decoder_part = SimpleDecoder(nfc[32] + 16 * len(attr_num), nc)
        self.decoder_small = SimpleDecoder(nfc[32] + 16 * len(attr_num), nc)

    def forward(self, imgs, y, label='real', part=None):
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]

        feat_2 = self.down_from_big(imgs[0])
        con_2 = self.condition_2(y)
        feat_2 = torch.cat([feat_2, con_2], dim=1)

        feat_4 = self.down_4(feat_2)
        con_4 = self.condition_4(y)
        feat_4 = torch.cat([feat_4, con_4], dim=1)

        feat_8 = self.down_8(feat_4)
        con_8 = self.condition_8(y)
        feat_8 = torch.cat([feat_8, con_8], dim=1)

        feat_16 = self.down_16(feat_8)
        con_16 = self.condition_16(y)
        feat_16 = torch.cat([feat_16, con_16], dim=1)
        feat_16 = self.se_2_16(feat_2, feat_16)

        feat_32 = self.down_32(feat_16)
        con_32 = self.condition_32(y)
        feat_32 = torch.cat([feat_32, con_32], dim=1)
        feat_32 = self.se_4_32(feat_4, feat_32)

        #  feat_last的大小为(B,512,8,8)
        feat_last = self.down_64(feat_32)
        con_last = self.condition_64(y)
        feat_last = torch.cat([feat_last, con_last], dim=1)
        feat_last = self.se_8_64(feat_8, feat_last)

        # rf_0 = torch.cat([self.rf_big_1(feat_last).view(-1),self.rf_big_2(feat_last).view(-1)])
        # rff_big = torch.sigmoid(self.rf_factor_big)
        rf_0 = self.rf_big(feat_last).view(-1)

        # feat_small的大小为(B,256,8,8)
        feat_4s = self.down_from_small(imgs[1])
        con_4s = self.condition_4s(y)
        feat_4s = torch.cat([feat_4s, con_4s], dim=1)

        feat_8s = self.down_8s(feat_4s)
        con_8s = self.condition_8s(y)
        feat_8s = torch.cat([feat_8s, con_8s], dim=1)

        feat_16s = self.down_16s(feat_8s)
        con_16s = self.condition_16s(y)
        feat_16s = torch.cat([feat_16s, con_16s], dim=1)

        feat_small = self.down_32s(feat_16s)
        con_small = self.condition_32s(y)
        feat_small = torch.cat([feat_small, con_small], dim=1)

        # rf_1 = torch.cat([self.rf_small_1(feat_small).view(-1),self.rf_small_2(feat_small).view(-1)])
        rf_1 = self.rf_small(feat_small).view(-1)

        if label == 'real':
            rec_img_big = self.decoder_big(feat_last)
            rec_img_small = self.decoder_small(feat_small)

            assert part is not None
            rec_img_part = None
            if part == 0:
                rec_img_part = self.decoder_part(feat_32[:, :, :8, :8])
            if part == 1:
                rec_img_part = self.decoder_part(feat_32[:, :, :8, 8:])
            if part == 2:
                rec_img_part = self.decoder_part(feat_32[:, :, 8:, :8])
            if part == 3:
                rec_img_part = self.decoder_part(feat_32[:, :, 8:, 8:])

            return torch.cat([rf_0, rf_1]), [rec_img_big, rec_img_small, rec_img_part]

        return torch.cat([rf_0, rf_1])


class AttrClassifier(nn.Module):
    def __init__(self, in_channel, attr_num):
        super(AttrClassifier, self).__init__()
        self.attr_num = attr_num
        attr_cls = []
        for i in range(len(self.attr_num)):
            attr_cls.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, 2 * in_channel, 1, 1),
                    batchNorm2d(2 * in_channel),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(2 * in_channel, 2 * in_channel, 1, 1),
                    batchNorm2d(2 * in_channel),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(2 * in_channel, attr_num[i], 1, 1)
                )
            )
        self.attr_cls = nn.ModuleList(attr_cls)
        self.gap_cls = nn.AdaptiveAvgPool2d(1)

    def forward(self, feat):
        # (B,in_channel,8,8) -> (B,in_channel,1,1)
        cls_feat = self.gap_cls(feat)
        cls_out = []
        for layer in self.attr_cls:
            cls_out.append(layer(cls_feat).squeeze())

        # [(B,attr_num[0])...(B,attr_num[i])]
        return cls_out


class Generator_v2(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024, attr_num=None, init_condition_dim=64, downCode_dim=256):
        super(Generator_v2, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.im_size = im_size
        self.attr_num = attr_num

        self.code_down = DownFeature(nz, downCode_dim)
        self.condition_embedd = ConditionEncoder(attr_num, init_condition_dim, '1d')

        self.init = InitLayer(downCode_dim + len(attr_num) * init_condition_dim, channel=nfc[4])  # => (1024, 4, 4)

        self.feat_8 = UpBlockComp(nfc[4], nfc[8])  # => (512, 8, 8)
        self.feat_16 = UpBlock(nfc[8] + 128 * len(attr_num), nfc[16])  # => (256, 16, 16)
        self.feat_32 = UpBlockComp(nfc[16], nfc[32])  # => (128, 32, 32)
        self.feat_64 = UpBlock(nfc[32] + 32 * len(attr_num), nfc[64])  # => (128, 64, 64)
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])  # => (64, 128, 128)
        self.feat_256 = UpBlock(nfc[128] + 8 * len(attr_num), nfc[256])  # => (32, 256, 256)

        self.condition_8 = ConditionEncoder(attr_num, (8 ** 2) * 128, '2d',8)
        self.condition_32 = ConditionEncoder(attr_num, (32 ** 2) * 32, '2d',32)
        self.condition_128 = ConditionEncoder(attr_num, (128 ** 2) * 8, '2d',128)

        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8] + 128 * len(attr_num), nfc[128] + 8 * len(attr_num))
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_128 = conv2d(nfc[128] + 8 * len(attr_num), nc, 1, 1, 0, bias=False)
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)

        if im_size > 256:
            self.feat_512 = UpBlockComp(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

    def forward(self, code, y):
        """z:latent code, y:label list"""

        down_code = self.code_down(code)
        con_embed = self.condition_embedd(y)
        conditional_code = torch.cat([down_code, con_embed], dim=1)

        feat_4 = self.init(conditional_code)
        feat_8 = self.feat_8(feat_4)
        con_8 = self.condition_8(y)

        feat_8 = torch.cat([feat_8, con_8], dim=1)
        feat_16 = self.feat_16(feat_8)

        feat_32 = self.feat_32(feat_16)
        con_32 = self.condition_32(y)
        feat_32 = torch.cat([feat_32,con_32],dim=1)

        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))

        feat_128 = self.feat_128(feat_64)
        con_128 = self.condition_128(y)
        feat_128 = torch.cat([feat_128,con_128],dim=1)
        feat_128 = self.se_128(feat_8, feat_128)

        feat_256 = self.se_256(feat_16, self.feat_256(feat_128))

        if self.im_size == 256:
            return [self.to_big(feat_256), self.to_128(feat_128)]

        feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
        if self.im_size == 512:
            return [self.to_big(feat_512), self.to_128(feat_128)]

        feat_1024 = self.feat_1024(feat_512)

        im_128 = torch.tanh(self.to_128(feat_128))
        im_1024 = torch.tanh(self.to_big(feat_1024))

        return [im_1024, im_128]


class DownFeature(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownFeature, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.down = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel)
        )

    def forward(self, x):
        x = self.down(x)

        return x


class ConditionEncoder(nn.Module):
    def __init__(self, attr_num, embedding_dim, type, condition_dim=None):
        super(ConditionEncoder, self).__init__()
        self.attr_num = attr_num
        self.nembedding = embedding_dim
        self.type = type
        self.condition_dim =condition_dim
        embed_list = []
        for i in range(len(attr_num)):
            embed_list.append(
                nn.Sequential(
                    # 将属性值数量+1所对应的下标作为处理属性缺失的下标, 这个label将不会学习embedding
                    # nn.Embedding(attr_num[i] + 1, self.nembedding, padding_idx=attr_num[i])
                    nn.Embedding(attr_num[i] + 1, self.nembedding)
                )
            )
        self.embedding = nn.ModuleList(embed_list)

    def forward(self, label):
        condition_embedd_list = []
        for i, module in enumerate(self.embedding):
            label_i_ = label[:, i]
            # 将表示属性缺失的-1替换为属性值数量+1所对应的下标
            condition_embedd_list.append(module(torch.where(label_i_ == -1, self.attr_num[i], label_i_)))
        if self.type == '1d':
            condition_embedd = torch.cat(condition_embedd_list, dim=1)
        elif self.type == '2d':
            b = condition_embedd_list[0].shape[0]
            condition_embedd = torch.cat(condition_embedd_list, dim=1).view(b, -1, self.condition_dim,
                                                                            self.condition_dim)
        return condition_embedd


class Classifier(nn.Module):
    """
    Extract attribute-specific embeddings and add attribute predictor for each.
    Args:
        attr_nums: 1-D list of numbers of attribute values for each attribute
        backbone: String that indicate the name of pretrained backbone
        dim_chunk: int, the size of each attribute-specific embedding
    """

    def __init__(self, attr_nums, backbone='alexnet', dim_chunk=340):
        super(Classifier, self).__init__()

        self.attr_nums = attr_nums
        if backbone == 'alexnet':
            self.backbone = torchvision.models.alexnet(weights='DEFAULT')
            self.backbone.classifier = self.backbone.classifier[:-2]
            dim_init = 4096
        if backbone == 'resnet':
            self.backbone = torchvision.models.resnet18(weights='DEFAULT')
            self.backbone.fc = nn.Sequential()
            dim_init = 512
        if backbone == 'vit':
            self.backbone = torchvision.models.vit_b_16(weights='DEFAULT')
            self.backbone.heads = nn.Sequential()
            dim_init = 768

        dis_proj = []
        for i in range(len(attr_nums)):
            dis_proj.append(nn.Sequential(
                nn.Linear(dim_init, dim_chunk),
                nn.ReLU(),
                nn.Linear(dim_chunk, dim_chunk)
            )
            )
        self.dis_proj = nn.ModuleList(dis_proj)

        attr_classifier = []
        for i in range(len(attr_nums)):
            attr_classifier.append(nn.Sequential(
                nn.Linear(dim_chunk, attr_nums[i]))
            )
        self.attr_classifier = nn.ModuleList(attr_classifier)

    def forward(self, img):
        """
        Returns:
            dis_feat: a list of extracted attribute-specific embeddings
            attr_classification_out: a list of classification prediction results for each attribute
        """
        feat = self.backbone(img)
        dis_feat = []
        for layer in self.dis_proj:
            dis_feat.append(layer(feat))

        attr_classification_out = []
        for i, layer in enumerate(self.attr_classifier):
            attr_classification_out.append(layer(dis_feat[i]).squeeze())
        return dis_feat, attr_classification_out


class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""

    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * 32)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
                batchNorm2d(out_planes * 2), GLU())
            return block

        self.main = nn.Sequential(nn.AdaptiveAvgPool2d(8),
                                  upBlock(nfc_in, nfc[16]),
                                  upBlock(nfc[16], nfc[32]),
                                  upBlock(nfc[32], nfc[64]),
                                  upBlock(nfc[64], nfc[128]),
                                  conv2d(nfc[128], nc, 3, 1, 1, bias=False),
                                  nn.Tanh())

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)


from random import randint


def random_crop(image, size):
    h, w = image.shape[2:]
    ch = randint(0, h - size - 1)
    cw = randint(0, w - size - 1)
    return image[:, :, ch:ch + size, cw:cw + size]


class TextureDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(TextureDiscriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4: 16, 8: 8, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ndf)

        self.down_from_small = nn.Sequential(
            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(nfc[256], nfc[128]),
            DownBlock(nfc[128], nfc[64]),
            DownBlock(nfc[64], nfc[32]), )
        self.rf_small = nn.Sequential(
            conv2d(nfc[16], 1, 4, 1, 0, bias=False))

        self.decoder_small = SimpleDecoder(nfc[32], nc)

    def forward(self, img, label):
        img = random_crop(img, size=128)

        feat_small = self.down_from_small(img)
        rf = self.rf_small(feat_small).view(-1)

        if label == 'real':
            rec_img_small = self.decoder_small(feat_small)

            return rf, rec_img_small, img

        return rf


if __name__ == '__main__':
    attr_num = [16, 17, 19, 14, 10, 15, 2, 11, 16, 7, 9, 15]
    # attr_num = [37, 81, 84]
    batch = 16
    shape = len(attr_num) * 340
    pic = torch.randn([batch,3,256,256])
    code = torch.randn([batch, shape])
    label = torch.randint(-1, 3, [batch, len(attr_num)])

    g = Generator_v2(ngf=64, nz=shape, im_size=256, attr_num=attr_num,init_condition_dim=64,downCode_dim=256)
    # print(label)
    # d = Discriminator_v2(ndf=64, im_size=256, attr_num=attr_num)
    summary(g, input_data=[code,label], col_names=['input_size', 'output_size', 'params_percent'])
    # summary(d, input_data=[pic,label,1], col_names=['input_size', 'output_size', 'params_percent'])
