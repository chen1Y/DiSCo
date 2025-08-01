# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
This is script is used to pre-train the disentangled representation learner with attribute clossification task
"""

import argparse
import datetime
import functools
import json

import constants as C
import os
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from argument_parser import add_base_args, add_train_args
from models.condition_model import ConditionCA
from dataloader import Data
from loss_function import hash_labels, TripletSemiHardLoss, FocalLoss
# from attention_pooler import Extractor
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.model import Extractor

torch.manual_seed(100)
ATTR_NAME = ['Collar', 'Color', 'Fastening', 'Neckline', 'Pattern', 'Sleeve']
# ATTR_NAME = ['Category', 'Collar', 'Color', 'Fabric', 'Fastening', 'Fit', 'Gender', 'Neckline', 'Pattern', 'Pocket',
#              'Sleeve', 'Sport']


def train(device, train_loader, model, optimizer, args, focal_loss=None, ):
    avg_total_loss = 0
    model.train()
    criterion = focal_loss if focal_loss is not None else functools.partial(F.cross_entropy, ignore_index=-1)
    for i, sample in enumerate(tqdm(train_loader)):
        img_query, label = sample
        if not args.use_cpu:
            img_query = img_query.to(device)
            label = label.to(device)

        model.zero_grad()
        dis_feat, cls_outs = model(img_query)

        cls_loss = 0
        for j in range(len(train_loader.dataset.attr_num)):
            entropy = criterion(cls_outs[j], label[:, j])
            if not torch.isnan(entropy):
                cls_loss += entropy

        # attr_rank_loss_local
        rank_loss = 0
        for j in range(len(train_loader.dataset.attr_num)):
            rank_loss += TripletSemiHardLoss(label[:, j], F.normalize(dis_feat[j]), margin=args.margin)

        # attr_rank_loss_global
        hash_label = hash_labels(label)
        rank_global_loss = TripletSemiHardLoss(hash_label, F.normalize(torch.cat(dis_feat, 1)),
                                               margin=args.margin)

        total_loss = args.weight_cls * cls_loss + args.weight_label_trip_local * rank_loss + args.weight_label_trip * rank_global_loss
        total_loss.backward()
        optimizer.step()

        avg_total_loss += total_loss

    return avg_total_loss / (i + 1)


def eval(device, test_loader, model, args):
    model.eval()
    attr_num = test_loader.dataset.attr_num

    attr_total = [0] * len(attr_num)
    attr_hit = [0] * len(attr_num)
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader)):
            img_query, label = sample
            img_query = img_query.cuda()
            label = label.cuda()
            _, cls_outs = model(img_query)

            for j in range(len(attr_num)):
                for b in range(img_query.shape[0]):
                    gt = label[b, j]
                    if gt != -1:
                        attr_total[j] += 1
                        pred = torch.argmax(cls_outs[j][b])
                        if pred == gt:
                            attr_hit[j] += 1

    return [hit / total for hit, total in zip(attr_hit, attr_total)]


def main(rank, args):
    dist.init_process_group(
        backend='nccl',
        world_size=args.world_size,
        rank=rank)
    torch.manual_seed(100 + rank)
    if rank == 0:
        writer = SummaryWriter(os.path.join(args.visual_dir, 'tensorboard/' + args.exp_name))

    device_id = args.gpus[rank]
    torch.cuda.set_device(device_id)
    device = torch.device('cuda', device_id)

    file_root = args.file_root
    img_root_path = args.img_root

    # load dataset
    if rank == 0:
        print('Loading dataset...')
    train_data = Data(file_root, img_root_path,
                      transforms.Compose([
                          transforms.Resize((256, 256)),
                          transforms.RandomHorizontalFlip(),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                      ]), 'train')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=args.world_size, rank=rank,
                                                                    shuffle=True, drop_last=True)
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_threads,
                                   sampler=train_sampler, drop_last=True)

    if rank == 0:
        valid_data = Data(file_root, img_root_path,
                          transforms.Compose([
                              transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                          ]), 'test')
        valid_loader = data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_threads,
                                       drop_last=False)

    # create the folder to save log, checkpoints and args config
    if rank == 0:
        if not args.ckpt_dir:
            name = datetime.datetime.now().strftime("%m-%d-%H:%M")
        else:
            name = args.ckpt_dir
        directory = '{name}'.format(name=name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, 'commandline_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    if args.backbone == 'cca':
        print(args.model_mode)
        model = ConditionCA(attr_nums=train_data.attr_num, mode=args.model_mode)

    else:
        model = Extractor(attr_nums=train_data.attr_num, backbone=args.backbone, dim_chunk=340)
    # model = Extractor(attr_nums=train_data.attr_num,backbone='vit',dim_chunk=340)
    # model = Extractor_AP(train_data.attr_num, backbone=args.backbone, dim_chunk=args.dim_chunk)
    # for par in model.backbone.parameters():
    #     par.requires_grad = False
    # for par in model.backbone.encoder.get_submodule('layers')[-2:].parameters():
    #     par.requires_grad = True
    # for par in model.backbone.encoder.ln.parameters():
    #     par.requires_grad = True

    model.to(device)
    model = DDP(model, device_ids=[device_id],
                find_unused_parameters=True if args.dataset_name in ['Shopping100k_subset', 'Shopping100k'] else False)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, betas=(args.momentum, 0.999))
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=args.lr, betas=(args.momentum, 0.999))
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.95, verbose=True)
    # scheduler = lr_scheduler.StepLR(optimizer,args.lr_decay_step,args.lr_decay_rate,verbose=True)

    if args.focalloss:
        focal_loss = FocalLoss(gamma=1, ignore_index=-1, reduction='mean')
    else:
        focal_loss = None

    previous_best_avg_test_acc = 0.0
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        avg_total_loss = train(device, train_loader, model, optimizer, args, focal_loss)
        if rank == 0:
            test_acc = eval(device, valid_loader, model, args)
            avg_test_acc = sum(test_acc) / len(test_acc)
            writer.add_scalar('loss', avg_total_loss, epoch + 1)
            writer.add_scalar('acc', avg_test_acc, epoch + 1)
            for i, acc in enumerate(test_acc):
                writer.add_scalar(f'{ATTR_NAME[i]}_acc', acc, epoch + 1)
            writer.flush()

            # result record
            print('Epoch %d, Train_loss: %.4f,  test_acc: %.4f \n'
                  % (epoch + 1, avg_total_loss, avg_test_acc))

            with open(os.path.join(directory, 'log.txt'), 'a') as f:
                f.write('Epoch %d, Train_loss: %.4f, test_acc: %.4f\n'
                        % (epoch + 1, avg_total_loss, avg_test_acc))

            # # store parameters
            # torch.save(model.state_dict(), os.path.join(directory, "ckpt_%d.pkl" % (epoch + 1)))
            # print('Saved checkpoints at {dir}/ckpt_{epoch}.pkl'.format(dir=directory, epoch=epoch + 1))

            if avg_test_acc > previous_best_avg_test_acc:
                torch.save(model.state_dict(), os.path.join(directory, "extractor_best.pkl"))
                print('Best model in {dir}/extractor_best.pkl'.format(dir=directory))
                previous_best_avg_test_acc = avg_test_acc
        scheduler.step()
        dist.barrier()
    if rank == 0:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_base_args(parser)
    add_train_args(parser)
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--model_mode', type=str, required=True, choices=['branch', 'singleton'], default='branch')
    parser.add_argument('--focalloss', action='store_true')
    args = parser.parse_args()

    args.world_size = len(args.gpus)
    os.environ['MASTER_ADDR'] = 'localhost'  #
    os.environ['MASTER_PORT'] = '8833'  #
    print(args)
    mp.spawn(main, nprocs=args.world_size, args=(args,), join=True)
