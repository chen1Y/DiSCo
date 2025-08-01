# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
This is script is used to joint train the disentangled representation learner and memory block.
"""

import argparse
import datetime
import json
import os

import faiss
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import constants as C
from argument_parser import add_base_args, add_train_args
from dataloader import DataTriplet, DataQuery, Data
from loss_function import hash_labels, TripletSemiHardLoss
from models.condition_model import ConditionCA, ManipulateBlock, ManipulateBlockV2, ManipulateBlockAAC

torch.manual_seed(100)


def train(device, train_loader, model, manip, optimizer, scaler, args):
    avg_total_loss = 0

    model.train()
    manip.train()

    for i, (imgs, one_hots, labels, indicator) in enumerate(tqdm(train_loader)):
        indicator = indicator.float()
        for key in one_hots.keys():
            one_hots[key] = one_hots[key].float()
        if not args.use_cpu:
            for key in imgs.keys():
                imgs[key] = imgs[key].to(device)
                one_hots[key] = one_hots[key].to(device)
                labels[key] = labels[key].to(device)
            indicator = indicator.to(device)

        model.zero_grad()
        manip.zero_grad()
        with autocast():
            feats = {}
            cls_outs = {}
            for key in imgs.keys():
                feats[key], cls_outs[key] = model(imgs[key])

            feat_manip = manip(torch.cat(feats['ref'], 1), indicator)
            feat_manip_split = list(torch.split(feat_manip, args.dim_chunk, dim=1))

            cls_outs_manip = []
            for attr_id, layer in enumerate(model.module.attr_classifier):
                cls_outs_manip.append(layer(feat_manip_split[attr_id]).squeeze())

            # attribute prediction loss
            cls_loss = 0
            for j in range(len(train_loader.dataset.attr_num)):
                for key in imgs.keys():
                    loss = F.cross_entropy(cls_outs[key][j], labels[key][:, j], ignore_index=-1)
                    if not torch.isnan(loss):
                        cls_loss += loss
                loss_att = F.cross_entropy(cls_outs_manip[j], labels['pos'][:, j], ignore_index=-1)
                if not torch.isnan(loss_att):
                    cls_loss += loss_att

            # label_triplet_loss
            hashs = {}
            for key in imgs.keys():
                hashs[key] = hash_labels(labels[key])

            # 提升效果有限， 计算复杂度高，可以考虑不使用
            # label_triplet_loss = TripletSemiHardLoss(torch.cat((hashs['ref'], hashs['pos'], hashs['neg']), 0),
            #                                          torch.cat((F.normalize(torch.cat(feats['ref'], 1)),
            #                                                     F.normalize(feat_manip),
            #                                                     F.normalize(torch.cat(feats['neg'], 1))), 0),
            #                                          margin=args.margin)

            # manipulation_triplet_loss
            criterion_c = nn.TripletMarginLoss(margin=args.margin)
            manip_triplet_loss = criterion_c(F.normalize(feat_manip),
                                             F.normalize(torch.cat(feats['pos'], 1)),
                                             F.normalize(torch.cat(feats['neg'], 1))
                                             )
            total_loss = args.weight_cls * cls_loss + args.weight_manip_trip * manip_triplet_loss
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # total_loss.backward()
        # optimizer.step()

        avg_total_loss += total_loss.data

    return avg_total_loss / (i + 1)


def eval(device, gallery_loader, query_loader, model, manip, args):
    model.eval()
    manip.eval()

    gt_labels = np.loadtxt(os.path.join(args.file_root, 'gt_test.txt'), dtype=int)

    gallery_feat = []
    query_fused_feats = []
    with torch.no_grad():
        # indexing the gallery
        for i, (img, _) in enumerate(tqdm(gallery_loader)):
            if not args.use_cpu:
                img = img.to(device)

            dis_feat, _ = model(img)
            gallery_feat.append(F.normalize(torch.cat(dis_feat, 1)).squeeze().cpu().numpy())

        # load the queries
        for i, (img, indicator, _, _) in enumerate(tqdm(query_loader)):
            indicator = indicator.float()
            if not args.use_cpu:
                img = img.to(device)
                indicator = indicator.to(device)

            dis_feat, _ = model(img)
            feat_manip = manip(torch.cat(dis_feat, 1), indicator)

            query_fused_feats.append(F.normalize(feat_manip).cpu().numpy())

    gallery_feat = np.concatenate(gallery_feat, axis=0).reshape(-1,
                                                                args.dim_chunk * len(gallery_loader.dataset.attr_num))
    fused_feat = np.array(np.concatenate(query_fused_feats, axis=0)).reshape(-1, args.dim_chunk * len(
        gallery_loader.dataset.attr_num))
    dim = args.dim_chunk * len(gallery_loader.dataset.attr_num)  # dimension
    num_query = fused_feat.shape[0]  # number of queries

    database = gallery_feat
    queries = fused_feat
    index = faiss.IndexFlatL2(dim)
    index.add(database)
    k = 30
    _, knn = index.search(queries, k)

    # load the GT labels for all gallery images
    label_data = np.loadtxt(os.path.join(args.file_root, 'labels_test.txt'), dtype=int)

    # compute top@k acc
    hits = 0
    for q in tqdm(range(num_query)):
        neighbours_idxs = knn[q]
        for n_idx in neighbours_idxs:
            if (label_data[n_idx] == gt_labels[q]).all():
                hits += 1
                break
    print('Top@{k} accuracy: {acc}'.format(k=k, acc=hits / num_query))

    return hits / num_query


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

    train_data = DataTriplet(file_root, img_root_path, args.triplet_file,
                             transforms.Compose([
                                 transforms.Resize((256, 256)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                             ]), 'train')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=args.world_size, rank=rank,
                                                                    shuffle=True, drop_last=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_threads,
                                               sampler=train_sampler, drop_last=True)

    if rank == 0:
        query_data = DataQuery(file_root, img_root_path,
                               'ref_test.txt', 'indfull_test.txt',
                               transforms.Compose([
                                   transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                               ]), mode='test')
        query_loader = torch.utils.data.DataLoader(query_data, batch_size=args.batch_size, shuffle=False,
                                                   sampler=torch.utils.data.SequentialSampler(query_data),
                                                   num_workers=args.num_threads,
                                                   drop_last=False)

        gallery_data = Data(file_root, img_root_path,
                            transforms.Compose([
                                transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                            ]), mode='test')

        gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=args.batch_size, shuffle=False,
                                                     sampler=torch.utils.data.SequentialSampler(gallery_data),
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

    model = ConditionCA(attr_nums=train_data.attr_num, mode=args.model_mode)
    if args.cca_version == '1':
        manip_block = ManipulateBlock(attr_num=train_data.attr_num, hidden_dim=model.hidden_dim,
                                      dim_chunk=model.dim_chunk)
    elif args.cca_version == '2':
        manip_block = ManipulateBlockV2(attr_num=train_data.attr_num, hidden_dim=model.hidden_dim,
                                        dim_chunk=model.dim_chunk)
    else :
        manip_block = ManipulateBlockAAC(attr_num=train_data.attr_num, hidden_dim=model.hidden_dim,
                                         dim_chunk=model.dim_chunk)
    model.to(device)
    manip_block.to(device)

    # start training from the pretrained weights if provided
    if args.load_pretrained_extractor:
        print('load %s\n' % args.load_pretrained_extractor)
        ckpt = torch.load(args.load_pretrained_extractor, map_location=device)
        model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    else:
        print(
            'Pretrained extractor not provided. Use --load_pretrained_extractor or the model will be randomly initialized.')

    model.to(device)

    model = DDP(model, device_ids=[device_id],
                find_unused_parameters=True if args.dataset_name in ['Shopping100k', 'Shopping100k_subset'] else False)
    manip_block = DDP(manip_block, device_ids=[device_id],
                      find_unused_parameters=True if args.dataset_name in ['Shopping100k',
                                                                           'Shopping100k_subset'] else False)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(manip_block.parameters()), lr=args.lr,
                                 betas=(args.momentum, 0.999))
    # scheduler = lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.95, verbose=True)
    scaler = torch.cuda.amp.GradScaler()
    previous_best_avg_test_acc = 0.0
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        avg_total_loss = train(device, train_loader, model, manip_block, optimizer, scaler, args)
        if rank == 0:
            avg_test_acc = eval(device, gallery_loader, query_loader, model, manip_block, args)

            writer.add_scalar('loss', avg_total_loss, epoch + 1)
            writer.add_scalar('acc', avg_test_acc, epoch + 1)
            writer.flush()

            # result record
            print('Epoch %d, Cls_loss: %.4f, test_acc: %.4f\n' % (epoch + 1, avg_total_loss, avg_test_acc))

            with open(os.path.join(directory, 'log.txt'), 'a') as f:
                f.write('Epoch %d, Cls_loss: %.4f, test_acc: %.4f\n' % (epoch + 1, avg_total_loss, avg_test_acc))

            # # store parameters
            # torch.save(model.state_dict(), os.path.join(directory, "extractor_ckpt_%d.pkl" % (epoch + 1)))
            # torch.save(manip_block.state_dict(), os.path.join(directory, "memory_ckpt_%d.pkl" % (epoch + 1)))
            # print('Saved checkpoints at {dir}/extractor_{epoch}.pkl, {dir}/memory_{epoch}.pkl'.format(dir=directory,
            #                                                                                           epoch=epoch + 1))
            if avg_test_acc > previous_best_avg_test_acc:
                torch.save(model.state_dict(), os.path.join(directory, "extractor_best.pkl"))
                torch.save(manip_block.state_dict(), os.path.join(directory, "memory_best.pkl"))
                print('Best model in {dir}/extractor_best.pkl and {dir}/memory_best.pkl'.format(dir=directory))
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
    parser.add_argument('--model_mode', type=str, choices=['branch', 'singleton'], default='branch')
    parser.add_argument('--cca_version', type=str, choices=['1', '2','3'], default='1')
    args = parser.parse_args()

    args.world_size = len(args.gpus)
    os.environ['MASTER_ADDR'] = 'localhost'  #
    os.environ['MASTER_PORT'] = '8836'  #
    print(args)
    mp.spawn(main, nprocs=args.world_size, args=(args,), join=True)
