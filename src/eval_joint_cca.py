"""
输入属性修改后的特征,用这个特征结合目标标签输入到GAN中,获得合成图后再提取一次特征,用这个特征进行检索
"""
import argparse
import os
import numpy as np
from tqdm import tqdm
import faiss
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from dataloader import Data, DataQuery
from models.condition_model import ConditionCA, ManipulateBlock, ManipulateBlockAAC
from models.model import Extractor, MemoryBlock
from argument_parser import add_base_args, add_eval_args, add_gan_args
from utils import split_labels, compute_NDCG, get_target_attr
import constants as C
from GAN.fastGAN_v2 import Generator_v2

torch.manual_seed(330)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_base_args(parser)
    add_eval_args(parser)
    add_gan_args(parser)
    args = parser.parse_args()
    if not args.use_cpu and not torch.cuda.is_available():
        print('Warning: Using CPU')
        args.use_cpu = True
    else:
        torch.cuda.set_device(args.gpu_id)

    file_root = args.file_root
    img_root_path = args.img_root

    # load dataset
    print('Loading gallery...')
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

    attr_num = gallery_data.attr_num
    ndf = 64
    ngf = 64
    # nz = 340 * len(attr_num)
    nz = 256
    img_size = 256

    model = ConditionCA(attr_nums=gallery_data.attr_num, mode='singleton')
    if args.manip_type == 'cam':
        memory = ManipulateBlock(attr_num=gallery_data.attr_num, hidden_dim=model.hidden_dim, dim_chunk=model.dim_chunk)
    elif args.manip_type == 'aac':
        memory = ManipulateBlockAAC(attr_num=gallery_data.attr_num, hidden_dim=model.hidden_dim,
                           dim_chunk=model.dim_chunk)
    else:
        memory = MemoryBlock(attr_nums=gallery_data.attr_num, dim_chunk=model.dim_chunk)

    generator = Generator_v2(ngf=ngf,nz=nz, im_size=img_size, attr_num=attr_num,init_condition_dim=256)
    # generator = Generator_v2(nz=nz, im_size=img_size, attr_num=attr_num)

    if args.load_pretrained_extractor:
        print('load {path} \n'.format(path=args.load_pretrained_extractor))
        extra_ckpt = torch.load(args.load_pretrained_extractor, map_location=lambda storage, loc: storage.cuda(args.gpu_id))
        model.load_state_dict({k.replace('module.', ''): v for k, v in extra_ckpt.items()})
        del extra_ckpt
    else:
        print('Pretrained extractor not provided. Use --load_pretrained_extractor or the model will be randomly initialized.')
    if args.load_pretrained_memory:
        print('load {path} \n'.format(path=args.load_pretrained_memory))
        ckpt = torch.load(args.load_pretrained_memory, map_location=lambda storage, loc: storage.cuda(args.gpu_id))
        memory.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    else:
        print('Pretrained memory not provided. Use --load_pretrained_memory or the model will be randomly initialized.')

    if args.load_gan:
        print('load {path} \n'.format(path=args.load_gan))
        checkpoint = torch.load(args.load_gan, map_location=lambda storage, loc : storage.cuda(args.gpu_id))
        # Remove prefix `module`.
        checkpoint['g'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
        generator.load_state_dict(checkpoint['g'])
        # model.load_state_dict(checkpoint['c'])
        # print('load cls checkpoint success')
        # load_params(net_ig, checkpoint['g_ema'])

        # net_ig.eval()
        print('load gan checkpoint success')
        del checkpoint

    if not os.path.exists(args.feat_dir):
        os.makedirs(args.feat_dir)

    if not args.use_cpu:
        model.cuda()
        memory.cuda()
        generator.cuda()

    model.eval()
    memory.eval()
    generator.eval()

    #indexing the gallery

    gallery_feat_path = os.path.join(args.feat_dir, 'gallery_feats.npy')
    if os.path.exists(gallery_feat_path):
        print('Loading existing gallery features from {path}'.format(path=gallery_feat_path))
        gallery_feat = np.load(gallery_feat_path)
    else:
        print('Computing gallery features...')
        gallery_feat = []
        with torch.no_grad():
            for i, (img, gallery_label) in enumerate(tqdm(gallery_loader)):
                if not args.use_cpu:
                    img = img.cuda()

                dis_feat, _ = model(img)
                gallery_feat.append(F.normalize(torch.cat(dis_feat, 1)).squeeze().cpu().numpy())

        gallery_feat = np.concatenate(gallery_feat, axis=0)
        np.save(gallery_feat_path, gallery_feat)
        print('Saved indexed features at {path}'.format(path=gallery_feat_path))

    #indexing the query
    query_inds = np.loadtxt(os.path.join(file_root, args.query_inds), dtype=int)
    gt_labels = np.loadtxt(os.path.join(file_root, args.gt_labels), dtype=int)
    ref_idxs = np.loadtxt(os.path.join(file_root, args.ref_ids), dtype=int)

    assert (query_inds.shape[0] == gt_labels.shape[0]) and (query_inds.shape[0] == ref_idxs.shape[0])

    print('Loading test queries...')
    query_data = DataQuery(file_root, img_root_path,
                           args.ref_ids, args.query_inds,
                           transforms.Compose([
                               transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                           ]), mode='test')
    query_loader = torch.utils.data.DataLoader(query_data, batch_size=args.batch_size, shuffle=False,
                                   sampler=torch.utils.data.SequentialSampler(query_data),
                                   num_workers=args.num_threads,
                                   drop_last=False)

    query_feat_path = os.path.join(args.feat_dir, 'query_feats.npy')
    fake_feat_path = os.path.join(args.feat_dir, 'fake_feats.npy')
    if os.path.exists(query_feat_path) and os.path.exists(fake_feat_path):
        print('Loading existing query features and fake features from {path}'.format(path=args.feat_dir))
        query_feats = np.load(query_feat_path)
        fake_feats = np.load(fake_feat_path)
    else:
        query_feats_list = []
        fake_feats_list = []
        with torch.no_grad():
            for i, (img, indicator,_,label) in enumerate(tqdm(query_loader)):
                indicator = indicator.float()
                if not args.use_cpu:
                    img = img.cuda()
                    indicator = indicator.cuda()
                    label = label.cuda()

                dis_feat, _ = model(img)
                feat_manip = memory(torch.cat(dis_feat, 1),indicator)

                noise = torch.randn(label.shape[0], 256).cuda()
                fake_imgs = generator(noise,label)

                # #new
                # feat_fake, _ = model(F.interpolate(fake_imgs[0],224))
                # feat_fake = torch.cat(feat_fake, 1)
                #old
                feat_fake_big,_ = model(F.interpolate(fake_imgs[0], 224))
                feat_fake_big = torch.cat(feat_fake_big, 1)
                feat_fake_sml,_ = model(F.interpolate(fake_imgs[1], 224))
                feat_fake_sml = torch.cat(feat_fake_sml, 1)
                feat_fake = 0.5 * feat_fake_big + 0.5 * feat_fake_sml

                query_feats_list.append(feat_manip.cpu().numpy())
                fake_feats_list.append(feat_fake.cpu().numpy())

        query_feats = np.concatenate(query_feats_list, axis=0)
        fake_feats = np.concatenate(fake_feats_list, axis=0)
        np.save(query_feat_path, query_feats)
        print('Saved query features at {path}'.format(path=query_feat_path))
        np.save(fake_feat_path, fake_feats)
        print('Saved fake features at {path}'.format(path=fake_feat_path))

    query_fused_feats = []
    query_feats = torch.tensor(query_feats.reshape(-1, args.dim_chunk * len(gallery_data.attr_num)))
    fake_feats = torch.tensor(fake_feats.reshape(-1, args.dim_chunk * len(gallery_data.attr_num)))

    # feat_manip需要过GAN生成假图,再过提取器放入列表中
    if args.feat_fusion:
        fused_feat = args.fusion_weight * fake_feats + (1 - args.fusion_weight) * query_feats
        query_fused_feats = F.normalize(fused_feat).cpu().numpy()
    else:
        query_fused_feats = F.normalize(fake_feats).cpu().numpy()

    #evaluate the top@k results
    gallery_feat = gallery_feat.reshape(-1, args.dim_chunk * len(gallery_data.attr_num))
    fused_feat = query_fused_feats.reshape(-1, args.dim_chunk * len(gallery_data.attr_num))
    dim = args.dim_chunk * len(gallery_data.attr_num)  # dimension
    num_database = gallery_feat.shape[0]  # number of images in database
    num_query = fused_feat.shape[0]  # number of queries

    database = gallery_feat
    queries = fused_feat
    index = faiss.IndexFlatL2(dim)
    index.add(database)
    k = args.top_k
    _, knn = index.search(queries, k)

    #load the GT labels for all gallery images
    label_data = np.loadtxt(os.path.join(file_root, 'labels_test.txt'), dtype=int)

    #compute top@k acc
    hits = 0
    for q in tqdm(range(num_query)):
        neighbours_idxs = knn[q]
        for n_idx in neighbours_idxs:
            if (label_data[n_idx] == gt_labels[q]).all():
                hits += 1
                break
    print('Top@{k} accuracy: {acc}'.format(k=k, acc=hits/num_query))

    #compute NDCG
    ndcg = []
    ndcg_target = []  # consider changed attribute only
    ndcg_others = []  # consider other attributes

    for q in tqdm(range(num_query)):
        rel_scores = []
        target_scores = []
        others_scores = []

        neighbours_idxs = knn[q]
        indicator = query_inds[q]
        target_attr = get_target_attr(indicator, gallery_data.attr_num)
        target_label = split_labels(gt_labels[q], gallery_data.attr_num)

        for n_idx in neighbours_idxs:
            n_label = split_labels(label_data[n_idx], gallery_data.attr_num)
            # compute matched_labels number
            match_cnt = 0
            others_cnt = 0

            for i in range(len(n_label)):
                if (n_label[i] == target_label[i]).all():
                    match_cnt += 1
                if i == target_attr:
                    if (n_label[i] == target_label[i]).all():
                        target_scores.append(1)
                    else:
                        target_scores.append(0)
                else:
                    if (n_label[i] == target_label[i]).all():
                        others_cnt += 1

            rel_scores.append(match_cnt / len(gallery_data.attr_num))
            others_scores.append(others_cnt / (len(gallery_data.attr_num) - 1))

        ndcg.append(compute_NDCG(np.array(rel_scores)))
        ndcg_target.append(compute_NDCG(np.array(target_scores)))
        ndcg_others.append(compute_NDCG(np.array(others_scores)))

    print('NDCG@{k}: {ndcg}, NDCG_target@{k}: {ndcg_t}, NDCG_others@{k}: {ndcg_o}'.format(k=k,
                                                                                          ndcg=np.mean(ndcg),
                                                                                          ndcg_t=np.mean(ndcg_target),
                                                                                          ndcg_o=np.mean(ndcg_others)))