"""
生成splits文件
按顺序选取一张train set里的图片作为ref，按顺序选择一个属性，修改其属性值为另一个值。
若完全符合目标属性值的图片存在于train set里，从中随机选择一个作为pos，再选取一个属性值不同的作为neg
重复上述过程
"""
import argparse
import os
import random

import pandas as pd
import numpy as np
from tqdm import tqdm

random.seed(330)
np.random.seed(330)


def hash_labels(labels):
    """
    将多属性标签转换为哈希值，便于筛选相同的标签样本
    输入
    """
    hash_labels = {}
    # 遍历所有labels_train
    for i in range(labels.shape[0]):
        # convert to string and store into the hashmap
        str_labels = ''.join([str(c) for c in labels[i]])
        get = hash_labels.get(str_labels)
        # 记下下标
        if get is not None:
            hash_labels[str_labels] = get + [i]
        else:
            hash_labels[str_labels] = [i]

    return hash_labels


def get_idx_label(label_vectors, attr_num):
    """
    convert the one-hot attribute labels to the label that indicate GT's position for each attribute
    Args:
        label_vector: one-hot labels
        attr_num: 1-D list of numbers of attribute values for each attribute
    """
    labels = []
    start_idx = 0
    for i in attr_num:
        sub_label = label_vectors[:, start_idx:start_idx + i]
        label = np.argmax(sub_label, axis=1)
        # handle missing labels
        sum_sub_label = np.sum(sub_label, axis=1)
        label[sum_sub_label == 0] = -1
        labels.append(label)
        start_idx += i
    idxed_label = np.stack(labels, axis=1)
    return idxed_label


def generate_train(args):
    # 读取训练图片的标签
    label = np.loadtxt(os.path.join(args.file_root, f"labels_train.txt"), dtype=int)
    print(f"label shape: {label.shape}")
    attr_num = np.loadtxt(os.path.join(args.file_root, "attr_num.txt"), dtype=int)
    print(attr_num)
    # 将one-hot标签转为下标标签
    idxed_label = get_idx_label(label, attr_num)
    print(f"indexed label shape: {label.shape}")
    # 建立hashmap便于查找符合要求的目标样本
    hashmap = hash_labels(idxed_label)
    # 所有图片的下标
    img_idx = set([i for i in range(label.shape[0])])
    ref = []
    pos = []
    neg = []
    ind = []
    for ref_i in tqdm(range(idxed_label.shape[0])):  # 遍历所有图片
        for attr_i in random.sample(range(len(attr_num)), random.randint(1, len(attr_num))):  # 之前的shopping100k
            # for attr_i in random.sample(range(len(attr_num)), random.randint(4, len(attr_num))):  # deepfashion
            if idxed_label[ref_i][attr_i] == -1:  # 跳过missing label
                continue
            for value in random.sample(range(attr_num[attr_i]), random.randint(2, 3)):  # 选择随机个属性shopping100k
            # for value in random.sample(range(attr_num[attr_i]), random.randint(1, 2)):  # 选择随机个属性 之前的 shopping100k
                # for value in random.sample(range(attr_num[attr_i]), random.randint(2, 3)): # 选择所有属性 deepfashion
                if value == idxed_label[ref_i][attr_i]:  # 跳过“修改成原有值”这种情况
                    continue
                # 将属性值修改成另一个，去hashmap里查找是否存在目标
                _target = [str(c) for c in idxed_label[ref_i]]
                _target[attr_i] = str(value)
                target_key = ''.join(_target)
                get = hashmap.get(target_key)
                if get is not None:
                    # 若存在目标，从目标中随机选一个作为pos，从目标的补集中取一个作为neg
                    ref.append(ref_i)
                    pos_i = random.choice(get)
                    pos.append(pos_i)
                    neg.append(random.choice(list(img_idx.difference(set(get)))))
                    ind.append(label[pos_i] - label[ref_i])

    ind = np.concatenate(ind, axis=0).reshape(-1, sum(attr_num))
    pd.DataFrame.from_dict({'ref': ref, 'pos': pos, 'neg': neg}).to_csv(
        os.path.join(args.file_root, "triplet_train.txt"), header=False, index=False, sep=' ')
    pd.DataFrame(ind).to_csv(os.path.join(args.file_root, "triplet_train_ind.txt"), header=False, index=False, sep=' ')


def generate_test(args):
    # 读取训练图片的标签
    label = np.loadtxt(os.path.join(args.file_root, f"labels_test.txt"), dtype=int)
    print(f"label shape: {label.shape}")
    attr_num = np.loadtxt(os.path.join(args.file_root, "attr_num.txt"), dtype=int)
    # 将one-hot标签转为下标标签
    idxed_label = get_idx_label(label, attr_num)
    print(f"indexed label shape: {label.shape}")
    # 建立hashmap便于查找符合要求的目标样本
    hashmap = hash_labels(idxed_label)
    # 所有图片的下标
    ref = []
    gt = []
    ind = []
    for ref_i in tqdm(range(2000)):  # 遍历头2000张图片
        for attr_i in range(len(attr_num)):  # 遍历图片的所有属性
            if idxed_label[ref_i][attr_i] == -1:  # 跳过missing label
                continue

            sample = random.sample(range(attr_num[attr_i]), random.randint(1, int(attr_num[attr_i])))
            for value in sample:  # 遍历属性的所有值
                if value == idxed_label[ref_i][attr_i]:
                    continue
                # 将属性值修改成另一个，去hashmap里查找是否存在目标
                _target = [str(c) for c in idxed_label[ref_i]]
                _target[attr_i] = str(value)
                target_key = ''.join(_target)
                get = hashmap.get(target_key)
                if get is not None:
                    # 若存在目标
                    ref.append(ref_i)
                    gt.append(label[get[0]])  # 记下目标标签
                    ind.append(label[get[0]] - label[ref_i])

    pd.DataFrame(ref).to_csv(
        os.path.join(args.file_root, "ref_test.txt"), header=False, index=False)

    gt = np.concatenate(gt, axis=0).reshape(-1, sum(attr_num))
    pd.DataFrame(gt).to_csv(os.path.join(args.file_root, "gt_test.txt"), header=False, index=False, sep=' ')

    ind = np.concatenate(ind, axis=0).reshape(-1, sum(attr_num))
    pd.DataFrame(ind).to_csv(os.path.join(args.file_root, "indfull_test.txt"), header=False, index=False, sep=' ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_root", type=str)
    args = parser.parse_args()
    generate_train(args)
    # generate_test(args)
