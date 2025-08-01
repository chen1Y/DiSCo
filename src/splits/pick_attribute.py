import argparse
import os
import random

import numpy as np

random.seed(330)
np.random.seed(330)


def pick_train(args, pick):
    # 读取训练图片的标签
    label = np.loadtxt(os.path.join(args.file_root, f"labels_train.txt"), dtype=int)
    print(f"label shape: {label.shape}")

    attr_num = np.loadtxt(os.path.join(args.file_root, "attr_num.txt"), dtype=int)
    preSum = np.insert(np.cumsum(attr_num), 0, 0)
    for p in pick:
        print(preSum[p - 1], ":", preSum[p] - 1)

    # 将这些属性提取出来
    subLabel = []
    for p in pick:
        subLabel.append(label[:, preSum[p - 1]:preSum[p]])

    label_picked = np.concatenate(subLabel, axis=1)
    print(f"label_picked shape: {label_picked.shape}")
    print(label_picked[0])

    # 将lable_picked 保存为txt, 每个元素按照空格隔开
    np.savetxt(os.path.join(args.save_root, f"labels_train.txt"), label_picked, fmt='%d', delimiter=' ')
    np.savetxt(os.path.join(args.save_root, f"attr_num.txt"), attr_num[[idx - 1 for idx in pick]], fmt='%d',
               delimiter='\n')


def pick_test(args, pick):
    # 读取训练图片的标签
    label = np.loadtxt(os.path.join(args.file_root, f"labels_test.txt"), dtype=int)
    print(f"label shape: {label.shape}")

    attr_num = np.loadtxt(os.path.join(args.file_root, "attr_num.txt"), dtype=int)
    preSum = np.insert(np.cumsum(attr_num), 0, 0)
    print(preSum)
    for p in pick:
        print(preSum[p - 1], ":", preSum[p] - 1)

    # 将这些属性提取出来
    subLabel = []
    for p in pick:
        subLabel.append(label[:, preSum[p - 1]:preSum[p]])

    label_picked = np.concatenate(subLabel, axis=1)
    print(f"label_picked shape: {label_picked.shape}")
    print(label_picked[0])

    # 将lable_picked 保存为txt, 每个元素按照空格隔开
    np.savetxt(os.path.join(args.save_root, f"labels_test.txt"), label_picked, fmt='%d', delimiter=' ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_root", type=str)
    parser.add_argument("--save_root", type=str)
    args = parser.parse_args()

    # pick = [1, 2, 3, 7, 9, 11]
    pick = [2, 3, 5, 8, 9, 11] # datrNet subset
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
    pick_train(args, pick)
    pick_test(args, pick)
