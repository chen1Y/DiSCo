#!/bin/bash

# export DATASET_PATH="/home/xxx/data/datasets/DeepFashion"
export DATASET_PATH="/home/xxx/data/datasets/Shopping100k/Images"

#export DATASET_NAME="DeepFashion_FineGrained"
export DATASET_NAME="Shopping100k_subset"

export MODELS_DIR="/home/xxx/data/checkpoint/adde-gan"
export CODE_DIR="/home/xxx/data/code/adde-gan"

export GAN_NAME="GAN_cca_aac_percept_rec_mse_cls_0.2"

export MODLE_NAME="encoder_cca_aac_lr1e-4wExpScheduler_b32"

python ${CODE_DIR}/src/train_gan.py --file_root ${CODE_DIR}/splits/${DATASET_NAME} --img_root ${DATASET_PATH} \
--name ${GAN_NAME} --lr 0.0001 --iter 150000 \
--class_ckpt ${MODELS_DIR}/${DATASET_NAME}/ADDE/${MODLE_NAME}/extractor_best.pkl \
--class_backbone cca --save_root ${MODELS_DIR}/GAN/${DATASET_NAME} --visual_dir /home/xxx/data/tf-logs/${DATASET_NAME} \
--gpu_id 1 --batch_size 16 --ngf 64 --ndf 64 --percept --reconstruct --version v2cca --cls_weight 0.2

