#!/bin/bash

# export DATASET_PATH="/home/xxx/DATA/datasets/Shopping100k/Images"
export DATASET_PATH="/home/xxx/data/datasets/DeepFashion"

export CODE_DIR="/home/xxx/data/code/adde-gan"
export MODELS_DIR="/home/xxx/data/checkpoint/adde-gan"

export DATASET_NAME="DeepFashion_FineGrained"
#export DATASET_NAME="Shopping100k_subset"

export MODEL_NAME="encoder_vit_manip_lr1e-4wExpScheduler_b32"

top_ks=(30 20)

for top_k in "${top_ks[@]}"; do
  echo "----------------------Running top_k: ${top_k}-------------------"
  python ${CODE_DIR}/src/eval.py --dataset_name ${DATASET_NAME} --file_root ${CODE_DIR}/splits/${DATASET_NAME} \
  --img_root ${DATASET_PATH} --load_pretrained_extractor ${MODELS_DIR}/${DATASET_NAME}/ADDE/${MODEL_NAME}/extractor_best.pkl \
  --load_pretrained_memory ${MODELS_DIR}/${DATASET_NAME}/ADDE/${MODEL_NAME}/memory_best.pkl \
  --backbone vit --batch_size 4 --gpu_id 1 --top_k "${top_k}"
  echo "----------------------------------------------------------------"
done