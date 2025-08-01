#!/bin/bash
#export DATASET_NAME="DeepFashion_FineGrained"
export DATASET_NAME="Shopping100k_subset"

export DATASET_PATH="/home/xxx/data/datasets/Shopping100k/Images"
#export DATASET_PATH="/home/xxx/data/datasets/DeepFashion"

export MODELS_DIR="/home/xxx/data/checkpoint/adde-gan"
export CODE_DIR="/home/xxx/data/code/adde-gan"

export MODEL_NAME="encoder_cca_aac_lr1e-4wExpScheduler_b32"

export GAN_NAME="GAN_cca_aac_percept_rec_mse_cls_0.2"

# eval parameters
fusion_weights=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
top_ks=(5 10 20)

for top_k in "${top_ks[@]}"; do
  for fusion_weight in "${fusion_weights[@]}"; do
    echo "----------------------Running top_k: ${top_k}, fusion_weight: ${fusion_weight}-------------------"
    python ${CODE_DIR}/src/eval_joint_cca.py \
    --dataset_name ${DATASET_NAME} --file_root ${CODE_DIR}/splits/${DATASET_NAME} \
    --img_root ${DATASET_PATH} --batch_size 4 \
    --load_pretrained_extractor ${MODELS_DIR}/${DATASET_NAME}/ADDE/${MODEL_NAME}/extractor_best.pkl \
    --load_pretrained_memory ${MODELS_DIR}/${DATASET_NAME}/ADDE/${MODEL_NAME}/memory_best.pkl \
    --load_gan ${MODELS_DIR}/GAN/${DATASET_NAME}/${GAN_NAME}/models/150000.pth \
    --feat_dir ${MODELS_DIR}/${DATASET_NAME}/EVAL/${GAN_NAME}/SINGLE_MANIP \
    --backbone cca --manip_type aac --gpu_id 0 --top_k "${top_k}" --feat_fusion --fusion_weight "${fusion_weight}"
    echo "---------------------------------------------------------------------------------------"
    echo " "
  done
done
