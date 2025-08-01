# export DATASET_PATH="/home/lgh/data/datasets/DeepFashion"
export DATASET_PATH="/home/lgh/data/datasets/Shopping100k/Images"

# export DATASET_NAME="DeepFashion_FineGrained"
export DATASET_NAME="Shopping100k_subset"

export MODELS_DIR="/home/xxx/data/checkpoint/adde-gan"
export CODE_DIR="/home/xxx/data/code/adde-gan"

export MODLE_NAME="encoder_cca_aac_lr1e-4wExpScheduler_b32"

export ENCODER_NAME="encoder_cca_lr1e-4wExpScheduler_b80"

python ${CODE_DIR}/src/train_attr_manip_cca.py --dataset_name ${DATASET_NAME} \
--file_root ${CODE_DIR}/splits/${DATASET_NAME} --img_root ${DATASET_PATH} \
--load_pretrained_extractor ${MODELS_DIR}/${DATASET_NAME}/ADDE/${ENCODER_NAME}/extractor_best.pkl \
--ckpt_dir ${MODELS_DIR}/${DATASET_NAME}/ADDE/${MODLE_NAME} --batch_size 32 \
--num_threads 6 --visual_dir /home/lgh/data/tf-logs/${DATASET_NAME} --exp_name ${MODLE_NAME} \
--gpus 1 2 --backbone cca --cca_version 3 --model_mode singleton --lr 0.0001 --num_epochs 50
