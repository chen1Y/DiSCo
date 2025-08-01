export DATASET_PATH="/home/xxx/data/datasets/DeepFashion"
#export DATASET_PATH="/home/xxx/data/datasets/Shopping100k/Images"

export DATASET_NAME="DeepFashion_FineGrained"
#export DATASET_NAME="Shopping100k_subset"

export MODELS_DIR="/home/xxx/data/checkpoint/adde-gan"
export CODE_DIR="/home/xxx/data/code/adde-gan"

export ENCODER_NAME="encoder_vit_lr1e-4wExpScheduler_b32"

python ${CODE_DIR}/src/init_mem.py --dataset_name ${DATASET_NAME} --file_root ${CODE_DIR}/splits/${DATASET_NAME} \
--img_root ${DATASET_PATH} --memory_dir ${MODELS_DIR}/${DATASET_NAME}/ADDE/${ENCODER_NAME} \
--load_pretrained_extractor ${MODELS_DIR}/${DATASET_NAME}/ADDE/${ENCODER_NAME}/extractor_best.pkl \
--num_threads 6 --gpu_id 0 --backbone vit --model_mode singleton
