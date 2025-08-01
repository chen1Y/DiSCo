# export DATASET_PATH="/home/xxx/data/datasets/DeepFashion"
export DATASET_PATH="/home/xxx/data/datasets/Shopping100k/Images"

# export DATASET_NAME="DeepFashion_FineGrained"
export DATASET_NAME="Shopping100k_subset"

export MODELS_DIR="/home/xxx/data/checkpoint/adde-gan"
export CODE_DIR="/home/xxx/data/code/adde-gan"

export ENCODER_NAME="encoder_cca_lr1e-4wExpScheduler_b80"

python ${CODE_DIR}/src/train_attr_pred_cca.py \
--dataset_name ${DATASET_NAME} --file_root ${CODE_DIR}/splits/${DATASET_NAME} \
--img_root ${DATASET_PATH} --ckpt_dir ${MODELS_DIR}/${DATASET_NAME}/ADDE/${ENCODER_NAME} \
--visual_dir /home/xxx/data/tf-logs/${DATASET_NAME} --exp_name ${ENCODER_NAME} \
--backbone cca --model_mode singleton --batch_size 80 --num_threads 6 \
--lr 0.0001 --num_epochs 50 --gpus 0 1 2