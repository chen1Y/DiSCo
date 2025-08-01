export DATASET_PATH="/home/xxx/DATA/datasets/Shopping100k/Images"
# export DATASET_PATH="/home/xxx/DATA/datasets/DeepFashion"

export MODELS_DIR="/home/xxx/DATA/checkpoint/adde-gan/"
export CODE_DIR="/home/xxx/DATA/code/adde-gan"

export DATASET_NAME="Shopping100k_subset"
# export DATASET_NAME="DeepFashion_FineGrained"

export GAN_NAME="GAN_cca_aac_percept_rec_mse_cls_0.2"

cd ${CODE_DIR}/src
python benchmark/fid.py --file_root ${CODE_DIR}/splits/${DATASET_NAME} --img_root ${DATASET_PATH} --name ${GAN_NAME} --eval_root ${MODELS_DIR}/${DATASET_NAME}
