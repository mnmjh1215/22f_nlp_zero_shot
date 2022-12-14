CODE_PATH=$(git rev-parse --show-toplevel)
SCRIPT_PATH=${CODE_PATH}/scripts
HF_CACHE=$HOME/.cache/huggingface
OUT_PATH=${CODE_PATH}/outputs
DATASET_PATH=${CODE_PATH}/datasets
SAVE_PATH=${CODE_PATH}/save
sudo docker run -i -t --user 10059 \
    -e WANDB_API_KEY=56f1c00de0e2eb979dce7c46de19471dbec92f94 \
    -e HYDRA_FULL_ERROR=1 \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${HF_CACHE}:/root/.cache/huggingface:rw \
    -v ${CODE_PATH}:/code:ro \
    -v ${SCRIPT_PATH}:/scripts:rw \
    -v ${OUT_PATH}:/outputs:rw \
    -v ${DATASET_PATH}:/datasets:rw \
    -v ${SAVE_PATH}:/save:rw \
  zshvid:v0.1 $@