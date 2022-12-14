#!/bin/bash
N_TEST=100
DELTA_TYPE=lora
USE_PREFIX=False
LR=0.005
SENTENCE_ITERATIONS=64
LM_MODEL=gpt-neo
NOW=$(date +"%b-%d_%H-%M-%S")

COMMAND="python eval.py --token_wise --randomized_prompt --img_dir ./datasets/val2014 --coco_json_path ./datasets/dataset_coco.json --strip_prompt False --learning_rate ${LR} --num_test ${N_TEST} --sentence_iterations ${SENTENCE_ITERATIONS} --delta_path ./delta_config/${DELTA_TYPE}/config_${LM_MODEL}.json --lm_model ${LM_MODEL} --result_save_path ${DELTA_TYPE}_${LM_MODEL}_${N_TEST}_prefix_${USE_PREFIX}_lr_${LR}_iter_${SENTENCE_ITERATIONS}_${NOW}.json"
echo $COMMAND
exec $COMMAND