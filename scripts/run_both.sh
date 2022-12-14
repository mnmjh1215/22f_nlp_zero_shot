#!/bin/bash
N_TEST=100
DELTA_TYPE=lora
USE_PREFIX=False
LR=0.001
SENTENCE_ITERATIONS=128
NOW=$(date +"%b-%d_%H-%M-%S")

COMMAND="python eval_both.py --token_wise --randomized_prompt --img_dir ./datasets/val2014 --coco_json_path ./datasets/dataset_coco.json --strip_prompt False --learning_rate ${LR} --num_test ${N_TEST} --sentence_iterations ${SENTENCE_ITERATIONS} --delta_path ./delta_config/${DELTA_TYPE}/config.json --result_save_path ${DELTA_TYPE}_${N_TEST}_prefix_${USE_PREFIX}_lr_${LR}_iter_${SENTENCE_ITERATIONS}_${NOW}.json"

echo $COMMAND
exec $COMMAND