#!/bin/bash
N_TEST=100
DELTA_TYPE=None
LM_MODEL=gpt-neo
NOW=$(date +"%b-%d_%H-%M-%S")

COMMAND="python eval.py --token_wise --randomized_prompt --img_dir ./datasets/val2014 --coco_json_path ./datasets/dataset_coco.json --strip_prompt False --num_test ${N_TEST} --lm_model ${LM_MODEL} --result_save_path ${DELTA_TYPE}_${LM_MODEL}_${N_TEST}_${NOW}.json"

echo $COMMAND
exec $COMMAND