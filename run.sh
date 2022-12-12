CUDA_VISIBLE_DEVICES=2 python eval.py --coco_json_path /cmsdata/ssd0/cmslab/coco2014/dataset_coco.json --img_dir /cmsdata/ssd0/cmslab/coco2014/val2014 --delta_path ./delta_config/lora_config.json --debug --result_save_path results/result_lora_debug.json

CUDA_VISIBLE_DEVICES=2 python eval.py --coco_json_path /cmsdata/ssd0/cmslab/coco2014/dataset_coco.json --img_dir /cmsdata/ssd0/cmslab/coco2014/val2014 --debug --result_save_path results/result_vanila_debug.json

CUDA_VISIBLE_DEVICES=2 python eval.py --coco_json_path /cmsdata/ssd0/cmslab/coco2014/dataset_coco.json --img_dir /cmsdata/ssd0/cmslab/coco2014/val2014 --result_save_path results/result_lora_full.json

CUDA_VISIBLE_DEVICES=2 python eval.py --coco_json_path /cmsdata/ssd0/cmslab/coco2014/dataset_coco.json --img_dir /cmsdata/ssd0/cmslab/coco2014/val2014 --debug --result_save_path results/result_vanila_full.json