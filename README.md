# Zero-Shot Video Captioning with Negative Image Retrieval

This repository contains code for our team project for SNU Natural Language Processing course (2022 Fall).

Our approach and code is based on [Zero-Shot Video Captioning with Evolving Pseudo-Tokens](https://arxiv.org/abs/2207.11100) ([code](https://github.com/YoadTew/zero-shot-video-to-text)).

## Data Preparation

To run our code, please download [COCO 2014 Val Images](http://images.cocodataset.org/zips/val2014.zip) and [Karpathy splits for Image Captioning](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits).

## Usage

### Baseline

To evaluate baseline on 1,000 test images, run following command.

```
python eval.py --token_wise --randomized_prompt --img_dir <path to COCO val2014 dir> --coco_json_path <path to karpathy split json> --num_test 1000 --result_save_path <path to save result json file>
```

### Our Approach

Our approach use images from validation split of Karpathy split as candidates for retrieving negative images.
To do so, we need to first compute embeddings for these images.

```
python compute_clip_embed.py <path to save embeddings.pt> --img_dir <path to COCO val2014 dir> --coco_json_path <path to karpathy split json> --split val 
```

Then, to evaluate our approach on 1,000 test images, run following command.

```
python dynamic_scale_neg_eval.py --token_wise --randomized_prompt --img_dir <path to COCO val2014 dir> --coco_json_path <path to karpathy split json> --coco_embedding_path <path to computed embeddings.pt> --num_test 1000 --num_neg 10 --neg_scale 1.0 --result_save_path <path to save result json file>
```

### Further Evaluation

Our newly proposed CLIPScore-Diff metric can be computed using following command.

```
python clipscore_diff_metric.py --img_dir <path to COCO val2014 dir> --result_path <path to saved result json file>
```