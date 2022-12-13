
# evaluate on mscoco data

import argparse
import logging
import clip
from model.NegAugCapGenerator import NegAugCLIPTextGenerator
import torch
import os
# from data_loader import VideosDataset, ImagesDataset, ImagesPairsDataset
from datetime import datetime
import shutil
import json
import sys
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from clipscore import CLIPScorer
from tqdm import tqdm
from collections import Counter
import spacy

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--randomized_prompt", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo")
    parser.add_argument("--db_filter_path", type=str, default=None, help="file to filter db items, e.g karpathy split")
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--target_seq_length", type=int, default=20)
    parser.add_argument("--cond_text", type=str, default="Image of a")
    parser.add_argument("--token_wise", action="store_true", help="Should we step the optimization at each token gen")
    parser.add_argument("--num_dummy_tokens", type=int, default=5)
    parser.add_argument("--sentence_iterations", type=int, default=16)
    parser.add_argument("--sampling_top_k", type=int, default=3)
    parser.add_argument("--db_start_idx", type=int, default=0)
    parser.add_argument("--db_num_images", type=int, default=0)
    parser.add_argument("--clip_loss_temperature", type=float, default=1.0)
    parser.add_argument("--clip_scale", type=float, default=1)
    parser.add_argument("--ce_scale", type=float, default=0.8)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--scheduler_type", type=NegAugCLIPTextGenerator.SchedType, default='cosine')
    parser.add_argument("--weight_decay_scale", type=float, default=0.3)
    parser.add_argument("--repetition_penalty", type=float, default=2.0, help='How much much to deter deter repeats')
    parser.add_argument("--entity_penalty", type=float, default=2, help='How much to deter CapsLock in middle of sent')
    parser.add_argument("--ending_bonus", type=float, default=2, help='How much to help the sentence to end')
    parser.add_argument("--end_token", type=str, default=".", help="Token to end text")
    parser.add_argument("--pairs_path", type=str, default="")

    parser.add_argument("--clip_reward_alpha", default=50, type=float)
    parser.add_argument("--clip_reward_beta", default=-10, type=float)
    parser.add_argument("--log_prob_reward_scale", default=1.0, type=float)

    parser.add_argument("--img_dir", type=str, default="dataset/COCO/val2014")
    parser.add_argument("--coco_json_path", type=str, default="dataset/COCO/karpathy_dataset_coco.json")
    parser.add_argument("--num_test", default=1000, type=int,
                        help="number of test samples to use. if -1, then use entire test set")
    parser.add_argument("--result_save_path", type=str, default="results.json")
    parser.add_argument("--strip_prompt", default=False, type=str2bool)
    parser.add_argument("--clip_embedding_path", default='coco_val_embeddings.pt')
    parser.add_argument("--num_neg", default=5, type=int)
    parser.add_argument("--neg_scale", default=0.5, type=float)
    parser.add_argument("--verbose", action='store_true')
    return parser

def str2bool(s):
    if s.lower().startswith('t') or s == '1':
        return True
    else:
        return False


def get_coco_test(coco_json_path):
    with open(coco_json_path) as f:
        coco = json.load(f)

    test = [coco['images'][i] for i in range(len(coco['images'])) if coco['images'][i]['split'] == 'test']
    print("# of test examples:", len(test))
    return test

def get_random_n(l, n, seed=0):
    np.random.seed(seed)
    np.random.shuffle(l)
    return l[:n]

def main():
    parser = get_parser()
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_generator = NegAugCLIPTextGenerator(**vars(args))
    scorer = CLIPScorer(args.clip_checkpoints, device)

    coco_test = get_coco_test(args.coco_json_path)
    if args.debug:
        print("[Debug Mode] only evaluate on 50 test images")
        coco_test = coco_test[:50]
        print([inst['filename'] for inst in coco_test])
    else:
        if 0 < args.num_test < len(coco_test):
            coco_test = get_random_n(coco_test, args.num_test, seed=args.seed)

    clip_embeddings = torch.load(args.clip_embedding_path).to(device)

    results = {}
    results['per_instance'] = []
    best_clip_unique_words = set()
    best_mixed_unique_words = set()
    best_clip_pos_counter = Counter()
    best_mixed_pos_counter = Counter()
    for ix, instance in tqdm(enumerate(coco_test), total=len(coco_test)):
        file_path = instance['filename']
        img_path = os.path.join(args.img_dir, file_path)
        refs = [instance['sentences'][i]['raw'] for i in range(len(instance['sentences']))]
        inst_result = {'filename': instance['filename'], 'refs': refs,
                       'best_clip': {}, 'best_mixed': {}}

        img_feature = text_generator.get_img_feature([img_path], None)
        clip_sorted_captions, mixed_sorted_captions, decoded_options, beam_caps = text_generator.generate(img_feature, clip_embeddings,
                                                                                                          verbose=args.verbose)

        best_clip_cap = clip_sorted_captions[0]
        best_mixed_cap = mixed_sorted_captions[0]

        if args.randomized_prompt and args.strip_prompt:
            # strip initial prompt
            # for simplicity, just remove first two words
            best_clip_cap = ' '.join(best_clip_cap.split()[2:])
            best_mixed_cap = ' '.join(best_mixed_cap.split()[2:])

        refclipscore, clipscore, refscore = scorer.RefCLIPScore(img_path, best_clip_cap, refs)
        inst_result['best_clip']['caption'] = best_clip_cap
        inst_result['best_clip']['refclipscore'] = refclipscore
        inst_result['best_clip']['clipscore'] = clipscore

        refclipscore, clipscore, refscore = scorer.RefCLIPScore(img_path, best_mixed_cap, refs)
        inst_result['best_mixed']['caption'] = best_mixed_cap
        inst_result['best_mixed']['refclipscore'] = refclipscore
        inst_result['best_mixed']['clipscore'] = clipscore

        results['per_instance'].append(inst_result)

        if args.debug:
            print(f"[{ix}] Best CLIP:", inst_result['best_clip'])
            print(f"[{ix}] Best Mixed:", inst_result['best_mixed'])

    # compute aggregate score
    mean_best_clip = {}
    mean_best_mixed = {}
    mean_best_clip['refclipscore'] = np.mean([per_inst['best_clip']['refclipscore'] for per_inst in results['per_instance']])
    mean_best_clip['clipscore'] = np.mean([per_inst['best_clip']['clipscore'] for per_inst in results['per_instance']])
    mean_best_mixed['refclipscore'] = np.mean([per_inst['best_mixed']['refclipscore'] for per_inst in results['per_instance']])
    mean_best_mixed['clipscore'] = np.mean([per_inst['best_mixed']['clipscore'] for per_inst in results['per_instance']])

    results['clipscore'] = {'best_clip': mean_best_clip, 'best_mixed': mean_best_mixed}
    print("[CLIPScore] Best CLIP:", mean_best_clip)
    print("[CLIPScore] Best Mixed:", mean_best_mixed)

    with open(args.result_save_path, 'w') as fw:
        json.dump(results, fw)

if __name__ == '__main__':
    main()
