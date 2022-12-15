# CLIPScore-Diff metric, that measures 'specificity' of the generated caption
# by comparing clipscore w.r.t. the image and clipscores w.r.t. other images in the test set

import os
import argparse
import json
import clip
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np

@torch.no_grad()
def get_img_feature(clip_model, clip_preprocess, img_path, device):
    img = Image.open(img_path)
    clip_img = clip_preprocess(img).unsqueeze(0).to(device)
    image_feature = clip_model.encode_image(clip_img)
    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
    return image_feature.detach()

@torch.no_grad()
def get_txt_feature(clip_model, text, device):
    if isinstance(text, str): text = [text]
    tokens = clip.tokenize(text).to(device)
    text_feature = clip_model.encode_text(tokens)
    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
    return text_feature.detach()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="dataset/COCO/val2014")
    parser.add_argument("--result_path", type=str, default='zerocap-1000.json')
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--ratios", default=[0.5, 0.75, 0.9, 0.99], type=float, nargs='+')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    with open(args.result_path) as f: res = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device,
                                      download_root=args.clip_checkpoints, jit=False)
    clip_model = clip_model.eval()

    # first, get image embeddings of test samples and text embeddings of generated texts
    img_embeddings = []
    txt_embeddings = []
    for item in tqdm(res['per_instance']):
        img_path = os.path.join(args.img_dir, item['filename'])
        img_embedding = get_img_feature(clip_model, clip_preprocess, img_path, device)
        img_embeddings.append(img_embedding)
        txt_embedding = get_txt_feature(clip_model, item['best_clip']['caption'], device)
        txt_embeddings.append(txt_embedding)
    img_embeddings = torch.cat(img_embeddings, dim=0)  # (N, D)
    txt_embeddings = torch.cat(txt_embeddings, dim=0)  # (N, D)

    txt2img_sims = txt_embeddings @ img_embeddings.T
    img2img_sims = img_embeddings @ img_embeddings.T

    # metric 1) diff between clip score and that of unrelated images
    ratios = args.ratios
    if 1.0 in ratios: ratios.remove(1.0)  # for 1.0, we need to remove real clipscore
    diffs = {k:[] for k in ratios}
    diffs[1.0] = []
    n_eval = txt2img_sims.size(0)
    for i in tqdm(range(n_eval)):
        clipscore = 2.5 * torch.clip(txt2img_sims[i, i], 0, None)

        for r in ratios:
            unrelated_img_idxs = torch.argsort(img2img_sims[i])[:int(n_eval * r)]
            unrelated_clipscore_max = 2.5 * torch.clip(txt2img_sims[i][unrelated_img_idxs], 0, None).max()
            diffs[r].append(clipscore - unrelated_clipscore_max)

        # for ratio 1.0, remove 
        rest_sims = txt2img_sims[i].clone()
        rest_sims[i] = -100
        max_of_rest_idx = torch.argmax(rest_sims)
        best_of_rest = 2.5 * torch.clip(txt2img_sims[i][max_of_rest_idx], 0, None)
        diffs[1.0].append(clipscore - best_of_rest)

    for k, v in diffs.items():
        print(f"CLIPScore-Diff-{k}:", torch.mean(torch.stack(v)).item())

if __name__ == '__main__':
    main()
