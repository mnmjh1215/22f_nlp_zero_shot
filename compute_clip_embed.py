
import os
import argparse
import json
import clip
import torch
from PIL import Image
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="dataset/COCO/val2014")
    parser.add_argument("--coco_json_path", type=str, default="dataset/COCO/karpathy_dataset_coco.json")
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--split", default='val', choices=['test', 'val', 'train'])
    parser.add_argument("save_path", help="path to save clip embeddings")
    
    return parser

def get_coco_split(coco_json_path, split='val'):
    with open(coco_json_path) as f:
        coco = json.load(f)

    l = [coco['images'][i] for i in range(len(coco['images'])) if coco['images'][i]['split'] == split]
    print(f"# of {split} examples:", len(l))
    return l

def get_img_feature(clip, clip_preprocess, img_path, device):
        img = Image.open(img_path)
        clip_img = clip_preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_feature = clip.encode_image(clip_img)
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            return image_feature.detach()

def main():
    args = get_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device,
                                      download_root=args.clip_checkpoints, jit=False)
    clip_model = clip_model.eval()
    
    split = get_coco_split(args.coco_json_path, args.split)
    
    embeddings = []
    for ix, instance in tqdm(enumerate(split), total=len(split)):
        file_path = instance['filename']
        img_path = os.path.join(args.img_dir, file_path)
        embeddings.append(get_img_feature(clip_model, clip_preprocess, img_path, device).cpu())

    embeddings = torch.cat(embeddings, dim=0)
    
    print(f"Shape of Embeddings: {embeddings.shape}")
    
    torch.save(embeddings, args.save_path)
    
if __name__ == '__main__':
    main()