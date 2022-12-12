
import torch
import torch.nn as nn
from PIL import Image
import clip


class CLIPScorer:
    def __init__(self,
                 clip_checkpoints='./clip_checkpoints',
                 device='cuda'
                 ):
        self.device = device
        self.w = 2.5
        self.clip, self.clip_preprocess = clip.load("ViT-B/32", device=self.device,
                                                    download_root=clip_checkpoints, jit=False)
        self.clip = self.clip.eval()
    
    @torch.no_grad()
    def get_txt_feature(self, text):
        if isinstance(text, str):
            text = [text]
        tokens = clip.tokenize(text).to(self.device)
        text_features = self.clip.encode_text(tokens)
        text_features = nn.functional.normalize(text_features, dim=-1)
        return text_features

    @torch.no_grad()
    def get_img_feature(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        img = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        img_feature = self.clip.encode_image(img)
        img_feature = nn.functional.normalize(img_feature, dim=-1)
        return img_feature
    
    @torch.no_grad()
    def CLIPScore(self, img, text):
        assert isinstance(text, str) or (isinstance(text, list) and len(text) == 1), \
            "This implementation expects one text"
        img_feature = self.get_img_feature(img)  # (1, D)
        text_features = self.get_txt_feature(text)  # (1, D)
        sim = img_feature @ text_features.T  # (1, 1)
        score = self.w * torch.clip(sim, 0, None)  # (1, N)
        return score.item()
    
    @torch.no_grad()
    def RefCLIPScore(self, img, text, ref):
        assert isinstance(text, str) or (isinstance(text, list) and len(text) == 1), \
            "This implementation expects one text"
        img_feature = self.get_img_feature(img)  # (1, D)
        text_features = self.get_txt_feature(text)  # (1, D)
        ref_features = self.get_txt_feature(ref)  # (M, D)

        img_sim = img_feature @ text_features.T  # (1, 1)
        img_score = self.w * torch.clip(img_sim, 0, None)  # (1, N)
        
        ref_sim = ref_features @ text_features.T  # (M, 1)
        ref_sim = ref_sim.max()
        ref_score = torch.clip(ref_sim, 0, None)
        
        score = 2 * img_score * ref_score / (img_score + ref_score)  # harmonic mean
        return score.item(), img_score.item(), ref_score.item()