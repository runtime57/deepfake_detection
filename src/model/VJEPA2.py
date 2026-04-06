from torch import nn
import torch
from transformers import AutoVideoProcessor, AutoModel

class VJEPA2_enc(nn.Module):
    def __init__(self):
        HF_REPO = "facebook/vjepa2-vitl-fpc64-256"
        self.model = AutoModel.from_pretrained(HF_REPO)

    def forward(self, frames):
        x = self.model(frames, skip_predictor=True).last_hidden_state
        B, N, D = x.shape
        cfg = model.config
        T = frames.shape[0]
        Tt = T // cfg.tubelet_size
        S = (cfg.crop_size // cfg.patch_size) ** 2
        x = x.view(B, Tt, S, D).mean(dim=2)
        return x