from torch import nn
import torch
from torch.nn import Sequential
import torchvision
import fairseq
from argparse import Namespace
from transformers import VivitModel
from .AASIST import aasist_encoder


class MlpAsVivitModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, av_channels, vivit_channels, as_channels, hidden_channels):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.aasist = aasist_encoder()

        self.mlp = Sequential(
            nn.Linear(in_features=vivit_channels+as_channels, out_features=hidden_channels),
            nn.ReLU(),
            nn.Linear(in_features=hidden_channels, out_features=hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_channels // 2, out_features=2),
        )

    def forward(self, vivit_feats, av_feats, aasist_audio, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        
        as_feats = self.aasist(aasist_audio)

        as_feats = as_feats.mean(dim=1)
        vivit_feats = vivit_feats.mean(dim=1)

        feats = torch.cat([vivit_feats, as_feats], dim=1)

        return {"logits": self.mlp(feats)}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info