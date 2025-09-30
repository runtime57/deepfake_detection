from torch import nn
import torch
from torch.nn import Sequential
import torchvision
import fairseq
from argparse import Namespace
from transformers import VivitModel
from .AASIST import aasist_encoder
from .MSTCN import MSTCN
import torch.nn.functional as F


class MstcnFullModel(nn.Module):
    """
    Model using all (AV-HuBert, ViViT and AASIST features) and MSTCN classifier
    """

    def __init__(self, av_channels, vivit_channels, as_channels, av_time, vivit_time, as_time, hidden_time):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.aasist = aasist_encoder()
        self.pool_as = nn.Linear(in_features=as_time, out_features=hidden_time)
        self.pool_av = nn.Linear(in_features=av_time, out_features=hidden_time)
        self.pool_vivit = nn.Linear(in_features=vivit_time, out_features=hidden_time)
        self.ms_tcn = MSTCN(input_size=av_channels+vivit_channels+as_channels)

    def forward(self, vivit_feats, av_feats, aasist_audio, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        as_feats = self.aasist(aasist_audio)

        as_feats = self.pool_as(as_feats.transpose(1, 2))
        av_feats = self.pool_av(av_feats.transpose(1, 2))
        vivit_feats = self.pool_vivit(vivit_feats.transpose(1, 2))

        feats = torch.cat([as_feats, av_feats, vivit_feats], dim=1)

        return {"logits": self.ms_tcn(feats)}

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