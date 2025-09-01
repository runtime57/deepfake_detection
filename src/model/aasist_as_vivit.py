from torch import nn
import torch
from torch.nn import Sequential
import torchvision
import fairseq
from argparse import Namespace
from transformers import VivitModel
from .AASIST import *
from .AMSDF import HGFM, GRS

class  AasistAsVivitModel(nn.Module):
    """
    Model using all (AV-HuBert, ViViT and AASIST features) and HGFM classifier
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
        """IGAM"""
        self.GAT_vivit     = GraphAttentionLayer(vivit_channels, 64, temperature=2.0)
        self.pool_vivit    = GraphPool(0.80, 64, 0.3)
        self.GAT_aasist    = GraphAttentionLayer(as_channels, 64, temperature=2.0)
        self.pool_aasist   = GraphPool(25/29, 64, 0.3)
        """HGFM"""
        self.Core_VA = HGFM()
        """GRS"""
        self.GRS_group1=GRS()
        self.GRS_group2=GRS()
        self.GRS_group3=GRS()
        self.drop = nn.Dropout(0.5, inplace=True)
        self.out_layer = nn.Linear(384, 64)
        self.out_layer2 = nn.Linear(64, 2)
        

    def forward(self, vivit_feats, av_feats, aasist_audio, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        as_feats = self.aasist(aasist_audio)
        """ IGAM """
        as_gat = self.GAT_aasist(as_feats)
        vivit_gat = self.GAT_vivit(vivit_feats)
        as_gat = self.pool_aasist(as_gat)
        vivit_gat = self.pool_vivit(vivit_gat)
        """ Heterogeneous graph fusion module"""
        VA_HG, VA_SN,attmap_VA = self.Core_VA(vivit_gat, as_gat)
        """Group-based Readout Scheme"""
        GAT_Group=[as_gat,vivit_gat]
        HGAT_Group=[VA_HG]
        SN_Group=[VA_SN]
        out1=self.GRS_group1(GAT_Group)
        out2=self.GRS_group2(HGAT_Group)
        out3=self.GRS_group3(SN_Group)
        """output"""
        last_hidden = torch.cat([out1,out2,out3], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        output = self.out_layer2(output)

        return {"logits": output}

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
