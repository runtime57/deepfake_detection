from torch import nn
import torch
from torch.nn import Sequential
import torchvision
import fairseq
from argparse import Namespace
import torch.nn.functional as F
from transformers import VivitModel
from .AASIST import *
from .AMSDF import HGFM, GRS
from src.utils.io_utils import ROOT_PATH

class  AasistFullModel(nn.Module):
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
        ckpt_path         = str(ROOT_PATH / 'src/model/av_hubert/ckpt/base_vox_433h.pt')
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.avhubert     = models[0].encoder.w2v_model if hasattr(models[0], 'decoder') else models[0]
        self.vivit        = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.aasist       = aasist_encoder()
        """IGAM"""
        self.GAT_avhubert  = GraphAttentionLayer(av_channels, 64, temperature=2.0)
        self.pool_avhubert = GraphPool(25/75, 64, 0.3)
        self.GAT_vivit     = GraphAttentionLayer(vivit_channels, 64, temperature=2.0)
        self.pool_vivit    = GraphPool(0.80, 64, 0.3)
        self.GAT_aasist    = GraphAttentionLayer(as_channels, 64, temperature=2.0)
        self.pool_aasist   = GraphPool(25/29, 64, 0.3)
        """HGFM"""
        self.Core_HV  = HGFM()
        self.Core_HA  = HGFM()
        self.Core_VA  = HGFM()
        self.Core_HVA = HGFM()
        """GRS"""
        self.GRS_group1 = GRS()
        self.GRS_group2 = GRS()
        self.GRS_group3 = GRS()
        self.drop       = nn.Dropout(0.5, inplace=True)
        self.out_layer  = nn.Linear(384, 64)
        self.out_layer2 = nn.Linear(64, 2)


    def forward(self, vivit_frames, av_video, av_audio, aasist_audio, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        def interpolate(x, factor=2):
            x = x.permute(0, 2, 1)
            x = F.interpolate(x, scale_factor=factor, mode='linear')
            x = x.permute(0, 2, 1)
            return x

        av_feats, _ = self.avhubert.extract_finetune(source={'video': av_video,
                                                        'audio': av_audio},
                                                    padding_mask=None,
                                                    output_layer=None)
        vivit_feats = self.vivit(pixel_values=vivit_frames).last_hidden_state[:, 1:, :]
        vivit_feats = vivit_feats.view(av_feats.shape[0], 16, 14, 14, 768).mean(dim=(2, 3))
        vivit_feats = vivit_feats.reshape(av_feats.shape[0], 16, 768)
        vivit_feats = interpolate(vivit_feats)
        as_feats = self.aasist(aasist_audio)

        # print("============== NEW ==================")
        # print(aasist_audio)

        """ IGAM """
        as_gat = self.GAT_aasist(as_feats)
        av_gat = self.GAT_avhubert(av_feats)
        vivit_gat = self.GAT_vivit(vivit_feats) 
        as_gat = self.pool_aasist(as_gat) 
        av_gat = self.pool_avhubert(av_gat) 
        vivit_gat = self.pool_vivit(vivit_gat)

        """ Heterogeneous graph fusion module"""
        HV_HG, HV_SN,attmap_HV = self.Core_HV(av_gat, vivit_gat)
        HA_HG, HA_SN,attmap_HA = self.Core_HA(av_gat, as_gat)
        VA_HG, VA_SN,attmap_VA = self.Core_VA(vivit_gat, as_gat)
        HVA_HG, HVA_SN,attmap_HVA = self.Core_HVA(HV_HG, VA_HG)

        """Group-based Readout Scheme"""
        GAT_Group=[av_gat,as_gat,vivit_gat]
        HGAT_Group=[HV_HG,HA_HG,VA_HG,HVA_HG]
        SN_Group=[HV_SN,HA_SN,VA_SN,HVA_SN]
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
