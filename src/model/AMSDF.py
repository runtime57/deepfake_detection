import random
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
import os
from .AASIST import *

class ASR_model(nn.Module):
    def __init__(self):
        super(ASR_model, self).__init__()
        cp_path = os.path.join('./pretrained_models/xlsr2_300m.pt')   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0].cuda()
        self.linear = nn.Linear(1024, 128)
        self.bn = nn.BatchNorm1d(num_features=50)
        self.selu = nn.SELU(inplace=True)

    def forward(self, x):
        emb = self.model(x, mask=False, features_only=True)['x']
        emb = self.linear(emb) 
        emb = F.max_pool2d(emb, (4,2)) 
        emb = self.bn(emb)
        emb = self.selu(emb)
        return emb

    
class base_encoder(nn.Module):
    def __init__(self):
        super(base_encoder, self).__init__()
        filts= [[1, 32], [32, 32], [32, 64], [64, 64]]
        self.conv_time=CONV(out_channels=70,
                              kernel_size=128,
                              in_channels=1)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[0], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[1])),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[3]))) 

    def forward(self, x, Freq_aug):
        x = x.unsqueeze(1)
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        emb = self.encoder(x) 
        out, _ = torch.max(torch.abs(emb), dim=2) 
        out = out.transpose(1, 2) 
        return out


class HGFM(nn.Module):
    def __init__(self):
        super(HGFM, self).__init__()
        self.HtrgGAT_layer1 = HtrgGraphAttentionLayer(64, 64, temperature=100)
        self.HtrgGAT_layer2 = HtrgGraphAttentionLayer(64, 64, temperature=100)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.stack_node = nn.Parameter(torch.randn(1, 1, 64))

    def forward(self, x1, x2):
        stack_node = self.stack_node.expand(x1.size(0), -1, -1)
        x1, x2, stack_node, _ = self.HtrgGAT_layer1(x1, x2, master=stack_node)
        x1_aug, x2_aug, stack_node2,attmap = self.HtrgGAT_layer2(x1, x2, master=stack_node)
        x1 = x1 + x1_aug 
        x2 = x2 + x2_aug
        stack_node = stack_node + stack_node2 
        x1 = self.drop_way(x1)
        x2 = self.drop_way(x2)
        stack_node = self.drop_way(stack_node)
        return x1+x2, stack_node, attmap


class GRS(nn.Module):
    def __init__(self):
        super(GRS, self).__init__()
        self.pool1 = GraphPool(0.5, 64, 0.3)
        self.pool2 = GraphPool(0.5, 64, 0.3)
    def forward(self, x_list):
        pool_list=[]
        for i in x_list:
            pool_list.append(self.pool2(self.pool1(i)))
        pool_cat=torch.cat(pool_list, dim=1)
        pool_max, _=torch.max(torch.abs(pool_cat),dim=1)
        pool_avg=torch.mean(pool_cat,dim=1)
        return torch.cat([pool_max,pool_avg], dim=1)