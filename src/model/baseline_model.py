from torch import nn
import torch
from torch.nn import Sequential
import torchvision
import fairseq
from argparse import Namespace
from transformers import VivitModel


class BaselineModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, in_channels, hidden_channels, dropout):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        # init av-hubert model
        # fairseq.utils.import_user_module(Namespace(user_dir='/home/runtime57/hse/coursework_2/deepfake_detection/src/model/av_hubert/avhubert'))
        ckpt_path = '/home/runtime57/hse/coursework_2/deepfake_detection/src/model/av_hubert/ckpt/base_vox_433h.pt'
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.avhubert = models[0]
        if hasattr(models[0], 'decoder'):
            print(f"Checkpoint: fine-tuned")
            self.avhubert = models[0].encoder.w2v_model
        else:
            print(f"Checkpoint: pre-trained w/o fine-tuning")
        self.avhubert.cuda()
        self.avhubert.eval()
        
        # init ViViT
        self.vivit = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400").eval()

        self.mlp = Sequential(
            nn.Linear(in_features=in_channels*2, out_features=hidden_channels),
            nn.ReLU(),
            nn.Linear(in_features=hidden_channels, out_features=hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_channels // 2, out_features=2),
        )

    def forward(self, vivit_frames, av_video, av_audio, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """

        av_feats, vivit_feats = self._extract_feats(vivit_frames, av_video, av_audio)
        av_pooled_feats = av_feats.mean(dim=1)
        feats = torch.cat([av_pooled_feats, vivit_feats], dim=1)

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

    def _extract_feats(self, vivit_frames, av_video, av_audio):
         with torch.no_grad():
            # print(av_video.device, av_audio.device, av_mask.device)
            av_feats, _ = self.avhubert.extract_finetune(source={'video': av_video, 'audio': av_audio}, padding_mask=None, output_layer=None)
            vivit_feats = self.vivit(pixel_values=vivit_frames).last_hidden_state[:, 0, :]
            return av_feats, vivit_feats
