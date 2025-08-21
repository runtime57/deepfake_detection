import torch
import numpy as np
from torch import nn
from sklearn.utils.class_weight import compute_class_weight


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted CrossEntropyLoss
    """
    def __init__(self):
        super().__init__()
        real = 4000
        fake = 16340
        weights = torch.tensor([(real + fake) / (2 * fake), (real + fake) / (2 * real)])
        self.loss = nn.CrossEntropyLoss(weight=weights)
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        return {"loss": self.loss(logits, labels)}
