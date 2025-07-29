import torch
import numpy as np
from torch import nn
from sklearn.utils.class_weight import compute_class_weight


class CrossEntropyLoss(nn.Module):
    """
    Weighted CrossEntropyLoss
    """
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
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels.numpy()), y=labels.numpy())
        weighted_loss = nn.CrossEntropyLoss(weight=weights)
        return {"loss": weighted_loss(logits, labels)}
