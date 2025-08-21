import torch

from src.metrics.base_metric import BaseMetric


class RecallMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        self.is_global = True

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        dclasses = logits.argmax(dim=-1)
        dlabels = labels.to(logits.device)
        true_positive = ((dclasses == dlabels) * dlabels).sum(dtype=torch.float32)
        false_negative = ((dclasses != dlabels) * dlabels).sum(dtype=torch.float32)
        return true_positive.item(), (true_positive + false_negative).item()
