import torch

from src.metrics.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
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

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dclasses = logits.argmax(dim=-1).to(device)
        dlabels = labels.to(device)
        print(dclasses, '\n', dlabels)
        result = (dclasses == dlabels).mean(dtype=torch.float32)
        print("OKAY")
        return result
