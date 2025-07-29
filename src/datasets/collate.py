import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    result_batch["av_audio"] = torch.cat(
        [elem["av_audio"].unsqueeze(0) for elem in dataset_items]
    )
    result_batch["av_frames"] = torch.cat(
        [elem["av_frames"].unsqueeze(0) for elem in dataset_items]
    )
    result_batch["vivit_frames"] = torch.cat(
        [elem["vivit_frames"].unsqueeze(0) for elem in dataset_items]
    )
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])

    return result_batch
