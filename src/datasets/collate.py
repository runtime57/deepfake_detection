import torch

max_sample_size=75

def aasist_pad(x, max_len: int = 64600):
    x_len = x.shape[0]
    if x_len == max_len:
        return x
    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = x.repeat(num_repeats)[:max_len]
    return padded_x.clone()


def collate_fn(dataset_items):
    '''
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    '''
    result_batch = {}
    vivit_list = []
    av_list = []
    labels_list = []
    aasist_list = []

    for elem in dataset_items:
        vivit_list.append(elem['vivit_feats'].float())
        av_list.append(elem['av_feats'].float())
        labels_list.append(elem['labels'])
        aasist_list.append(aasist_pad(elem['aasist_audio']).float().unsqueeze(0))

    result_batch['vivit_feats'] = torch.cat(vivit_list, dim=0)
    result_batch['av_feats']     = torch.cat(av_list, dim=0)
    result_batch['aasist_audio'] = torch.cat(aasist_list, dim=0)
    result_batch['labels']       = torch.tensor(labels_list)
    return result_batch
