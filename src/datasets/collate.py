import torch

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
    av_audio = []
    av_frames = []
    # vivit_list = []
    labels_list = []
    aasist_list = []
    vjepa_list = []
    # mae_list = [[], [], []]
    for elem in dataset_items:
        av_audio.append(elem['av_audio'].float())
        av_frames.append(elem['av_frames'].float())
        # vivit_list.append(elem['vivit_frames'].float())
        labels_list.append(elem['labels'])
        aasist_list.append(elem['aasist_audio'][:, :48450])
        vjepa_list.append(elem['vjepa_frames'])
        # for i in range(3):
        #     mae_list[i].append(elem[f'mae_{i}'])

    # result_batch['vivit_frames'] = torch.cat(vivit_list, dim=0)
    result_batch['av_video']     = torch.cat(av_frames, dim=0)
    result_batch['av_audio']     = torch.cat(av_audio, dim=0)
    result_batch['labels']       = torch.tensor(labels_list)
    result_batch['aasist_audio'] = torch.cat(aasist_list, dim=0)
    result_batch['vjepa_frames'] = torch.cat(vjepa_list, dim=0)
    # for i in range(3):
    #     result_batch[f'mae_{i}'] = torch.cat(mae_list[i], dim=0)
    return result_batch