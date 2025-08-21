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


def av_video_pad(x, max_len = 75):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len].clone()
    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = x.repeat(num_repeats, 1, 1, 1)[:max_len]
    return padded_x.clone()

def av_audio_pad(x, max_len = 75):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len].clone()
    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = x.repeat(num_repeats, 1)[:max_len]
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
    labels_list = []
    aasist_list = []

    for elem in dataset_items:
        vivit_list.append(elem['vivit_frames'].float())
        labels_list.append(elem['labels'])
        aasist_list.append(aasist_pad(elem['aasist_audio']).float().unsqueeze(0))

    av_video, av_audio = new_collater(dataset_items)

    result_batch['vivit_frames'] = torch.cat(vivit_list, dim=0)
    result_batch['av_video']     = av_video
    result_batch['av_audio']     = av_audio
    result_batch['labels']       = torch.tensor(labels_list)
    result_batch['aasist_audio'] = torch.cat(aasist_list, dim=0)
    return result_batch

def new_collater(samples):
    audio_source, video_source = [av_audio_pad(s['av_audio']).float().unsqueeze(0) for s in samples], [av_video_pad(s['av_frames']).float().unsqueeze(0) for s in samples]
    collated_audios = torch.cat(audio_source, dim=0).transpose(1, 2)
    collated_videos = torch.cat(video_source, dim=0).permute((0, 4, 1, 2, 3)).contiguous()
    return collated_videos, collated_audios
