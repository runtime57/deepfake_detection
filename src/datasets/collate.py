import torch

max_sample_size = 196

def collate_fn(dataset_items):
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

    result_batch["vivit_frames"] = torch.cat(
        [elem["vivit_frames"] for elem in dataset_items], dim=0
    )
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])

    result_batch["av_input"] = collater(dataset_items)
    print(result_batch['av_input']['source']['video'].shape)
    
    return result_batch

def collater(samples):
    audio_source, video_source = [s["av_audio"] for s in samples], [s["av_frames"] for s in samples]
    print(audio_source[0].shape, video_source[0].shape)
    if audio_source[0] is None:
        audio_source = None
    if video_source[0] is None:
        video_source = None
    if audio_source is not None:
        audio_sizes = [len(s) for s in audio_source]
    else:
        audio_sizes = [len(s) for s in video_source]
    
    audio_size = min(min(audio_sizes), max_sample_size)
    if audio_source is not None:
        collated_audios, padding_mask, audio_starts = collater_audio(audio_source, audio_size)
    else:
        collated_audios, audio_starts = None, None
    if video_source is not None:
        print('vid not none')
        collated_videos, padding_mask, audio_starts = collater_audio(video_source, audio_size, audio_starts)
    else:
        collated_videos = None
    source = {"audio": collated_audios, "video": collated_videos}
    net_input = {"source": source, "padding_mask": padding_mask}
    return net_input

def collater_audio(audios, audio_size, audio_starts=None):
    audio_feat_shape = list(audios[0].shape[1:])
    collated_audios = audios[0].new_zeros([len(audios), audio_size]+audio_feat_shape)
    padding_mask = (
        torch.BoolTensor(len(audios), audio_size).fill_(False) # 
    )
    start_known = audio_starts is not None
    audio_starts = [0 for _ in audios] if not start_known else audio_starts
    for i, audio in enumerate(audios):
        diff = len(audio) - audio_size
        if diff == 0:
            collated_audios[i] = audio
        elif diff < 0: # never should go here, because we take minimal length
            print("FUUUUUCK")
            exit(0)
            collated_audios[i] = torch.cat(
                [audio, audio.new_full([-diff]+audio_feat_shape, 0.0)]
            )
            padding_mask[i, diff:] = True
        else:
            collated_audios[i], audio_starts[i] = crop_to_max_size(
                audio, audio_size, audio_starts[i] if start_known else None
            )
    if len(audios[0].shape) == 2:
        print('here 1')
        collated_audios = collated_audios.transpose(1, 2) # [B, T, F] -> [B, F, T]
    else:
        print('here 2')
        collated_audios = collated_audios.permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]
    return collated_audios, padding_mask, audio_starts

def crop_to_max_size(wav, target_size, start=None):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0
        # longer utterances
        if start is None:
            start, end = 0, target_size
        else:
            end = start + target_size
        return wav[start:end], start