import av
import numpy as np
from transformers import VivitImageProcessor


# ViViT Preprocess

def vivit_preprocess(file_path):
    """
    Args:
        file_path (str): filename of the mp4 file.
    Returns:
        vivit_frames (torch.Tensor): preprocessed frames.
    """

    # we need to sample 32 frames with stide 2 (check ViViT: A Video Vision Transformer)
    # It's ok with our fps 25, because model was trained on Kinetiks Dataset (also 25 fps)

    container = av.open(file_path)

    indices = sample_frame_indices(clip_len=32, frame_sample_rate=2, seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container=container, indices=indices)

    image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

    frames = image_processor(list(video), return_tensors="pt")
    return frames


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`list[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`list[int]`): List of sampled frame indices

    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# AV Preprocess

def av_preprocess(path):
    """
    Args:
        fname (str): filename of the mp4 file.
    Returns:
        av_audio (Tensor): preprocessed audio, av_frames (Tensor): preprocessed frames.
    """

    # WTF and HOW TO ?

    return
