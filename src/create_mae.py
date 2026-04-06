from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
import numpy as np
import torch
from decord import VideoReader, cpu
import numpy as np
import json
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
import safetensors

def read_json(fname):
    """
    Read the given json file.

    Args:
        fname (str): filename of the json file.
    Returns:
        json (list[OrderedDict] | OrderedDict): loaded json.
    """
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)

def sample_frames(path: str, num_frames: int = 48):
    vr = VideoReader(path, ctx=cpu(0))
    n = len(vr)
    idx = np.linspace(0, 100, num_frames, dtype=np.int64) % n
    frames = vr.get_batch(idx).asnumpy()
    return [np.transpose(f, (2, 0, 1)) for f in frames]

processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Large")

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent

def create_mae(video_path):
    video = sample_frames(str(ROOT_PATH / video_path))
    element = {}
    for i in range(3):
        element[f'mae_{i}'] = processor(video[i * 16 : (i + 1) * 16], return_tensors="pt")['pixel_values'].permute(0, 2, 1, 3, 4).contiguous()
    return element


create_mae(ROOT_PATH / 'data/fakeavcelebs/test-set-1')
create_mae(ROOT_PATH / 'data/fakeavcelebs/test-set-2')
create_mae(ROOT_PATH / 'data/fakeavcelebs/train')