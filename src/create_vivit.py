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


def sample_frames(path: str, num_frames: int = 100):
    vr = VideoReader(path, ctx=cpu(0))
    n = len(vr)
    idx = np.linspace(0, 99, 32, dtype=np.int64) % n
    frames = vr.get_batch(idx).asnumpy()
    return [np.transpose(f, (2, 0, 1)) for f in frames]

processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Large")

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent

def create_mae(data_path):
    elements = read_json(data_path / "split.json")
    current_index = 0
    for i, row in tqdm(enumerate(elements), total=len(elements)):
        vvt_path = data_path / f"{current_index:06}_vvt.safetensors"
        video = sample_frames(str(ROOT_PATH / row['path']))
        element = {}
        for i in range(4):
           element[f'vivit'] = processor(video, return_tensors="pt")['pixel_values'].contiguous()
        safetensors.torch.save_file(element, vvt_path)
        current_index += 1


create_mae(ROOT_PATH / 'data/fakeavcelebs/one_batch')