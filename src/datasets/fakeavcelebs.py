import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import safetensors
import safetensors.torch
import shutil
import fairseq
import os
from argparse import Namespace
from transformers import VivitModel, AutoVideoProcessor
from tqdm.auto import tqdm
from scipy.io import wavfile
from python_speech_features import logfbank
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from src.utils.split_utils import generate_split, gen_one_batch
# from src.datasets.preprocess import Processor
from transformers import VideoMAEImageProcessor
from decord import VideoReader, cpu

class FakeAVCelebsDataset(BaseDataset):
    """
    index contains next columns:
        path (str):  path to elemnt
        label (int): fake or real

    and each element contains:
        av_feats (torch.Tensor):     extracted AV-Hubert features
        vivit_feats (torch.Tensor):  extracted ViViT features
        aasist_audio (torch.Tensor): extracted audio for AASIST (4 seconds / 64600 ticks)
    """

    def __init__(self, name="train", *args, **kwargs):
        """
        Args:
            name (str): partition name
        """
        index_path = ROOT_PATH / "data" / "fakeavcelebs" / name / "index.json"
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)
        self._paths = [elem['path'] for elem in read_json(ROOT_PATH / "data" / "fakeavcelebs" / name / "split.json")]
        self._vjepaproc = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        """
        Create index.json for given partition.

        Args:
            name (str): partition name

        """
        if name == "one_batch":
           gen_one_batch()
        # else:
        #    generate_split()

        index = []
        data_path = ROOT_PATH / "data" / "fakeavcelebs" / name
        elements = read_json(str(data_path / "split.json"))

        print("Creating FakeAVCelebs Dataset")
        # processor = Processor()
        current_index = 0
        failed = 0
        for i, row in tqdm(enumerate(elements), total=len(elements)):
            # create dataset
            element_path = data_path / f"{current_index:06}.safetensors"
            # if os.path.exists(element_path): 
            #     current_index += 1
            #     continue
            st_path = str(ROOT_PATH / row['path'].replace('mp4', 'safetensors'))
            label = 1 if row['method'] == 'real' else 0
            element = safetensors.torch.load_file(st_path)
            # --- add arc ---
            # arc_feats = np.expand_dims(np.vstack(self.arc.get_feats(row['path'])), axis=0)
            # element['arc_feats'] = arc_feats
            # safetensors.torch.save_file(element, st_path)
            # --- continue ---
            element_path = data_path / f"{current_index:06}.safetensors"
            safetensors.torch.save_file(element, element_path)
            index.append({"path": str(element_path), "label": label})
            current_index += 1

        print(f"Total number: {len(elements)}")
        print(f"Processed: {current_index}")
        print(f"Failed: {failed}")
        write_json(index, str(data_path / "index.json"))
        return index

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        data_path = data_dict["path"]
        obj = self.load_object(data_path)
        instance_data = {"labels": data_dict["label"]}
        for key in obj:
            instance_data[key] = obj[key]
        
        instance_data['vjepa_frames'] = self._sample_frames(self._paths[ind])
        
        # obj = self.load_object(data_path.replace('.safetensors', '_vivit.safetensors'))
        # instance_data['vivit_frames'] = obj['vivit']
        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def _sample_frames(self, path: str, num_frames: int = 75):
        vr = VideoReader(str(ROOT_PATH / path), ctx=cpu(0))
        n = len(vr)
        idx = np.arange(0, 75) % n
        frames = vr.get_batch(idx).asnumpy()
        return self._vjepaproc(frames, return_tensors="pt")['pixel_values_videos']

    
