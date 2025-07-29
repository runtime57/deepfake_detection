import numpy as np
import torch
import torchvision
import safetensors
import safetensors.torch
import shutil
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from src.utils.split_utils import generate_split
from src.datasets.preprocess import vivit_preprocess, av_preprocess


class FakeAVCelebsDataset(BaseDataset):
    """
    index contains next columns:
        path (str):  path to elemnt
        label (int): fake or real

    and each element contains:
        av_audio (torch.Tensor):     preprocessed audio for AV-Hubert
        av_frames (torch.Tensor):    preprocessed frames for AV-Hubert
        vivit_frames (torch.Tensor): preprocessed frames for ViViT
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

        super.__init__(index, *args, **kwargs)

    def _create_index(self, name):
        """
        Create index.json for given partition.

        Args:
            name (str): partition name

        """
        generate_split()

        index = []
        data_path = ROOT_PATH / "data" / "fakeavcelebs" / name
        elements = read_json(str(data_path / "split.json"))

        print("Creating FakeAVCelebs Dataset")
        for i, row in tqdm(enumerate(elements)):
            # create dataset
            row_path = row['path']
            label = 1 if row['method'] == 'real' else 0

            av_audio, av_frames = av_preprocess(row_path)
            vivit_frames = vivit_preprocess(row_path)

            element_path = data_path / f"{i:06}.safetensors"
            element = {"av_audio": av_audio, "av_frames": av_frames, "vivit_frames": vivit_frames}
            safetensors.torch.save_file(element, element_path)

            index.append({"path": str(element_path), "label": label})
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
        av_audio = obj["av_audio"]
        av_frames = obj["av_frames"]
        vivit_frames = obj["vivit_frames"]
        label = data_dict["label"]

        instance_data = {"av_audio": av_audio, "av_frames": av_frames, "vivit_frames": vivit_frames, "labels": label}
        instance_data = self.preprocess_data(instance_data)

        return instance_data
