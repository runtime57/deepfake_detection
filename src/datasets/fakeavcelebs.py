import numpy as np
import cv2
import torch
import torchvision
import safetensors
import safetensors.torch
import shutil
from tqdm.auto import tqdm
from scipy.io import wavfile
from python_speech_features import logfbank
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from src.utils.split_utils import generate_split, gen_one_batch
from src.datasets.preprocess import Processor

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

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        """
        Create index.json for given partition.

        Args:
            name (str): partition name

        """
        if name == "one_batch":
            gen_one_batch()
        else:
            generate_split()

        index = []
        data_path = ROOT_PATH / "data" / "fakeavcelebs" / name
        elements = read_json(str(data_path / "split.json"))

        print("Creating FakeAVCelebs Dataset")
        
        processor = Processor()
        current_index = 0
        failed = 0
        for i, row in tqdm(enumerate(elements), total=len(elements)):
            # create dataset
            st_path = processor.run(row)
            label = 1 if row['method'] == 'real' else 0
            if st_path is None:
                failed += 1
                print(f"Failed: {row['path']}")
                continue
            element = safetensors.torch.load_file(st_path)
            element_path = data_path / f"{current_index:06}.safetensors"
            current_index += 1
            safetensors.torch.save_file(element, element_path)
            index.append({"path": str(element_path), "label": label})
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
        av_audio = obj["av_audio"]
        av_frames = obj["av_frames"]
        vivit_frames = obj["vivit_frames"]
        aasist_audio = obj["aasist_audio"]
        label = data_dict["label"]

        instance_data = {"av_audio": av_audio, "av_frames": av_frames, "vivit_frames": vivit_frames, "aasist_audio": aasist_audio, "labels": label}
        instance_data = self.preprocess_data(instance_data)

        return instance_data
