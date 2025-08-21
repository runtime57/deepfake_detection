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

class FeatsFakeAVCelebsDataset(BaseDataset):
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
        index_path = ROOT_PATH / "data" / "featsfakeavcelebs" / name / "index.json"
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
        data_path = ROOT_PATH / "data" / "featsfakeavcelebs" / name
        elements = read_json(str(data_path / "split.json"))

        print("Instantiating models")
        ckpt_path = '/home/runtime57/hse/coursework_2/deepfake_detection/src/model/av_hubert/ckpt/base_vox_433h.pt'
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        avhubert = models[0]
        if hasattr(models[0], 'decoder'):
            avhubert = models[0].encoder.w2v_model
        vivit = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")

        print("Creating FakeAVCelebs Dataset")
        processor = Processor()
        current_index = 0
        failed = 0
        for i, row in tqdm(enumerate(elements), total=len(elements)):
            # create dataset
            st_path = processor.run(row)
            label = 1 if row['method'] == 'real' else 0
            row_element = safetensors.torch.load_file(st_path)
            av_feats, vivit_feats = self._extract_feats(avhubert, vivit, row_element['av_video'], row_element['av_audio'], row_element['vivit_frames'])

            element = { 'av_feats': av_pooled_feats, 'vivit_feats': vivit_feats, 'aasist_audio': row_element['aasist_audio'] }
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
        instance_data = self.preprocess_data(instance_data)

        return instance_data
    
    def _extract_feats(self, avhubert, vivit, av_video, av_audio, vivit_frames):
            def av_video_pad(self, x, max_len = 75):
                x_len = x.shape[0]
                if x_len >= max_len:
                    return x[:max_len].clone()
                # if too short
                num_repeats = int(max_len / x_len) + 1
                padded_x = x.repeat(num_repeats, 1, 1, 1)[:max_len]
                return padded_x.clone()

            def av_audio_pad(self, x, max_len = 75):
                x_len = x.shape[0]
                if x_len >= max_len:
                    return x[:max_len].clone()
                # if too short
                num_repeats = int(max_len / x_len) + 1
                padded_x = x.repeat(num_repeats, 1)[:max_len]
                return padded_x.clone()

        with torch.inference_mode():
                av_feats, _ = avhubert.extract_finetune(source={'video': av_video_pad(av_video).unsqueeze(0), 'audio': av_audio_pad(av_audio).unsqueeze(0)}, padding_mask=None, output_layer=None)
                vivit_feats = vivit(pixel_values=vivit_frames).last_hidden_state[:, 0, :]
        av_pooled_feats = av_feats.mean(dim=1)
        return av_pooled_feats, vivit_feats
    