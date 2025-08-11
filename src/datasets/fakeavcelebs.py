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
from src.datasets.preprocess import vivit_preprocess, av_preprocess
from src.model.av_hubert.avhubert import custom_utils as custom_utils

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

        image_crop_size = 88
        image_mean = 0.421
        image_std = 0.165

        self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                custom_utils.Normalize(image_mean, image_std) ])


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
        for i, row in tqdm(enumerate(elements)):
            # create dataset
            row_path = row['path']
            label = 1 if row['method'] == 'real' else 0

            video_fn, audio_fn = av_preprocess(row_path)
            av_frames, av_audio = self.hubert_load_feature(video_fn, audio_fn)

            vivit_frames = vivit_preprocess(row_path)

            element_path = data_path / f"{i:06}.safetensors"
            element = {"av_audio": torch.from_numpy(av_audio).float(), "av_frames": torch.from_numpy(av_frames).float(), "vivit_frames": vivit_frames}
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


    def hubert_load_feature(self, video_fn, audio_fn):
        """
        Load image and audio feature
        Returns:
        video_feats: numpy.ndarray of shape [T, H, W, 1], audio_feats: numpy.ndarray of shape [T, F]
        """
        def stacker(feats, stack_order=4):  # video in 25 fps, audio in 100 fps ==> stack_order=4  (check av_hubert/issues/85)
            """
            Concatenating consecutive audio frames
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats

        video_feats = self.hubert_load_video(video_fn) # [T, H, W, 1]

        audio_fn = audio_fn.split(':')[0]
        sample_rate, wav_data = wavfile.read(audio_fn)
        assert sample_rate == 16_000 and len(wav_data.shape) == 1
        audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32) # [T, F]
        audio_feats = stacker(audio_feats) # [T/stack_order_audio, F*stack_order_audio]

        diff = len(audio_feats) - len(video_feats)
        if diff < 0:
            audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
        elif diff > 0:
            audio_feats = audio_feats[:-diff]
        return video_feats, audio_feats

    def hubert_load_video(self, video_fn):
        feats = self.load_video(video_fn)
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats
    
    def load_video(self, path):
        for i in range(3):
            try:
                cap = cv2.VideoCapture(path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frames.append(frame)
                    else:
                        break
                frames = np.stack(frames)
                return frames
            except Exception:
                print(f"failed loading {path} ({i} / 3)")
                if i == 2:
                    raise ValueError(f"Unable to load {path}")

FakeAVCelebsDataset(name="one_batch")