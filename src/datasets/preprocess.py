import av
import numpy as np
from transformers import VivitImageProcessor
from src.utils.io_utils import ROOT_PATH
from src.model.av_hubert.avhubert.preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
import dlib, cv2, os, subprocess, pickle
import numpy as np
import skvideo
import skvideo.io
from scipy.io import wavfile
from csv import DictReader
from pathlib import Path
from tqdm import tqdm
from src.model.av_hubert.avhubert import custom_utils as custom_utils
from python_speech_features import logfbank
import cv2
import torch
import torchvision
import safetensors
import safetensors.torch
import shutil


# ViViT Preprocess

class ViViT_Processor():
    def __init__(self):
        self.image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

    def vivit_preprocess(self, file_path, must=0):
        """
        Args:
            file_path (str): filename of the mp4 file.
        Returns:
            vivit_frames (torch.Tensor): preprocessed frames.
        """
        # we need to sample 32 frames with stide 2 (check ViViT: A Video Vision Transformer)
        # It's ok with our fps 25, because model was trained on Kinetiks Dataset (also 25 fps)
        container = av.open(str(ROOT_PATH / file_path))

        indices = self.sample_frame_indices(clip_len=32, frame_sample_rate=2, seg_len=container.streams.video[0].frames)
        video = self.read_video_pyav(container=container, indices=indices)

        frames = self.image_processor(list(video), return_tensors="pt").pixel_values.clone()
        if frames.shape[1] < 32:
            num = 32 // frames.shape[1] + 1
            frames = frames.repeat(1, num, 1, 1, 1)[:, :32, :, :]
        return frames


    def read_video_pyav(self, container, indices):
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


    def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
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
        end_idx = np.random.randint(min(converted_len, seg_len-1), seg_len)
        start_idx = max(0, end_idx - converted_len)
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices


# AV Preprocess

class AV_Processor:
    def __init__(self):
        self.FACE_PREDICTOR_PATH='src/model/shape_predictor/shape_predictor_68_face_landmarks.dat'
        self.CNN_DETECTOR_PATH='src/model/shape_predictor/mmod_human_face_detector.dat'
        self.MEAN_FACE_PATH='src/model/shape_predictor/20words_mean_face.npy'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(ROOT_PATH / self.FACE_PREDICTOR_PATH))
        self.cnn_detector = dlib.cnn_face_detection_model_v1(str(ROOT_PATH / self.CNN_DETECTOR_PATH))
        self.mean_face_landmarks = np.load(str(ROOT_PATH / self.MEAN_FACE_PATH))


    def load_video(self, path):
        videogen = skvideo.io.vread(str(ROOT_PATH / path), outputdict={"-s": '224x224'})
        frames = np.array([frame for frame in videogen])
        return frames

    def read_video(self, filename):
        cap = cv2.VideoCapture(filename)
        while(cap.isOpened()):                                                 
            ret, frame = cap.read() # BGR
            if ret:                      
                yield frame                                                    
            else:                                                              
                break                                                         
        cap.release()


    def detect_landmark(self, image):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            rects = self.detector(gray, 1)
            if len(rects) == 0:
                rects = self.cnn_detector(gray)
                rects = [d.rect for d in rects]
            coords = None
            for (_, rect) in enumerate(rects):
                shape = self.predictor(gray, rect)
                coords = np.zeros((68, 2), dtype=np.int32)
                for i in range(0, 68):
                    coords[i] = (shape.part(i).x, shape.part(i).y)
            return coords


    def preprocess_video(self, path, dst_path, must):
        if not must and os.path.exists(dst_path):
            return

        STD_SIZE=(256, 256)
        stablePntsIDs=[33, 36, 39, 42, 45]
        
        frames = self.load_video(path)
        landmarks = []
        for frame in frames:
            landmark = self.detect_landmark(frame)
            landmarks.append(landmark)
        preprocessed_landmarks = landmarks_interpolate(landmarks)
        
        # print(preprocessed_landmarks)

        if not preprocessed_landmarks:
                print("A"*500)
                frame_gen = self.read_video(path)
                frames = [cv2.resize(x, (96, 96)) for x in frame_gen]
                write_video_ffmpeg(frames, dst_path, "/usr/bin/ffmpeg")
                return
        else:        
            rois = crop_patch(path, preprocessed_landmarks, self.mean_face_landmarks, stablePntsIDs, STD_SIZE, window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
            write_video_ffmpeg(rois, dst_path, "/usr/bin/ffmpeg")


    def preprocess_audio(self, path, dst_path, ffmpeg, must):
        if not must and os.path.exists(dst_path):
            return
        cmd = f'{ffmpeg} -i "{path}" -f wav -vn -y -ar 16000 -ac 1 "{dst_path}" -loglevel quiet'
        subprocess.call(cmd, shell=True)


    def av_preprocess(self, path, must=0):
        path = str(ROOT_PATH / path)
        mouth_roi_path = '/'.join(path.split('/')[:-1] + ['mouth_roi_' + path.split('/')[-1]])
        wav_path = path.replace('mp4', 'wav')
        self.preprocess_video(path, mouth_roi_path, must)
        self.preprocess_audio(path, wav_path, '/usr/bin/ffmpeg', must)
        return mouth_roi_path, wav_path


def aasist_load(audio_fn, max_len: int = 64600):
    sample_rate, x = wavfile.read(audio_fn)
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]
    return x   

def av_crop(x, max_len = 75):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len].clone()
    return x

class Processor:
    def __init__(self, name="train"):
        image_crop_size = 88
        image_mean = 0.421
        image_std = 0.165

        self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                custom_utils.Normalize(image_mean, image_std) ])
        index_path = ROOT_PATH / "data" / "fakeavcelebs" / name / "index.json"
        
        self.avp = AV_Processor()
        self.vvtp = ViViT_Processor()

    def run(self, row, must=0):
        row_path = row['path']
        label = 1 if row['method'] == 'real' else 0
        
        st_path = str(ROOT_PATH / row_path.replace('mp4', 'safetensors'))
        if not must and os.path.exists(st_path):
            return st_path
        video_fn, audio_fn = self.avp.av_preprocess(row_path, must)
        av_frames, av_audio = self.hubert_load_feature(video_fn, audio_fn)
        vivit_frames = self.vvtp.vivit_preprocess(row_path, must)
        aasist_audio = aasist_load(audio_fn)
        element = {
            "av_audio": av_crop(torch.from_numpy(av_audio)),
            "av_frames": av_crop(torch.from_numpy(av_frames)).to(torch.uint8), 
            "vivit_frames": vivit_frames.clone().to(torch.uint8), 
            "aasist_audio": torch.from_numpy(aasist_audio).clone()
        }
        safetensors.torch.save_file(element, st_path)
        os.remove(video_fn)
        os.remove(audio_fn)
        return st_path

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
        for i in range(5):
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
                print(f"failed loading {path} ({i} / 5)")
                if i == 3:
                    av = self.avp.av_preprocess('/'.join(path.split('/')[:-1] + [path.split('/')[-1].replace('mouth_roi_', '')]), must=1)
                if i == 4:
                    raise ValueError(f"Unable to load {path}")