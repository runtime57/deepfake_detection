import av
import numpy as np
from transformers import VivitImageProcessor
from src.utils.io_utils import ROOT_PATH
from src.model.av_hubert.avhubert.preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
import dlib, cv2, os, subprocess, pickle
import numpy as np
import skvideo
import skvideo.io
from csv import DictReader
from pathlib import Path
from tqdm import tqdm


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
    return frames.pixel_values


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

FACE_PREDICTOR_PATH='src/model/shape_predictor/shape_predictor_68_face_landmarks.dat'
CNN_DETECTOR_PATH='src/model/shape_predictor/mmod_human_face_detector.dat'
MEAN_FACE_PATH='src/model/shape_predictor/20words_mean_face.npy'

def load_video(path):
    videogen = skvideo.io.vread(path)
    frames = np.array([frame for frame in videogen])
    return frames


def detect_landmark(image, detector, cnn_detector, predictor):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rects = detector(gray, 1)
        if len(rects) == 0:
            rects = cnn_detector(gray)
            rects = [d.rect for d in rects]
        coords = None
        for (_, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            coords = np.zeros((68, 2), dtype=np.int32)
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords


def preprocess_video(path, dst_path, detector, cnn_detector, predictor, mean_face_landmarks):
    if os.path.exists(dst_path):
        return

    STD_SIZE=(256, 256)
    stablePntsIDs=[33, 36, 39, 42, 45]
    
    frames = load_video(path)
    landmarks = []
    for frame in frames:
        landmark = detect_landmark(frame, detector, cnn_detector, predictor)
        landmarks.append(landmark)
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    
    if not preprocessed_landmarks:
            frame_gen = read_video(video_pathname)
            frames = [cv2.resize(x, (args.crop_width, args.crop_height)) for x in frame_gen]
            write_video_ffmpeg(frames, dst_path, "/usr/bin/ffmpeg")
            return
    else:        
        rois = crop_patch(path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
        write_video_ffmpeg(rois, dst_path, "/usr/bin/ffmpeg")


def preprocess_audio(path, dst_path, ffmpeg):    
    if os.path.exists(dst_path):
        return
    cmd = ffmpeg + " -i " + path + " -f wav -vn -y " + " -ar 16000 -ac 1 " + dst_path + ' -loglevel quiet'
    subprocess.call(cmd, shell=True)


def av_preprocess(path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FACE_PREDICTOR_PATH)
    cnn_detector = dlib.cnn_face_detection_model_v1(CNN_DETECTOR_PATH)
    mean_face_landmarks = np.load(MEAN_FACE_PATH)
    path = str(ROOT_PATH / path)
    mouth_roi_path = '/'.join(path.split('/')[:-1] + ['mouth_roi_' + path.split('/')[-1]])
    wav_path = path.replace('mp4', 'wav')
    preprocess_video(path, mouth_roi_path, detector, cnn_detector, predictor, mean_face_landmarks)
    preprocess_audio(path, wav_path, '/usr/bin/ffmpeg')
    return mouth_roi_path, wav_path