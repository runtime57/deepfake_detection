import shutil
from src.utils.io_utils import ROOT_PATH
from csv import DictReader
import dlib
import cv2
import skvideo.io
import tqdm
import os
import numpy as np
import src.model.av_hubert.fairseq
import src.model.av_hubert.avhubert.hubert_pretraining
import src.model.av_hubert.avhubert.hubert
from src.model.av_hubert.fairseq.fairseq import checkpoint_utils, options, tasks, utils
from src.model.av_hubert.avhubert.preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
import src.model.av_hubert.avhubert.utils as avhubert_utils

# from AV-Hubert colab


def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def preprocess_video(cfg, input_video_path, output_video_path, face_predictor_path, mean_face_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    STD_SIZE = (256, 256)
    mean_face_landmarks = np.load(mean_face_path)
    stablePntsIDs = [33, 36, 39, 42, 45]
    videogen = skvideo.io.vread(input_video_path)
    frames = np.array([frame for frame in videogen])
    landmarks = []
    for frame in frames:
        landmark = detect_landmark(frame, detector, predictor)
        landmarks.append(landmark)
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    
    rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE,
                      window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
    print(rois.shape)
    transform = avhubert_utils.Compose([
      avhubert_utils.Normalize(0.0, 255.0),
      avhubert_utils.CenterCrop((cfg.image_crop_size, cfg.image_crop_size)),
      avhubert_utils.Normalize(cfg.image_mean, cfg.image_std)])
    rois = transform(rois)

    ffmpeg_path = shutil.which("ffmpeg") # or shutil.which("ffmpeg.exe")
    write_video_ffmpeg(rois, output_video_path, ffmpeg_path)


def _play_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл!")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def create_mouth_videos(data_path=ROOT_PATH / "data" / "FakeAVCeleb" / "meta_data.csv", ckpt_path = 'C:\\Users\\oslik\\hse\\DL\\deepfake_detection\\src\\model\\av_hubert\\ckpt\\base_vox_433h.pt'):
    """
    Create videos with only mouth region
    Args:
        data_path: path to metadata.csv of dataset
    """
    face_predictor_path = str(ROOT_PATH / 'src' / 'model' / 'shape_predictor' / 'shape_predictor_68_face_landmarks.dat')
    mean_face_path = str(ROOT_PATH / 'src' / 'model' / 'shape_predictor' / '20words_mean_face.npy')
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    # with open(data_path, newline='') as metadata:
    #     reader = DictReader(metadata)
    #     for row in reader:
    #         video_path = str(ROOT_PATH / 'data' / row[''] / row['path'])
    #         mouth_roi_path =  str(ROOT_PATH / 'data' / row[''] / str('mouth_roi_' + row['path']))
    #         # print(os.path.exists(os.path.dirname(mouth_roi_path)))
    #         preprocess_video(task.cfg, video_path, mouth_roi_path, face_predictor_path, mean_face_path)
    #         # print(os.path.exists(os.path.dirname(mouth_roi_path)))
    #         _play_video(mouth_roi_path)
    #         break


create_mouth_videos()
