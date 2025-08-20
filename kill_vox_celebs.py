from src.utils.io_utils import ROOT_PATH, read_json, write_json
from sklearn.model_selection import train_test_split
from csv import DictReader
from random import shuffle
from pathlib import Path
import os 

def get_mp4_paths(directory='.'):
    start_path = Path(directory)
    mp4_list = [
        str(file.absolute())
        for file in start_path.rglob('*.mp4')
    ]
    vox_split = [{'path': '/'.join(full_path.split('/')[6:]), 'method': 'real'} for full_path in mp4_list if 'mouth_roi' not in full_path]
    return vox_split

def get_st_paths(directory='.'):
    start_path = Path(directory)
    mp4_list = [
        str(file.absolute())
        for file in start_path.rglob('*.safetensors')
    ]
    vox_split = [{'path': '/'.join(full_path.split('/')[6:]), 'method': 'real'} for full_path in mp4_list if 'mouth_roi' not in full_path]
    return vox_split

def create_vox_index():
    dir_path = str(ROOT_PATH / 'data/VoxCelebTest/videos')
    videos = get_mp4_paths(dir_path)
    shuffle(videos)
    index = videos[:3570]  # maybe crop to 3230 to get size of train 20'000
    for path in videos[3570:]:
        os.remove(str(ROOT_PATH / path['path']))
    write_json(index, dir_path + '/index.json')

def add_to_train():
    dir_path = str(ROOT_PATH / 'data/VoxCelebTest/videos')
    train_path = str(ROOT_PATH / 'data/fakeavcelebs/train/split.json')
    train_split = read_json(train_path)
    index = read_json(dir_path + '/index.json')
    train_split += index
    os.remove(train_path)
    write_json(train_split, train_path)

add_to_train()