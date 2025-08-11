
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from sklearn.model_selection import train_test_split
from csv import DictReader
from random import shuffle


def generate_split(random_state=79098):
    """
    split all subjects (500) to train and test groups in proportion 430:70

    test_sets:
        faceswap
        faceswap-wav2lip
        fsgan
        fsgan-wav2lip
        rtvc
        wav2lip
        test-set-1:  equal number of methods
        test-set-2:  equal number of types
    """

    print("Generating Splits")

    data_path = ROOT_PATH / "data" / "FakeAVCeleb" / "meta_data.csv"

    data = []
    with open(data_path, newline='') as metadata:
        reader = DictReader(metadata)
        for row in reader:
            data.append({
                'source': row['source'],
                'target': row['target1'],
                'method': row['method'],
                'type': row['type'],
                'race': row['race'],
                'gender': row['gender'],
                'path': 'data/' + row[''] + '/' + row['path']
            })

    train = []
    _, test_ids = train_test_split([row['source'] for row in data if row['method'] == 'real'], test_size=70, random_state=random_state)

    # generate TRAIN
    for row in data:
        if row['source'] in test_ids or row['target'] in test_ids:
            continue
        train.append(row)

    # generate TEST
    methods = ["faceswap", "faceswap-wav2lip", "fsgan", "fsgan-wav2lip", "rtvc", "wav2lip"]
    test_groups = dict()
    for method in methods:
        group = []
        for row in data:
            if row['source'] in test_ids and row['method'] == method:
                group.append(row)
        if len(group) > 70:
            _, group = train_test_split(group, test_size=70, random_state=random_state)
        test_groups[method] = group

    test_groups["set-1"] = []
    for meth in methods:
        _, test = train_test_split([row for row in data if row['source'] in test_ids and row['method'] == meth], test_size=11, random_state=random_state)
        test_groups["set-1"] += test

    test_groups["set-2"] = []
    for fake_type in ["FakeVideo-FakeAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio"]:
        _, test = train_test_split([row for row in data if row['source'] in test_ids and row['type'] == fake_type], test_size=23, random_state=random_state)
        test_groups["set-2"] += test

    train_path = ROOT_PATH / "data" / "fakeavcelebs" / "train"
    train_path.mkdir(exist_ok=True, parents=True)
    shuffle(train)
    write_json(train, str(train_path / "split.json"))

    real = [row for row in data if row['source'] in test_ids and row['method'] == 'real']
    test_path = ROOT_PATH / "data" / "fakeavcelebs"
    for method in methods + ["set-1", "set-2"]:
        test_groups[method] += real
        shuffle(test_groups[method])
        (test_path / f"test-{method}").mkdir(exist_ok=True, parents=True)
        write_json(test_groups[method], str(test_path / f"test-{method}" / "split.json"))
    print("Success")


def gen_one_batch():
    print("Generating One Batch")

    data_path = ROOT_PATH / "data" / "miniFakeAVCeleb" / "meta_data.csv"
    data = []
    with open(data_path, newline='') as metadata:
        reader = DictReader(metadata)
        for row in reader:
            data.append({
                'source': row['source'],
                'target': row['target1'],
                'method': row['method'],
                'type': row['type'],
                'race': row['race'],
                'gender': row['gender'],
                'path': 'data/' + row[''] + '/' + row['path']
            })
    shuffle(data)

    one_batch_path = ROOT_PATH / "data" / "fakeavcelebs" / "one_batch"
    one_batch_path.mkdir(exist_ok=True, parents=True)
    write_json(data, str(one_batch_path / "split.json"))
    print("Success")