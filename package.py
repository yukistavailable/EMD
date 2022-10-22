import glob
import os
import pickle
import random
from tqdm import tqdm


def pickle_examples_with_split_ratio(
        paths,
        train_path,
        val_path,
        train_val_split=0.1):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory
    """
    with open(train_path, 'wb') as ft:
        with open(val_path, 'wb') as fv:
            for p, label in tqdm(paths):
                label = int(label)
                with open(p, 'rb') as f:
                    img_bytes = f.read()
                    r = random.random()
                    example = (label, img_bytes)
                    if r < train_val_split:
                        pickle.dump(example, fv)
                    else:
                        pickle.dump(example, ft)


def package(
    dir='dir',
    save_dir='experiment/content_data',
    train_test_split_ratio=0.2,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_path = os.path.join(save_dir, "train.obj")
    val_path = os.path.join(save_dir, "val.obj")

    total_file_list = sorted(
        glob.glob(os.path.join(dir, "*.jpg")) +
        glob.glob(os.path.join(dir, "*.png")) +
        glob.glob(os.path.join(dir, "*.tif"))
    )

    cur_file_list = []
    for file_name in tqdm(total_file_list):
        label = os.path.basename(file_name).split('_')[0]
        label = int(label)
        cur_file_list.append((file_name, label))

    pickle_examples_with_split_ratio(
        cur_file_list,
        train_path=train_path,
        val_path=val_path,
        train_val_split=train_test_split_ratio
    )
