"""
    Created by yuchen on 6/20/18
    Description: This script generate the correct dataset structure for VAE outlier experiments

    souce directory structure:
    src_dir
    ├── test
    │   ├── class1
    │   │   └── 1.jpg
    │   ├── class2
    │   │   └── 1.jpg
    │   └── class3
    │       └── 1.jpg
    └── train
        ├── class1
        │   └── 1.jpg
        ├── class2
        │   └── 1.jpg
        └── class3
            └── 1.jpg

    The output folder has following structure:
    tar_dir
    ├── vae_test
    │   ├── diseaseN (100 of diseaseN)
    │   ├── disease2 (100 of disease2)
    │   ├── disease1 (100 of disease1)
    │   └── NV (basically 250 val images)
    └── vae_train
        ├── val
        │   └── normal (5% of images without class N)
        └── train
            └── normal (95% of images without class N)

    We avoid copy images by create symlink to the images in src_dir
"""
import argparse
import os
from os.path import join as pathjoin
import glob
import random
from tqdm import tqdm
import logging
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', required=True, type=str)
parser.add_argument('--tar_dir', required=True, type=str)
parser.add_argument('--normal', action='store_true',
                    help='if this flag is set to be true, reverse normal and abnormal')
args = parser.parse_args()
SRC_DIR = args.src_dir
TAR_DIR = args.tar_dir

# check if src_dir exists
if not os.path.isdir(SRC_DIR):
    logging.error("src_dir %s not exists!", SRC_DIR)
    exit()

# check if tar_dir exists
if os.path.exists(TAR_DIR):
    logging.error("tar_dir %s exists!", TAR_DIR)
    exit()
os.mkdir(TAR_DIR)

# list all classes
classes = os.listdir(pathjoin(SRC_DIR, 'train'))
if '.DS_Store' in classes:
    classes.remove('.DS_Store')
classes.sort()
paths = [os.path.abspath(path) for path in glob.iglob(pathjoin(SRC_DIR, '**', '*.jpeg'), recursive=True)]

def get_class(img_path, classes):
    """
    get class name from the image path
    """
    for cls in classes:
        if cls in img_path:
            return cls
    logging.error("img_path %s is invalid!", img_path)

# arrange path according to class
paths_by_class = {cls: [] for cls in classes}
for path in paths:
    paths_by_class[get_class(path, classes)].append(path)

# split 95 - 5 train/test split
nv_paths = paths_by_class['NORMAL']
random.shuffle(nv_paths)
split = int(0.95 * len(nv_paths))
nv_train_paths = nv_paths[:split]
nv_val_paths = nv_paths[split:]
logging.info('num. train images: {}\tnum. val images: {}'.format(
             len(nv_train_paths), len(nv_val_paths)))

def symlink_without_replace(src_file, tar_dir):
    """
    create a symlink in `tar_dir` as `tar_dir/src_file`. If there exists
    `src_file` in `tar_dir`, use `tar_dir/1_src_file`, `tar_dir/11_src_file` etc.
    :param src_file: a path to a file
    :param tar_dir: a path to a dir
    """
    assert os.path.isfile(src_file)
    assert os.path.isdir(tar_dir)
    img_id = paths.index(src_file)
    file_name = '{}.jpg'.format(img_id)
    if file_name in os.listdir(tar_dir):
        logging.error('repeat file name. Impossible!')
    os.symlink(src_file, pathjoin(tar_dir, file_name))


# make folders
vae_train_folder = pathjoin(args.tar_dir, 'vae_train')
vae_test_folder = pathjoin(args.tar_dir, 'vae_test')
os.mkdir(vae_train_folder)
os.mkdir(vae_test_folder)

# vae_test_folder
for cls in classes:
    test_cls_folder = pathjoin(vae_test_folder, cls)
    os.mkdir(test_cls_folder)

# vae_train_folder
vae_train_train_folder = pathjoin(vae_train_folder, 'train')
vae_train_val_folder = pathjoin(vae_train_folder, 'val')
os.mkdir(vae_train_train_folder)
os.mkdir(vae_train_val_folder)
vae_train_train_normal_folder = pathjoin(vae_train_train_folder, 'normal')
vae_train_val_normal_folder = pathjoin(vae_train_val_folder, 'normal')
os.mkdir(vae_train_train_normal_folder)
os.mkdir(vae_train_val_normal_folder)

# fill in vae_train/train folder
for img in tqdm(nv_train_paths):
    symlink_without_replace(src_file=img, tar_dir=vae_train_train_normal_folder)

# fill in vae_train/val
for img in tqdm(nv_val_paths):
    symlink_without_replace(src_file=img, tar_dir=vae_train_val_normal_folder)

# fill in vae_test/NV
for img in nv_val_paths[:250]:
    symlink_without_replace(src_file=img, tar_dir=pathjoin(vae_test_folder, 'NORMAL'))

# fill in vae_test/other_class
for cls in classes:
    for img in paths_by_class[cls][:100]:
        symlink_without_replace(src_file=img, tar_dir=pathjoin(vae_test_folder, cls))
