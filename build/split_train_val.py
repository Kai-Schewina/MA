from __future__ import absolute_import
from __future__ import print_function

import shutil
import os
import pandas as pd
import random
from tqdm import tqdm

##############
# DEPRECATED #
##############

def move_val_to_partition(subjects_root_path, patients):
    for patient in tqdm(patients):
        src = os.path.join(subjects_root_path, "train", str(patient))
        dest = os.path.join(subjects_root_path, "val", str(patient))
        shutil.move(src, dest)


def train_val_split(dataset_dir):
    listfile = pd.read_csv(os.path.join(dataset_dir, "train/listfile.csv"))
    listfile["subjects"] = listfile["stay"].str[:8]
    subjects = list(listfile["subjects"].unique())

    frac = 0.8
    if "ards" in dataset_dir:
        frac = 1.0

    random.seed(42)
    random.shuffle(subjects)
    val_set = subjects[:int((len(subjects) + 1) * (1-frac))]

    train_patients = [x for x in subjects if x not in val_set]
    val_patients = [x for x in subjects if x in val_set]

    # Find out all episodes corresponding to the split patients
    train_listfile = listfile[listfile["subjects"].isin(train_patients)]
    val_listfile = listfile[listfile["subjects"].isin(val_patients)]
    train_listfile = train_listfile.drop("subjects", axis=1)
    val_listfile = val_listfile.drop("subjects", axis=1)
    train_listfile.to_csv(os.path.join(dataset_dir, 'train_listfile.csv'), index=False)
    val_listfile.to_csv(os.path.join(dataset_dir, 'val_listfile.csv'), index=False)
r
    # Copy listfiles
    shutil.copy(os.path.join(dataset_dir, 'test/listfile.csv'),
                os.path.join(dataset_dir, 'test_listfile.csv'))

    # Create own folder
    os.makedirs(os.path.join(dataset_dir, "val"), exist_ok=True)
    move_val_to_partition(dataset_dir, val_listfile["stay"])


if __name__ == '__main__':
    # ards_path = "../data/ards_ihm"
    ards_path = ""
    if os.path.exists(ards_path):
        train_val_split(ards_path)
    else:
        train_val_split("../data/in-hospital-mortality_v6/")

