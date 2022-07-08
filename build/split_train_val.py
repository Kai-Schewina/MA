from __future__ import absolute_import
from __future__ import print_function

import shutil
import os
import pandas as pd


def train_val_split(dataset_dir):
    listfile = pd.read_csv(os.path.join(dataset_dir, "train/listfile.csv"))
    listfile["subjects"] = listfile["stay"].str[:8]
    subjects = pd.DataFrame(listfile["subjects"].unique())

    frac = 0.8
    if "ards" in dataset_dir:
        frac = 1.0
    train_patients = subjects.sample(frac=frac)
    val_patients = subjects.drop(train_patients.index)

    train_listfile = listfile[listfile["subjects"].isin(train_patients.iloc[:, 0])]
    val_listfile = listfile[listfile["subjects"].isin(val_patients.iloc[:, 0])]
    train_listfile = train_listfile.drop("subjects", axis=1)
    val_listfile = val_listfile.drop("subjects", axis=1)
    train_listfile.to_csv(os.path.join(dataset_dir, 'train_listfile.csv'), index=False)
    val_listfile.to_csv(os.path.join(dataset_dir, 'val_listfile.csv'), index=False)

    shutil.copy(os.path.join(dataset_dir, 'test/listfile.csv'),
                os.path.join(dataset_dir, 'test_listfile.csv'))


if __name__ == '__main__':
    ards_path = "../data/ards_ihm"
    if os.path.exists(ards_path):
        train_val_split(ards_path)
    else:
        train_val_split("../data/in-hospital-mortality_v5/")

