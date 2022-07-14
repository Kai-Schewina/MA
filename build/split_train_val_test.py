from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import random


def move_to_partition(subjects_root_path, patients, partition):
    if not os.path.exists(os.path.join(subjects_root_path, partition)):
        os.mkdir(os.path.join(subjects_root_path, partition))
    for patient in patients:
        src = os.path.join(subjects_root_path, str(patient))
        dest = os.path.join(subjects_root_path, partition, str(patient))
        shutil.move(src, dest)


def split_train_test(subjects_root_path):
    subjects = os.listdir(subjects_root_path)
    if "listfile.csv" in subjects:
        subjects.remove("listfile.csv")
    if "all_stays.csv" in subjects:
        subjects.remove("all_stays.csv")

    random.seed(2346452394)
    # Train Test Split
    random.shuffle(subjects)
    test_set = subjects[:int((len(subjects) + 1) * .20)]

    train_patients = [x for x in subjects if x not in test_set]
    test_patients = [x for x in subjects if x in test_set]

    # Train Val Split
    val_set = train_patients[:int((len(train_patients) + 1) * .20)]
    val_patients = [x for x in train_patients if x in val_set]
    train_patients = [x for x in train_patients if x not in val_set]

    assert len(set(train_patients) & set(test_patients)) == 0
    assert len(set(train_patients) & set(val_patients)) == 0
    assert len(set(val_patients) & set(test_patients)) == 0

    move_to_partition(subjects_root_path, train_patients, "train")
    move_to_partition(subjects_root_path, test_patients, "test")
    move_to_partition(subjects_root_path, val_patients, "val")


if __name__ == '__main__':
    split_train_test("../data/output/")
