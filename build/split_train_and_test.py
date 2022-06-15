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
    random.shuffle(subjects)
    test_set = subjects[:int((len(subjects) + 1) * .20)]

    train_patients = [x for x in subjects if x not in test_set]
    test_patients = [x for x in subjects if x in test_set]

    assert len(set(train_patients) & set(test_patients)) == 0

    move_to_partition(subjects_root_path, train_patients, "train")
    move_to_partition(subjects_root_path, test_patients, "test")


if __name__ == '__main__':
    split_train_test("../data/output/")
