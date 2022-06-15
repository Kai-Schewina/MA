from __future__ import absolute_import
from __future__ import print_function

import shutil
import argparse
import os
import random


def main(dataset_dir):

    with open(os.path.join(dataset_dir, 'train/listfile.csv')) as listfile:
        lines = listfile.readlines()
        header = lines[0]
        lines = lines[1:]

    random.shuffle(lines)
    val_patients = lines[:int((len(lines) + 1) * .20)]

    train_lines = [x for x in lines if x not in val_patients]
    val_lines = [x for x in lines if x in val_patients]
    assert len(train_lines) + len(val_lines) == len(lines)

    with open(os.path.join(dataset_dir, 'train_listfile.csv'), 'w') as train_listfile:
        train_listfile.write(header)
        for line in train_lines:
            train_listfile.write(line)

    with open(os.path.join(dataset_dir, 'val_listfile.csv'), 'w') as val_listfile:
        val_listfile.write(header)
        for line in val_lines:
            val_listfile.write(line)

    shutil.copy(os.path.join(dataset_dir, 'test/listfile.csv'),
                os.path.join(dataset_dir, 'test_listfile.csv'))


if __name__ == '__main__':
    main("../data/in-hospital-mortality_v5/")
