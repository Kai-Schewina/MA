from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import os
from tqdm import tqdm


def dataframe_from_csv(path, header=0, index_col=None):
    df = pd.read_csv(path, header=header, index_col=index_col)
    df.columns = map(str.lower, df.columns)
    return df


def get_subjects(path):
    if "train" in os.listdir(path):
        subjects_1 = os.listdir(os.path.join(path, "train"))
        subjects_1 = [os.path.join(path, "train", x) for x in subjects_1 if ".csv" not in x]

        subjects_2 = os.listdir(os.path.join(path, "test"))
        subjects_2 = [os.path.join(path, "test", x) for x in subjects_2 if ".csv" not in x]

        subjects = subjects_1 + subjects_2

        if os.path.exists(os.path.join(path, "val")):
            subjects_2 = os.listdir(os.path.join(path, "val"))
            subjects_2 = [os.path.join(path, "val", x) for x in subjects_2 if ".csv" not in x]
            subjects = subjects + subjects_2

        del subjects_1
        del subjects_2
    else:
        subjects = os.listdir(path)
        subjects = [os.path.join(path, x) for x in subjects if ".csv" not in x]
    return subjects


def create_combined_df(subjects):
    dfs = []
    for subj in tqdm(subjects):
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(subj)))
        for ts in patient_ts_files:
            file = pd.read_csv(os.path.join(subj, ts))
            file["subject_id"] = subj
            dfs.append(file)
    combined = pd.concat(dfs)
    return combined
