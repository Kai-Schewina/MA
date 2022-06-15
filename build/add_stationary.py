import os
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
import util


def add_features(subj):
    patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(subj)))
    for ts in patient_ts_files:
        path = os.path.join(subj, ts)
        try:
            episode = pd.read_csv(os.path.join(subj, ts.replace("_timeseries", "")))
            ts = pd.read_csv(path)
            columns = ["Ethnicity", "Gender", "Age", "height", "weight"]
            for col in columns:
                ts[col] = episode[col]
                ts[col] = ts[col].ffill()
            ts.to_csv(path, index=False)
        except pd.errors.EmptyDataError:
            print(subj)
            print(ts)
            continue


def main():
    pool = mp.Pool(processes=mp.cpu_count() - 1)
    path = "../data/output/"
    subjects = util.get_subjects(path)
    for _ in tqdm(pool.imap(add_features, subjects), total=len(subjects)):
        pass


if __name__ == "__main__":
    main()
