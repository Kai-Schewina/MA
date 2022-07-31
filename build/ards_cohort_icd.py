import os
from sqlalchemy import create_engine
import pandas as pd
import shutil
from tqdm import tqdm
import random
from split_train_val_test import move_to_partition
from util import get_subjects


def count_icd(subj_root_path):
    subject_list = get_subjects(subj_root_path)
    subject_list = [s.split(sep="\\")[1] for s in subject_list]
    subject_list = list(map(int, subject_list))

    engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres')

    icd_10_j80 = pd.read_sql_query("select * from mimic_hosp.diagnoses_icd where icd_code LIKE 'J80%'", con=engine)
    icd_10_j80 = icd_10_j80[icd_10_j80['subject_id'].isin(subject_list)]
    icd_9_51882 = pd.read_sql_query("select * from mimic_hosp.diagnoses_icd where icd_code = '51882'", con=engine)
    icd_9_51882 = icd_9_51882[icd_9_51882['subject_id'].isin(subject_list)]

    both_hadm = icd_10_j80['hadm_id'].tolist() + icd_9_51882['hadm_id'].tolist()

    return list(set(both_hadm))


def create_cohort_icd(subj_root_path, out_path, cohort):
    subject_list = get_subjects(subj_root_path)

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for subject in tqdm(subject_list):
        file_list = ["events.csv", "stays.csv"]
        stays = pd.read_csv(os.path.join(subject, "stays.csv"))
        hadms = stays["hadm_id"].tolist()
        for i, e in enumerate(hadms):
            if e in cohort:
                file_list.append("episode" + str(i + 1) + ".csv")
                file_list.append("episode" + str(i + 1) + "_timeseries.csv")
        if len(file_list) > 2:
            subj_out_path = os.path.join(out_path, subject.split(sep="\\")[1])
            if not os.path.exists(subj_out_path):
                os.mkdir(subj_out_path)
            for e in file_list:
                shutil.copy(os.path.join(subject, e), os.path.join(subj_out_path, e))


def create_all_stays(in_path, out_path, cohort_hadm):
    all_stays = pd.read_csv(os.path.join(in_path, "all_stays.csv"))
    all_stays = all_stays[all_stays["hadm_id"].isin(cohort_hadm)]
    all_stays.to_csv(os.path.join(out_path, "all_stays.csv"))


def train_test_split(subjects_root_path):
    all_stays = pd.read_csv(os.path.join(subjects_root_path, "all_stays.csv"))
    subjects = list(all_stays["subject_id"].unique())

    random.seed(2346452394)
    # Train Test Split
    random.shuffle(subjects)
    test_set = subjects[:int((len(subjects) + 1) * .50)]

    train_patients = [x for x in subjects if x not in test_set]
    test_patients = [x for x in subjects if x in test_set]

    assert len(set(train_patients) & set(test_patients)) == 0

    move_to_partition(subjects_root_path, train_patients, "train")
    move_to_partition(subjects_root_path, test_patients, "test")


def main():
    in_path = "../data/output/"
    out_path = "../data/ards_icd/"
    cohort_hadm = count_icd(in_path)
    create_cohort_icd(in_path, out_path, cohort_hadm)
    create_all_stays(in_path, out_path, cohort_hadm)
    train_test_split(out_path)


if __name__ == "__main__":
    main()
