import pandas as pd
import os
import random
random.seed(42)

# Supposed to be executed after train/test split, but before train/val split
# This function takes an in-hospital-mortality path as an input and changes the train listfile
# that is used to read in data to achieve majority class undersampling to the quota of the target-train set(ards-ihm)


def undersampling(data, ards_listfile):
    try:
        file_path = os.path.join(data, "train", "listfile.csv")
    except FileNotFoundError:
        file_path = os.path.join(data, "train_listfile.csv")
    ards_file = pd.read_csv(ards_listfile)

    train_file = pd.read_csv(file_path)

    train_file["subject_id"] = train_file["stay"].str[:8]
    ards_file["subject_id"] = ards_file["stay"].str[:8]

    train_grouped = train_file.groupby("subject_id").max("y_true")
    train_alive = list(train_grouped[train_grouped["y_true"] == 0].index)
    train_dead = list(train_grouped[train_grouped["y_true"] == 1].index)
    ards_grouped = ards_file.groupby("subject_id").max("y_true")
    ards_train_dead = list(ards_grouped[ards_grouped["y_true"] == 1].index)

    new_quota = len(ards_train_dead) / len(ards_grouped)
    new_count = (1 / new_quota) * len(train_dead)
    while train_file.shape[0] > new_count:
        random_subject = train_alive[random.randint(0, len(train_alive) - 1)]
        train_file.drop(train_file[train_file["subject_id"] == random_subject].index, inplace=True)

    train_file.drop("subject_id", axis=1, inplace=True)
    train_file.to_csv(os.path.join(data, "train", "train_listfile_balanced.csv"), index=False)
    return str(round(train_file["y_true"].mean(), 2))


if __name__ == "__main__":
    data = "../data/in-hospital-mortality_v5/"
    ards_listfile = "../data/ards_ihm/train_listfile.csv"
    new_quota = undersampling(data, ards_listfile)
    print("New balanced mortality quota: " + new_quota)
