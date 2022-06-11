import pandas as pd
import os
import random
random.seed(42)

# This function takes an in-hospital-mortality path as an input and changes the train listfile
# that is used to read in data to achieve majority class undersampling


def undersampling(data):
    train_file = pd.read_csv(data)

    train_file["subject_id"] = train_file["stay"].str[:8]

    train_grouped = train_file.groupby("subject_id").max("y_true")
    train_alive = list(train_grouped[train_grouped["y_true"] == 0].index)
    train_dead = list(train_grouped[train_grouped["y_true"] == 1].index)

    while train_file.shape[0] > (len(train_dead) * 2):
        random_subject = train_alive[random.randint(0, len(train_alive) - 1)]
        train_file.drop(train_file[train_file["subject_id"] == random_subject].index, inplace=True)

    train_file.drop("subject_id", axis=1, inplace=True)
    train_file.to_csv(os.path.join(data, "train_listfile_balanced.csv"), index=False)
    return str(round(train_file["y_true"].mean(), 2))


if __name__ == "__main__":
    data = "../data/in-hospital-mortality_v3/train_listfile.csv"
    new_quota = undersampling(data)
    print("New balanced mortality quota: " + new_quota)
