import utils
import os
from build.readers import InHospitalMortalityReader
from preprocessing import Discretizer, Normalizer
import pickle
import shutil


def main(data, full_data_path, timestep=1.0, normalizer_state="", small_part=False, remove_outliers=False):

    # Copy files
    files = os.listdir(full_data_path)
    for e in files:
        if ".normalizer" in e or ".json" in e:
            shutil.copy(os.path.join(full_data_path, e), os.path.join(data, e))
        if ".normalizer" in e and "balanced" not in e and normalizer_state == "":
            normalizer_state = e

    # Build readers, discretizers, normalizers
    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data, 'train'),
                                             listfile=os.path.join(data, 'train_listfile.csv'))

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data, 'test'),
                                            listfile=os.path.join(data, 'test_listfile.csv'),
                                            period_length=48.0)

    discretizer = Discretizer(data_path=data, timestep=float(timestep),
                              store_masks=True,
                              impute_strategy='previous',
                              start_time='zero',
                              remove_outliers=remove_outliers)

    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)
    normalizer_path = os.path.join(data, normalizer_state)
    normalizer.load_params(normalizer_path)

    train_raw = utils.load_data(train_reader, discretizer, normalizer, small_part, return_names=True)
    test = utils.load_data(test_reader, discretizer, normalizer, small_part, return_names=True)

    with open(os.path.join(data, "train_raw.pkl"), "wb") as f:
        pickle.dump(train_raw, f)
    with open(os.path.join(data, "test.pkl"), "wb") as f:
        pickle.dump(test, f)


if __name__ == "__main__":
    main(data="../data/ards_ihm_v2/", full_data_path="../data/in-hospital-mortality_v7/", remove_outliers=True)
