import utils
import os
from build.readers import InHospitalMortalityReader
from preprocessing import Discretizer, Normalizer
import pickle


def main(data, timestep=1.0, normalizer_state="ihm_ts_1.00_impute_previous_start_zero_masks_True_n_20937.normalizer",
         small_part=False):

    # Build readers, discretizers, normalizers
    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data, 'train'),
                                             listfile=os.path.join(data, 'train_listfile.csv'))

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data, 'train'),
                                           listfile=os.path.join(data, 'val_listfile.csv'))

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data, 'test'),
                                            listfile=os.path.join(data, 'test_listfile.csv'),
                                            period_length=48.0)

    discretizer = Discretizer(data_path=data, timestep=float(timestep),
                              store_masks=True,
                              impute_strategy='previous',
                              start_time='zero')

    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)
    normalizer_path = os.path.join(data, normalizer_state)
    normalizer.load_params(normalizer_path)

    train_raw = utils.load_data(train_reader, discretizer, normalizer, small_part)
    val_raw = utils.load_data(val_reader, discretizer, normalizer, small_part)
    test = utils.load_data(test_reader, discretizer, normalizer, small_part, return_names=True)

    with open(os.path.join(data, "train_raw.pkl"), "wb") as f:
        pickle.dump(train_raw, f)
    with open(os.path.join(data, "val_raw.pkl"), "wb") as f:
        pickle.dump(val_raw, f)
    with open(os.path.join(data, "test.pkl"), "wb") as f:
        pickle.dump(test, f)


if __name__ == "__main__":
    main(data="../data/in-hospital-mortality_v4/")
