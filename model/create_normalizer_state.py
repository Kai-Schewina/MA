from __future__ import absolute_import
from __future__ import print_function

from build.readers import InHospitalMortalityReader
from preprocessing import Discretizer, Normalizer
import os
import utils
import pickle


def main(task="ihm", timestep=1.0, impute_strategy="previous", start_time="zero",
         store_masks=True, n_samples=-1, data_path="", balanced=False):

    dataset_dir = os.path.join(data_path, 'train')
    if balanced:
        reader = InHospitalMortalityReader(dataset_dir=dataset_dir, period_length=48.0,
                                           listfile=os.path.join(data_path, 'train_listfile_balanced.csv'))
    else:
        reader = InHospitalMortalityReader(dataset_dir=dataset_dir, period_length=48.0)

    # create the discretizer
    discretizer = Discretizer(timestep=float(timestep),
                              store_masks=store_masks,
                              impute_strategy=impute_strategy,
                              start_time=start_time,
                              data_path=data_path)

    discretizer_header = discretizer.transform(reader.read_example(0)["X"])[1].split(',')
    continuous_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=continuous_channels)

    n_samples = n_samples
    if n_samples == -1:
        n_samples = reader.get_number_of_examples()

    for i in range(n_samples):
        if i % 1000 == 0:
            print('Processed {} / {} samples'.format(i, n_samples), end='\r')
        ret = reader.read_example(i)
        data, new_header = discretizer.transform(ret['X'], end=ret['t'])
        normalizer.feed_data(data)
    print('\n')

    if balanced:
        file_name = '{}_ts_{:.2f}_impute_{}_start_{}_masks_{}_n_{}_balanced.normalizer'.format(
            task, timestep, impute_strategy, start_time, store_masks, n_samples)
    else:
        file_name = '{}_ts_{:.2f}_impute_{}_start_{}_masks_{}_n_{}.normalizer'.format(
            task, timestep, impute_strategy, start_time, store_masks, n_samples)
    file_name = data_path + file_name
    print('Saving the state in {} ...'.format(file_name))
    normalizer.save_params(file_name)


if __name__ == '__main__':
    main(data_path="../data/in-hospital-mortality_v5/", balanced=False)

