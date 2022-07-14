from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import platform
import pickle
import json
import os
from tqdm import tqdm

import pandas as pd


class Discretizer:
    def __init__(self, data_path, timestep=1.0, store_masks=True, impute_strategy='zero', start_time='zero',
                 config_path=os.path.join(os.path.dirname(__file__), '../resources/discretizer_config.json'),
                 remove_outliers=False):

        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config['id_to_channel']
            self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
            self._is_categorical_channel = config['is_categorical_channel']
            self._possible_values = config['possible_values']
            self._normal_values = config['normal_values']

        self._header = ["Hours"] + self._id_to_channel
        self._timestep = timestep
        self._store_masks = store_masks
        self._start_time = start_time
        self._impute_strategy = impute_strategy

        # for normal value calculation
        self._data_path = data_path
        self._train_path = os.path.join(self._data_path, "train")
        self._remove_outliers = remove_outliers
        self.check_normal_values()

    def check_normal_values(self):
        if self._remove_outliers:
            file = os.path.join(self._data_path, "normal_values_nooutliers.json")
        else:
            file = os.path.join(self._data_path, "normal_values.json")
        if os.path.exists(file):
            with open(file) as f:
                normal_values = json.load(f)
            self._normal_values = normal_values
        else:
            self.get_normal_values()
        print("Loaded impute values.")

    def get_normal_values(self):
        train_path = os.path.join(self._train_path)
        timeseries = list(os.listdir(train_path))
        timeseries.remove("listfile.csv")
        if "train_listfile_balanced.csv" in timeseries:
            timeseries.remove("train_listfile_balanced.csv")

        dfs = []
        print("Calculating impute values.")
        for ts in tqdm(timeseries):
            dfs.append(pd.read_csv(os.path.join(train_path, ts)))
        combined = pd.concat(dfs)
        if "troponin-t" in combined.columns:
            combined = combined.drop("troponin-t", axis=1)
        if self._remove_outliers:
            for cols in combined:
                self._normal_values[cols] = [None,
                                             combined[cols].mean() - (3 * combined[cols].std()),
                                             combined[cols].mean() + (3 * combined[cols].std())]
                combined.loc[(combined[cols] - combined[cols].mean()).abs() >= 3 * combined[cols].std(), cols] = np.nan

        # means = pd.DataFrame(combined.mean())
        means = pd.DataFrame(combined.median())
        for index, row in means.iterrows():
            if index != "hours":
                if not self._is_categorical_channel[index]:
                    if self._remove_outliers:
                        self._normal_values[index][0] = round(row[0], 5)
                    else:
                        self._normal_values[index] = round(row[0], 5)

        for col in combined.columns:
            if col == "Ethnicity" or col == "Gender" or col == "vent":
                self._normal_values[col] = "0.0"
                continue
            if col != "hours":
                if self._is_categorical_channel[col]:
                    self._normal_values[col] = combined[col].mode().iloc[0]
        if self._remove_outliers:
            with open(os.path.join(self._data_path, "normal_values_nooutliers.json"), "w", encoding="utf-8") as f:
                json.dump(self._normal_values, f, ensure_ascii=False, indent=4)
        else:
            with open(os.path.join(self._data_path, "normal_values.json"), "w", encoding="utf-8") as f:
                json.dump(self._normal_values, f, ensure_ascii=False, indent=4)
        print("Finished calculating impute values.")

    def transform(self, X, header=None, end=None):
        if header is None:
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps

        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if end is None:
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time

        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        cur_len = 0
        begin_pos = [0 for i in range(N_channels)]
        end_pos = [0 for i in range(N_channels)]
        for i in range(N_channels):
            channel = self._id_to_channel[i]
            begin_pos[i] = cur_len
            if self._is_categorical_channel[channel]:
                end_pos[i] = begin_pos[i] + len(self._possible_values[channel])
            else:
                end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]

        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]

        def write(data, bin_id, channel, value, begin_pos):
            channel_id = self._channel_to_id[channel]
            value = float(value)
            if self._is_categorical_channel[channel]:
                try:
                    category_id = self._possible_values[channel].index(str(value))
                except Exception as e:
                    print(e)
                    data[bin_id, begin_pos[channel_id]] = np.nan
                    print("0 was inserted instead of string")
                    return
                N_values = len(self._possible_values[channel])
                one_hot = np.zeros((N_values,))
                one_hot[category_id] = 1
                for pos in range(N_values):
                    data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
            else:
                try:
                    data[bin_id, begin_pos[channel_id]] = float(value)
                    # if self._remove_outliers:
                    #     if float(value) > self._normal_values[channel][2] or float(value) < self._normal_values[channel][1]:
                    #         data[bin_id, begin_pos[channel_id]] = 0.0
                except ValueError:
                    data[bin_id, begin_pos[channel_id]] = np.nan
                    print("0 was inserted instead of string")

        for row in X:
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins

            for j in range(1, len(row)):
                if row[j] == "" or row[j] == " ":
                    continue
                channel = header[j]
                channel_id = self._channel_to_id[channel]

                mask[bin_id][channel_id] = 1

                write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

        # impute missing values

        if self._impute_strategy not in ['zero', 'normal_value', 'previous', 'next']:
            raise ValueError("impute strategy is invalid")

        if self._impute_strategy in ['normal_value', 'previous']:
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if self._impute_strategy == 'normal_value':
                        imputed_value = self._normal_values[channel]
                    if self._impute_strategy == 'previous':
                        if len(prev_values[channel_id]) == 0:
                            if self._remove_outliers and not self._is_categorical_channel[channel]:
                                imputed_value = self._normal_values[channel][0]
                            else:
                                imputed_value = self._normal_values[channel]
                        else:
                            imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        if self._impute_strategy == 'next':
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins-1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if len(prev_values[channel_id]) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        # create new header
        new_header = []
        for channel in self._id_to_channel:
            if self._is_categorical_channel[channel]:
                values = self._possible_values[channel]
                for value in values:
                    new_header.append(channel + "->" + value)
            else:
                new_header.append(channel)

        if self._store_masks:
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)

        new_header = ",".join(new_header)

        return (data, new_header)


class Normalizer:
    def __init__(self, fields=None):
        self._means = None
        self._stds = None
        self._fields = None
        if fields is not None:
            self._fields = [col for col in fields]

        self._sum_x = None
        self._sum_sq_x = None
        self._count = 0
        self._median = None
        self._percentile_25 = None
        self._percentile_75 = None

    def feed_data(self, x):
        x = np.array(x)
        x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
        self._count += x.shape[0]
        self._median = np.median(x, axis=0)
        self._percentile_25 = np.percentile(x, q=20, axis=0)
        self._percentile_75 = np.percentile(x, q=80, axis=0)

        if self._sum_x is None:
            self._sum_x = np.sum(x, axis=0)
            self._sum_sq_x = np.sum(x**2, axis=0)
        else:
            self._sum_x += np.sum(x, axis=0)
            self._sum_sq_x += np.sum(x**2, axis=0)

    def save_params(self, save_file_path):
        eps = 1e-7
        with open(save_file_path, "wb") as save_file:
            N = self._count
            self._means = 1.0 / N * self._sum_x
            self._stds = np.sqrt(1.0/(N - 1) * (self._sum_sq_x - 2.0 * self._sum_x * self._means + N * self._means**2))
            self._stds[self._stds < eps] = eps
            pickle.dump(obj={'means': self._means,
                             'stds': self._stds,
                             'median': self._median,
                             "percentile_75": self._percentile_75,
                             "percentile_25": self._percentile_25},
                        file=save_file,
                        protocol=2)

    def load_params(self, load_file_path):
        with open(load_file_path, "rb") as load_file:
            if platform.python_version()[0] == '2':
                dct = pickle.load(load_file)
            else:
                dct = pickle.load(load_file, encoding='latin1')
            self._means = dct['means']
            self._stds = dct['stds']
            self._median = dct['median']
            self._percentile_25 = dct['percentile_25']
            self._percentile_75 = dct['percentile_75']

    def transform(self, X):
        if self._fields is None:
            fields = range(X.shape[1])
        else:
            fields = self._fields
        ret = 1.0 * X
        for col in fields:
            # ret[:, col] = (X[:, col] - self._means[col]) / self._stds[col]
            ret[:, col] = (X[:, col] - self._median[col]) / (self._percentile_75[col] - self._percentile_25[col])
        return ret
