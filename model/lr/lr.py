from __future__ import absolute_import
from __future__ import print_function

from build.readers import InHospitalMortalityReader
import utils
from metrics import print_metrics_binary
from utils import save_results
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from feature_extractor import extract_features
from sklearn.utils import resample
from sklearn.metrics import log_loss

import pickle
import os
import numpy as np
import json
from tqdm import tqdm
import pandas as pd


def convert_to_dict(data, header, channel_info):
    """ convert data from readers output in to array of arrays format """
    ret = [[] for i in range(data.shape[1] - 1)]
    for i in range(1, data.shape[1]):
        ret[i-1] = [(t, x) for (t, x) in zip(data[:, 0], data[:, i]) if x != ""]
        channel = header[i]
        if len(channel_info[channel]['possible_values']) != 0:
            try:
                ret[i-1] = list(map(lambda x: (x[0], channel_info[channel]['values'][x[1]]), ret[i-1]))
            except Exception as e:
                print(channel)
                raise e
        ret[i-1] = list(map(lambda x: (float(x[0]), float(x[1])), ret[i-1]))
    return ret


def extract_features_from_rawdata(chunk, header, period, features):
    with open("channel_info.json") as channel_info_file:
        channel_info = json.loads(channel_info_file.read())
    data = [convert_to_dict(X, header, channel_info) for X in chunk]
    return extract_features(data, period, features)


def read_and_extract_features(reader, period, features):
    ret = utils.read_chunk(reader, reader.get_number_of_examples())
    X = extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'])


def main(data, period, features, l2, C, output_dir, ards_path, mode="full_data", bootstrap=False):
    if not os.path.exists("data.pkl") or not os.path.exists("ards_data.pkl"):
        train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data, 'train'),
                                                 listfile=os.path.join(data, 'train_listfile.csv'),
                                                 period_length=48.0)

        val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data, 'val'),
                                               listfile=os.path.join(data, 'val_listfile.csv'),
                                               period_length=48.0)

        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data, 'test'),
                                                listfile=os.path.join(data, 'test_listfile.csv'),
                                                period_length=48.0)

        print('Reading data and extracting features ...')
        (train_X, train_y, train_names) = read_and_extract_features(train_reader, period, features)
        print('  train data shape = {}'.format(train_X.shape))
        (val_X, val_y, val_names) = read_and_extract_features(val_reader, period, features)
        print('  validation data shape = {}'.format(val_X.shape))
        (test_X, test_y, test_names) = read_and_extract_features(test_reader, period, features)
        print('  test data shape = {}'.format(test_X.shape))

        print('Imputing missing values ...')
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0, copy=True)
        imputer.fit(train_X)
        train_X = np.array(imputer.transform(train_X), dtype=np.float32)
        val_X = np.array(imputer.transform(val_X), dtype=np.float32)
        test_X = np.array(imputer.transform(test_X), dtype=np.float32)

        print('Normalizing the data to have zero mean and unit variance ...')
        scaler = StandardScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        val_X = scaler.transform(val_X)
        test_X = scaler.transform(test_X)

        with open("data.pkl", "wb") as f:
            pickle.dump([train_X, train_y, train_names, val_X, val_y, val_names, test_X, test_y, test_names], f)
        print("Data saved to data.pkl")

        train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(ards_path, 'train'),
                                                 listfile=os.path.join(ards_path, 'train_listfile.csv'),
                                                 period_length=48.0)

        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(ards_path, 'test'),
                                                listfile=os.path.join(ards_path, 'test_listfile.csv'),
                                                period_length=48.0)
        print('Reading data and extracting features for ARDS data...')
        (train_ards_X, train_ards_y, train_ards_names) = read_and_extract_features(train_reader, period, features)
        (test_ards_X, test_ards_y, test_ards_names) = read_and_extract_features(test_reader, period, features)

        train_ards_X = np.array(imputer.transform(train_ards_X), dtype=np.float32)
        test_ards_X = np.array(imputer.transform(test_ards_X), dtype=np.float32)

        train_ards_X = scaler.transform(train_ards_X)
        test_ards_X = scaler.transform(test_ards_X)

        with open("ards_data.pkl", "wb") as f:
            pickle.dump([train_ards_X, train_ards_y, train_ards_names, test_ards_X, test_ards_y, test_ards_names], f)
        print("ARDS Data saved to data.pkl")
    else:
        with open("data.pkl", "rb") as f:
            train_X, train_y, train_names, val_X, val_y, val_names, test_X, test_y, test_names = pickle.load(f)
        print("Data loaded from data.pkl")

        with open("ards_data.pkl", "rb") as f:
            train_ards_X, train_ards_y, train_ards_names, test_ards_X, test_ards_y, test_ards_names = pickle.load(f)
        print("ARDS Data loaded.")

    if mode == "full_data":
        penalty = ('l2' if l2 else 'l1')
        file_name = '{}.{}.{}.C{}'.format(period, features, penalty, C)

        logreg = LogisticRegression(penalty=penalty, C=C, random_state=42)
        logreg.fit(train_X, train_y)

        result_dir = os.path.join(output_dir, 'results')
        os.makedirs(result_dir, exist_ok=True)

        with open(os.path.join(result_dir, 'train_{}.json'.format(file_name)), 'w') as res_file:
            ret = print_metrics_binary(train_y, logreg.predict_proba(train_X))
            ret = {k: float(v) for k, v in ret.items()}
            json.dump(ret, res_file)

        with open(os.path.join(result_dir, 'val_{}.json'.format(file_name)), 'w') as res_file:
            ret = print_metrics_binary(val_y, logreg.predict_proba(val_X))
            ret = {k: float(v) for k, v in ret.items()}
            json.dump(ret, res_file)

        prediction = logreg.predict_proba(test_X)[:, 1]

        with open(os.path.join(result_dir, 'test_{}.json'.format(file_name)), 'w') as res_file:
            ret = print_metrics_binary(test_y, prediction)
            ret = {k: float(v) for k, v in ret.items()}
            json.dump(ret, res_file)

        save_results(test_names, prediction, test_y,
                     os.path.join(output_dir, 'predictions', file_name + '.csv'))

    elif mode == "transfer":
        penalty = ('l2' if l2 else 'l1')
        file_name = '{}.{}.{}.C{}'.format(period, features, penalty, C)

        logreg = LogisticRegression(penalty=penalty, C=C, random_state=42)
        logreg.fit(train_X, train_y)

        result_dir = os.path.join(output_dir, 'results')
        os.makedirs(result_dir, exist_ok=True)
        if not bootstrap:
            with open(os.path.join(result_dir, 'transfer_train_{}.json'.format(file_name)), 'w') as res_file:
                ret = print_metrics_binary(train_ards_y, logreg.predict_proba(train_ards_X))
                ret = {k: float(v) for k, v in ret.items()}
                json.dump(ret, res_file)

            with open(os.path.join(result_dir, 'transfer_test_{}.json'.format(file_name)), 'w') as res_file:
                ret = print_metrics_binary(test_ards_y, logreg.predict_proba(test_ards_X))
                ret = {k: float(v) for k, v in ret.items()}
                json.dump(ret, res_file)
        else:
            # Execute Random Sampling
            for i in tqdm(range(1000)):
                data_sample, labels_sample = resample(test_ards_X, test_ards_y, replace=True,
                                                      random_state=np.random.RandomState(42 + i))
                preds = logreg.predict_proba(data_sample)
                results = print_metrics_binary(labels_sample, preds, verbose=0)
                results = [log_loss(labels_sample, preds), results["auroc"], results["auprc"], results["acc"], results["prec1"], results["rec1"], results["f1"]]
                if i == 0:
                    eval_df = pd.DataFrame([results])
                    eval_df.columns = ["loss", "auroc", "auprc", "acc", "prec1", "rec1", "f1"]
                else:
                    eval_df.loc[i] = results
            os.makedirs("./cv_test/", exist_ok=True)
            eval_df.to_csv("./cv_test/logit.csv", index=False)
    else:
        print("Wrong value for mode.")
        raise ValueError


if __name__ == '__main__':
    main(data="../../data/in-hospital-mortality_v7", period="first2days", mode="transfer",
         features="all", l2=True, C=0.001, output_dir=".", ards_path="../../data/ards_ihm_v2",
         bootstrap=False)
