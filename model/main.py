from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import imp
import re
from sklearn.utils import class_weight
import pandas as pd

from build.readers import InHospitalMortalityReader
from preprocessing import Discretizer, Normalizer
import metrics
import keras_utils
import utils
from lstm import Network

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers.core import Dense
from keras.models import Model


def main(target_repl_coef=0.0, data='../data/in-hospital-mortality/', output_dir='.', dim=256, depth=1, epochs=3,
         load_state="", mode="train", batch_size=64, l2=0, l1=0, save_every=1, prefix="", dropout=0.0, rec_dropout=0.0,
         batch_norm=False, timestep=1.0, imputation="previous", small_part=True, whole_data=False, optimizer="adam",
         lr=0.001, beta_1=0.9, verbose=2, normalizer_state="./ihm_ts_1.00_impute_previous_start_zero_masks_True_n_20918.normalizer",
         transfer=False, balance=False, balance_loss=False):
    
    if small_part:
        save_every = 2 ** 30
    
    target_repl = (target_repl_coef > 0.0 and mode == 'train')
    
    # Build readers, discretizers, normalizers
    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data, 'train'),
                                             listfile=os.path.join(data, 'train_listfile.csv'))
    
    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data, 'train'),
                                           listfile=os.path.join(data, 'val_listfile.csv'))
    
    discretizer = Discretizer(data_path=data, timestep=float(timestep),
                              store_masks=True,
                              impute_strategy='previous',
                              start_time='zero')
    
    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
    input_dim = len(discretizer_header)

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer.load_params(normalizer_state)
    
    # Build the model
    model = Network(dim=dim, batch_norm=batch_norm, dropout=dropout, rec_dropout=rec_dropout, task='ihm',
                    target_repl=target_repl, deep_supervision=False, num_classes=1, depth=depth,
                    input_dim=input_dim)
    suffix = ".bs{}{}{}.ts{}{}".format(batch_size,
                                       ".L1{}".format(l1) if l1 > 0 else "",
                                       ".L2{}".format(l2) if l2 > 0 else "",
                                       timestep,
                                       ".trc{}".format(target_repl_coef) if target_repl_coef > 0 else "")
    model.final_name = prefix + model.say_name() + suffix
    print("==> model.final_name:", model.final_name)
    
    # Compile the model
    print("==> compiling the model")
    optimizer_config = {'class_name': optimizer,
                        'config': {'lr': lr,
                                   'beta_1': beta_1}}

    loss = 'binary_crossentropy'

    if balance_loss:
        train_file = pd.read_csv(os.path.join(data, 'train_listfile.csv'))
        train_trues = np.array(train_file["y_true"])
        loss_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_trues), y=train_trues)
    else:
        loss_weights = None
    
    model.compile(optimizer=optimizer_config,
                  loss=loss,
                  loss_weights=loss_weights)
    model.summary()
    
    # Load model weights
    n_trained_chunks = 0
    if load_state != "":
        model.load_weights(load_state)
        n_trained_chunks = int(re.match(".*epoch([0-9]+).*", load_state).group(1))
    
    # Read data
    train_raw = utils.load_data(train_reader, discretizer, normalizer, small_part)
    val_raw = utils.load_data(val_reader, discretizer, normalizer, small_part)
    
    if target_repl:
        T = train_raw[0][0].shape[0]
    
        def extend_labels(data):
            data = list(data)
            labels = np.array(data[1])  # (B,)
            data[1] = [labels, None]
            data[1][1] = np.expand_dims(labels, axis=-1).repeat(T, axis=1)  # (B, T)
            data[1][1] = np.expand_dims(data[1][1], axis=-1)  # (B, T, 1)
            return data
    
        train_raw = extend_labels(train_raw)
        val_raw = extend_labels(val_raw)
    
    if mode == 'train':
    
        # Prepare training
        path = os.path.join(output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')
    
        metrics_callback = keras_utils.InHospitalMortalityMetrics(train_data=train_raw,
                                                                  val_data=val_raw,
                                                                  target_repl=(target_repl_coef > 0),
                                                                  batch_size=batch_size,
                                                                  verbose=verbose)
        # make sure save directory exists
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        saver = ModelCheckpoint(path, verbose=1, period=save_every)
    
        keras_logs = os.path.join(output_dir, 'keras_logs')
        if not os.path.exists(keras_logs):
            os.makedirs(keras_logs)
        csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                               append=True, separator=';')
    
        print("==> training")
        # Balance class weights
        if balance:
            train_file = pd.read_csv(os.path.join(data, 'train_listfile.csv'))
            train_trues = np.array(train_file["y_true"])
            class_weights = {
                0: 1.0,
                1: train_trues.sum() / (len(train_trues) - train_trues.sum())
            }
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_trues), y=train_trues)
        else:
            class_weights = None

        model.fit(x=train_raw[0],
                  y=train_raw[1],
                  validation_data=val_raw,
                  epochs=n_trained_chunks + epochs,
                  initial_epoch=n_trained_chunks,
                  callbacks=[metrics_callback, saver, csv_logger],
                  shuffle=True,
                  verbose=verbose,
                  batch_size=batch_size,
                  class_weight=class_weights)
    
    elif mode == 'test':
    
        # ensure that the code uses test_reader
        del train_reader
        del val_reader
        del train_raw
        del val_raw
    
        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data, 'test'),
                                                listfile=os.path.join(data, 'test_listfile.csv'),
                                                period_length=48.0)
        ret = utils.load_data(test_reader, discretizer, normalizer, small_part,
                              return_names=True)
    
        data = ret["data"][0]
        labels = ret["data"][1]
        names = ret["names"]
    
        predictions = model.predict(data, batch_size=batch_size, verbose=1)
        predictions = np.array(predictions)[:, 0]
        metrics.print_metrics_binary(labels, predictions)
    
        path = os.path.join(output_dir, "test_predictions", os.path.basename(load_state)) + ".csv"
        utils.save_results(names, predictions, labels, path)
    else:
        raise ValueError("Wrong value for mode")


if __name__ == "__main__":
    full_data_dir = "../../data/in-hospital-mortality_v2"
    main(small_part=False, whole_data=True, dim=16, timestep=1.0, depth=2, dropout=0.0,
         batch_size=32, mode="train", load_state="",
         transfer=False, data=full_data_dir, balance=False)

