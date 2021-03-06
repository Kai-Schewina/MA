from __future__ import absolute_import
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re
import metrics
import pickle
import numpy as np
import utils

from sklearn.utils import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Input, Dropout, Dense, Masking, Bidirectional
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Precision, Recall


def build_model(depth, units, dropout, input_dim, kernel_reg):
    model = keras.Sequential()
    model.add(Input(shape=(None, input_dim)))
    for i in range(depth - 1):
        model.add(Bidirectional(LSTM(units=int(units/2), return_sequences=True, dropout=dropout, recurrent_dropout=dropout, kernel_regularizer=L1L2(l1=kernel_reg, l2=kernel_reg),
                       recurrent_regularizer=L1L2(l1=0.00, l2=0.00), bias_regularizer=L1L2(l1=0.00, l2=0.00),
                       activity_regularizer=L1L2(l1=0.0, l2=0.0))))
    model.add(LSTM(units=units, dropout=dropout, recurrent_dropout=dropout, kernel_regularizer=L1L2(l1=kernel_reg, l2=kernel_reg),
                   recurrent_regularizer=L1L2(l1=0.00, l2=0.00), bias_regularizer=L1L2(l1=0.00, l2=0.00),
                   activity_regularizer=L1L2(l1=0.0, l2=0.0)))
    model.add(Dropout(rate=dropout))
    model.add(Dense(1, activation="sigmoid"))
    return model


def main(data, output_dir='.', dim=256, depth=1, epochs=20,
         load_state="", mode="train", batch_size=64, l2=0, l1=0, save_every=1, prefix="", dropout=0.0, rec_dropout=0.0,
         batch_norm=False, timestep=1.0, small_part=False, optimizer="adam", kernel_reg=0.0,
         lr=0.001, beta_1=0.9, verbose=2, balance=False, balance_loss=False, balanced_sample=False):

    def weighted_bincrossentropy(y_true, y_pred, weight_zero=0.28, weight_one=0.72):
        bin_crossentropy = keras.backend.binary_crossentropy(y_true, y_pred)

        weights = y_true * weight_one + (1. - y_true) * weight_zero
        weighted_bin_crossentropy = weights * bin_crossentropy

        return keras.backend.mean(weighted_bin_crossentropy)

    # Load Data
    if mode == "train":
        if balanced_sample:
            with open(data + "/train_raw_balanced.pkl", "rb") as f:
                train_raw = pickle.load(f)
                input_dim = train_raw[0].shape[2]
                train_raw = (np.asarray(train_raw[0]).astype('float32'), np.asarray(train_raw[1]).astype('float32'))
        else:
            with open(data + "/train_raw.pkl", "rb") as f:
                train_raw = pickle.load(f)
                train_raw = (np.asarray(train_raw["data"][0]).astype('float32'), np.asarray(train_raw["data"][1]).astype('float32'))
                input_dim = train_raw[0].shape[2]
        with open(data + "/val_raw.pkl", "rb") as f:
            val_raw = pickle.load(f)
            val_raw = (np.asarray(val_raw["data"][0]).astype('float32'), np.asarray(val_raw["data"][1]).astype('float32'))
    elif mode == "test":
        with open(data + "/test.pkl", "rb") as f:
            test = pickle.load(f)
            input_dim = test["data"][0].shape[2]
    elif mode == "val_test":
        with open(data + "/val_raw.pkl", "rb") as f:
            test = pickle.load(f)
            input_dim = test[0].shape[2]
    else:
        raise ValueError("Wrong value for mode")

    # Setup
    if small_part:
        save_every = 2 ** 30

    if balance_loss:
        loss = weighted_bincrossentropy
    else:
        loss = 'binary_crossentropy'

    # Build the model
    model = build_model(depth, dim, dropout, input_dim, kernel_reg)

    suffix = "_bs{}{}{}_ts{}".format(batch_size,
                                     "_L1{}".format(l1) if l1 > 0 else "",
                                     "_L2{}".format(l2) if l2 > 0 else "",
                                     timestep)
    say_name = "{}.n{}{}{}{}.dep{}".format('k_lstm',
                                           dim,
                                           ".bn" if batch_norm else "",
                                           ".d{}".format(dropout) if dropout > 0 else "",
                                           ".rd{}".format(rec_dropout) if rec_dropout > 0 else "",
                                           depth)
    model.final_name = prefix + say_name + suffix
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss=loss,
                  metrics=[AUC(), AUC(curve="PR"), BinaryAccuracy(), Precision(), Recall()])
    model.summary()

    # Load model weights
    n_trained_chunks = 0
    if load_state != "":
        model.load_weights(load_state)
        n_trained_chunks = int(re.match(".*epoch([0-9]+).*", load_state).group(1))

    if mode == 'train':
        # Balancing
        class_weight = None
        if balance:
            weight_for_0 = (1 / len(train_raw[1])) * (len(train_raw[1]) / 2.0)
            weight_for_1 = (1 / sum(train_raw[1])) * (len(train_raw[1]) / 2.0)
            class_weight = {0: weight_for_0, 1: weight_for_1}
            print('Weight for class 0: {:.2f}'.format(weight_for_0))
            print('Weight for class 1: {:.2f}'.format(weight_for_1))

        # Prepare training
        path = os.path.join(output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')
        os.makedirs(path, exist_ok=True)
        saver = ModelCheckpoint(path, verbose=1, period=save_every)
        keras_logs = os.path.join(output_dir, 'keras_logs')
        os.makedirs(keras_logs, exist_ok=True)
        csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                               append=True, separator=';')

        metrics_callback = metrics.InHospitalMortalityMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              batch_size=batch_size,
                                                              verbose=verbose)

        print("==> training")
        model.fit(x=train_raw[0],
                  y=train_raw[1],
                  validation_data=(val_raw[0], val_raw[1]),
                  # validation_split=0.2,
                  epochs=n_trained_chunks + epochs,
                  initial_epoch=n_trained_chunks,
                  callbacks=[metrics_callback, saver, csv_logger],
                  shuffle=True,
                  verbose=verbose,
                  batch_size=batch_size,
                  class_weight=class_weight)

    if mode == 'test':
        data = np.asarray(test["data"][0]).astype('float32')
        labels = np.asarray(test["data"][1]).astype('float32')
        names = test["names"]

        predictions = model.predict(data, batch_size=batch_size, verbose=1)
        predictions = np.array(predictions)[:, 0]
        metrics.print_metrics_binary(labels, predictions)

        path = os.path.join(output_dir, "test_predictions", os.path.basename(load_state)) + ".csv"
        utils.save_results(names, predictions, labels, path)

        eval = model.evaluate(data, labels, batch_size=batch_size)
        print(eval)

    if mode == 'val_test':
        data = np.asarray(test[0]).astype('float32')
        labels = np.asarray(test[1]).astype('float32')

        predictions = model.predict(data, batch_size=batch_size, verbose=1)
        predictions = np.array(predictions)[:, 0]
        metrics.print_metrics_binary(labels, predictions)
        eval = model.evaluate(data, labels, batch_size=batch_size)
        print(eval)


if __name__ == "__main__":
    path = "../data/in-hospital-mortality_v6_5/"
    main(data=path, mode="test", dropout=0.3, depth=2, batch_size=8, dim=16, epochs=28, balanced_sample=False, balance=False, kernel_reg=0.01,
         load_state=".\keras_states\k_lstm.n16.d0.3.dep2_bs8_ts1.0.epoch9.test0.2556713819503784.state")

