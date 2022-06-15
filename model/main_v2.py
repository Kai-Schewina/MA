from __future__ import absolute_import
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re
import metrics
import pickle
import numpy as np
import utils
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow import keras


def build_model(depth, units, dropout, input_dim):
    model = keras.Sequential()
    model.add(layers.Input(shape=(None, input_dim)))
    model.add(layers.Masking())
    for i in range(depth - 1):
        model.add(layers.Bidirectional(layers.LSTM(units=int(units/2), return_sequences=True,
                                                   dropout=dropout, recurrent_dropout=dropout)))
    model.add(layers.LSTM(units=units))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def main(data, output_dir='.', dim=256, depth=1, epochs=20,
         load_state="", mode="train", batch_size=64, l2=0, l1=0, save_every=1, prefix="", dropout=0.0, rec_dropout=0.0,
         batch_norm=False, timestep=1.0, small_part=False, optimizer="adam",
         lr=0.001, beta_1=0.9, verbose=2, balance=False, balance_loss=False, balanced_sample=False):

    def weighted_bincrossentropy(y_true, y_pred, weight_zero=0.125, weight_one=0.875):
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
        else:
            with open(data + "/train_raw.pkl", "rb") as f:
                train_raw = pickle.load(f)
                input_dim = train_raw[0].shape[2]
        with open(data + "/val_raw.pkl", "rb") as f:
            val_raw = pickle.load(f)
    elif mode == "test":
        with open(data + "/test.pkl", "rb") as f:
            test = pickle.load(f)
            input_dim = test["data"][0].shape[2]
    else:
        raise ValueError("Wrong value for mode")

    # Setup
    if small_part:
        save_every = 2 ** 30

    optimizer_config = {'class_name': optimizer,
                        'config': {'lr': lr,
                                   'beta_1': beta_1}}
    if balance_loss:
        loss = weighted_bincrossentropy
    else:
        loss = 'binary_crossentropy'

    # Build the model
    model = build_model(depth, dim, dropout, input_dim)

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
    model.compile(optimizer=optimizer_config,
                  loss=loss)
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
            # class_weight = {0: weight_for_0, 1: weight_for_1}
            print('Weight for class 0: {:.2f}'.format(weight_for_0))
            print('Weight for class 1: {:.2f}'.format(weight_for_1))
            # class_weight = compute_class_weight("balanced", classes=np.unique(train_raw[1]), y=train_raw[1])

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
                  validation_data=val_raw,
                  epochs=n_trained_chunks + epochs,
                  initial_epoch=n_trained_chunks,
                  callbacks=[metrics_callback, saver, csv_logger],
                  shuffle=True,
                  verbose=verbose,
                  batch_size=batch_size,
                  class_weight=class_weight)

    if mode == 'test':
        data = test["data"][0]
        labels = test["data"][1]
        names = test["names"]

        predictions = model.predict(data, batch_size=batch_size, verbose=1)
        predictions = np.array(predictions)[:, 0]
        metrics.print_metrics_binary(labels, predictions)

        path = os.path.join(output_dir, "test_predictions", os.path.basename(load_state)) + ".csv"
        utils.save_results(names, predictions, labels, path)


if __name__ == "__main__":
    path = "../data/in-hospital-mortality_v4/"
    main(data=path, mode="train", dropout=0.3, depth=2, batch_size=8, dim=16, epochs=10, lr=0.001, balanced_sample=True)

