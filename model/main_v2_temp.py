from __future__ import absolute_import
from __future__ import print_function

import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

from numpy.random import seed
seed(42)
random.seed(42)
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.random.set_seed(42)

import re
import metrics
import pickle
import numpy as np
import utils
import pandas as pd
from sklearn.utils import resample
from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras.layers import LSTM, Input, Dropout, Dense, Masking, Bidirectional
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Precision, Recall
import matplotlib.pyplot as plt


def build_model(depth, units, dropout, rec_dropout, input_dim, kernel_reg):
    model = keras.Sequential()
    model.add(Input(shape=(None, input_dim)))
    for i in range(depth - 1):
        model.add(Bidirectional(LSTM(units=int(units/2), return_sequences=True, dropout=dropout, recurrent_dropout=rec_dropout)))
    model.add(LSTM(units=units, dropout=dropout, recurrent_dropout=rec_dropout))
    model.add(Dropout(rate=dropout, seed=42))
    model.add(Dense(1, activation="sigmoid"))
    return model


def main(data, output_dir='.', dim=256, depth=1, epochs=20,
         load_state="", mode="train", batch_size=64, l2=0, l1=0, save_every=1, prefix="", dropout=0.0, rec_dropout=0.0,
         batch_norm=False, timestep=1.0, small_part=False, kernel_reg=0.0,
         lr=0.001, verbose=2, balance=False, balance_loss=False, balanced_sample=False, percentage=None):

    # Load Data
    if mode == "train":
        if balanced_sample:
            with open(data + "/train_raw_balanced.pkl", "rb") as f:
                train_raw = pickle.load(f)
                train_raw = (np.asarray(train_raw["data"][0]).astype('float32'), np.asarray(train_raw["data"][1]).astype('float32'))
                input_dim = train_raw[0].shape[2]
        else:
            with open(data + "/train_raw.pkl", "rb") as f:
                train_raw = pickle.load(f)
                train_raw = (np.asarray(train_raw["data"][0]).astype('float32'), np.asarray(train_raw["data"][1]).astype('float32'))
                input_dim = train_raw[0].shape[2]
        with open(data + "/val_raw.pkl", "rb") as f:
            val_raw = pickle.load(f)
            val_raw = (np.asarray(val_raw["data"][0]).astype('float32'), np.asarray(val_raw["data"][1]).astype('float32'))
    elif mode == "test" or mode == "bootstrap_test":
        with open(data + "/test.pkl", "rb") as f:
            test = pickle.load(f)
            input_dim = test["data"][0].shape[2]
    else:
        raise ValueError("Wrong value for mode")

    # Potential Subsampling
    if percentage is not None:
        np.random.seed(42)
        idx = np.random.choice(np.arange(len(train_raw[0])), int(len(train_raw[0]) * percentage), replace=False)
        train_raw = (train_raw[0][idx], train_raw[1][idx])
    # Setup
    if small_part:
        save_every = 2 ** 30

    loss = 'binary_crossentropy'

    # Build the model
    model = build_model(depth, dim, dropout, rec_dropout, input_dim, kernel_reg)

    suffix = "_bs{}{}{}_ts{}".format(batch_size,
                                     "_L1{}".format(l1) if l1 > 0 else "",
                                     "_L2{}".format(l2) if l2 > 0 else "",
                                     timestep)
    say_name = "{}.n{}{}{}{}.dep{}".format('k_lstm',
                                           dim,
                                           ".bn" if batch_norm else "",
                                           ".d{}".format(dropout) if dropout > 0 else "",
                                           ".rd{}".format(rec_dropout) if rec_dropout > 0 else "",
                                           ".lr{}".format(lr) if lr > 0 else "",
                                           depth)
    model.final_name = prefix + say_name + suffix

    opt = Adam(learning_rate=lr)
    my_metrics = [AUC(), AUC(curve="PR"), BinaryAccuracy(), Precision(), Recall()]
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=my_metrics)
    model.summary()

    # Load model weights
    n_trained_chunks = 0
    if load_state != "":
        model.load_weights(load_state).expect_partial()
        n_trained_chunks = int(re.match(".*epoch([0-9]+).*", load_state).group(1))

    if mode == 'train':
        print("Initial Performance:")
        eval = model.evaluate(val_raw[0], val_raw[1], batch_size=batch_size)
        print(eval)
        # Prepare training
        if percentage is not None:
            path = os.path.join(output_dir, 'keras_states/' + model.final_name + "_perc_" + str(percentage))
            saver = ModelCheckpoint(path, verbose=verbose, period=save_every, save_best_only=True)
        else:
            path = os.path.join(output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')
            saver = ModelCheckpoint(path, verbose=verbose, period=save_every)
        os.makedirs(path, exist_ok=True)


        keras_logs = os.path.join(output_dir, 'keras_logs')
        os.makedirs(keras_logs, exist_ok=True)
        csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                               append=True, separator=';')

        metrics_callback = metrics.InHospitalMortalityMetrics(train_data=train_raw,
                                                              val_data=val_raw,
                                                              batch_size=batch_size,
                                                              verbose=verbose)
        print("==> training")
        history = model.fit(x=train_raw[0],
                  y=train_raw[1],
                  validation_data=(val_raw[0], val_raw[1]),
                  epochs=n_trained_chunks + epochs,
                  initial_epoch=n_trained_chunks,
                  callbacks=[metrics_callback, saver, csv_logger],
                  shuffle=True,
                  verbose=verbose,
                  batch_size=batch_size)
        # Visualize history
        # Plot history: Loss
        plt.plot(history.history['val_loss'])
        plt.plot(history.history['loss'])
        plt.title('Validation loss history')
        plt.ylabel('Loss value')
        plt.xlabel('No. epoch')
        plt.show()

        if percentage is not None:
            return path
        else:
            return path.replace("{epoch}", str(epochs)).replace("{val_loss}", str(history.history['val_loss'][epochs-1]))

    elif mode == 'test':
        data = np.asarray(test["data"][0]).astype('float32')
        labels = np.asarray(test["data"][1]).astype('float32')
        names = test["names"]

        predictions = model.predict(data, batch_size=batch_size, verbose=verbose)
        predictions = np.array(predictions)[:, 0]
        metrics.print_metrics_binary(labels, predictions)

        path = os.path.join(output_dir, "test_predictions", os.path.basename(load_state)) + ".csv"
        utils.save_results(names, predictions, labels, path)

        eval = model.evaluate(data, labels, batch_size=batch_size)
        print(eval)

    elif mode == "bootstrap_test":
        data = np.asarray(test["data"][0]).astype('float32')
        labels = np.asarray(test["data"][1]).astype('float32')

        # Execute Random Sampling
        for i in tqdm(range(10000)):
            data_sample, labels_sample = resample(data, labels, replace=True, random_state=np.random.RandomState(42+i))

            eval = model.evaluate(data_sample, labels_sample, batch_size=batch_size, verbose=verbose)
            if i == 0:
                eval_df = pd.DataFrame([eval])
                eval_df.columns = ["loss", "auroc", "auprc", "accuracy", "precision", "recall"]
            else:
                eval_df.loc[i] = eval
        eval_df["f1"] = 2 * (eval_df["precision"] * eval_df["recall"]) / (eval_df["precision"] + eval_df["recall"])
        os.makedirs("./cv_test/", exist_ok=True)
        eval_df.to_csv("./cv_test/" + model.final_name + "balanced_sample_" + str(balanced_sample) + ".csv", index=False)


if __name__ == "__main__":
    tf.random.set_seed(42)
    path = "../data/in-hospital-mortality_v7/"
    # Balanced!!!
    # Returnt 4 Epochen als optimal
    main(data=path, mode="bootstrap_test", dropout=0.4, rec_dropout=0.3, depth=3, batch_size=32, dim=64, epochs=7,
         balanced_sample=True, lr=0.01, percentage=None, verbose=0,
         load_state="")
