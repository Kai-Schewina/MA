from __future__ import absolute_import
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
from metrics import f1


def build_model(depth, units, dropout, rec_dropout, input_dim, kernel_reg):
    model = keras.Sequential()
    model.add(Input(shape=(None, input_dim)))
    for i in range(depth - 1):
        model.add(Bidirectional(LSTM(units=int(units/2), return_sequences=True, dropout=dropout, recurrent_dropout=rec_dropout)))
    model.add(LSTM(units=units, dropout=dropout, recurrent_dropout=rec_dropout))
    model.add(Dropout(rate=dropout))
    model.add(Dense(1, activation="sigmoid"))
    return model


def main(data, output_dir='.', dim=256, depth=1, epochs=20,
         load_state="", mode="train", batch_size=64, l2=0, l1=0, save_every=1, prefix="", dropout=0.0, rec_dropout=0.0,
         batch_norm=False, timestep=1.0, small_part=False, kernel_reg=0.0,
         lr=0.001, verbose=2, balance=False, balance_loss=False, balanced_sample=False):

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
    elif mode == "val_test":
        with open(data + "/val_raw.pkl", "rb") as f:
            test = pickle.load(f)
            input_dim = test[0].shape[2]
    else:
        raise ValueError("Wrong value for mode")

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
        early_stopping = EarlyStopping(patience=5)

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

        # Plot history: Accuracy
        plt.plot(history.history['val_auc_1'])
        plt.plot(history.history['auc_1'])
        plt.title('Validation auprc history')
        plt.ylabel('auprc value')
        plt.xlabel('No. epoch')
        plt.show()

    elif mode == 'test':
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

    elif mode == 'val_test':
        data = np.asarray(test[0]).astype('float32')
        labels = np.asarray(test[1]).astype('float32')

        predictions = model.predict(data, batch_size=batch_size, verbose=1)
        predictions = np.array(predictions)[:, 0]
        metrics.print_metrics_binary(labels, predictions)
        eval = model.evaluate(data, labels, batch_size=batch_size)
        print(eval)

    elif mode == "bootstrap_test":
        data = np.asarray(test["data"][0]).astype('float32')
        labels = np.asarray(test["data"][1]).astype('float32')

        # Execute Random Sampling
        for i in tqdm(range(10000)):
            data_sample, labels_sample = resample(data, labels, replace=True, random_state=np.random.RandomState(42+i))

            eval = model.evaluate(data_sample, labels_sample, batch_size=batch_size, verbose=0)
            if i == 0:
                eval_df = pd.DataFrame([eval])
                eval_df.columns = ["loss", "auroc", "auprc", "accuracy", "precision", "recall"]
            else:
                eval_df.loc[i] = eval

        eval_df["f1"] = 2 * (eval_df["precision"] * eval_df["recall"]) / (eval_df["precision"] + eval_df["recall"])
        os.makedirs("./cv_test/", exist_ok=True)
        eval_df.to_csv("./cv_test/" + model.final_name + ".csv", index=False)


if __name__ == "__main__":
    path = "../data/in-hospital-mortality_v7/"
    main(data=path, mode="bootstrap_test", dropout=0.4, rec_dropout=0.3, depth=3, batch_size=32, dim=64, epochs=4,
         balanced_sample=True, balance=False, lr=0.01,
         load_state="./keras_states/k_lstm.n64.d0.4.rd0.3.dep.lr0.01_bs32_ts1.0.epoch4.test0.2864157259464264.state")
