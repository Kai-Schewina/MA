import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Input, Dropout, Dense, Bidirectional
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import pickle
import re
import os
from model import metrics
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Precision, Recall
from datetime import datetime
import random
from tqdm import tqdm
from sklearn.utils import resample


def build_model(depth, units, dropout, recurrent_droput, input_dim, kernel_reg):
    model = keras.Sequential()
    model.add(Input(shape=(None, input_dim)))
    for i in range(depth - 1):
        model.add(Bidirectional(LSTM(units=int(units/2), return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_droput)))
    model.add(LSTM(units=units, dropout=dropout, recurrent_dropout=recurrent_droput))
    model.add(Dropout(rate=dropout))
    model.add(Dense(1, activation="sigmoid"))
    return model


def plot(history):
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


def fine_tune(mode="train", depth=2, units=64, dropout=0.3, recurrent_droput=0.3, kernel_reg=0.00, load_state="", save_every=1, batch_size=16,
              lr=0.001, balance=False, epochs=10, freeze_layers=None, pop_layers=None, n_splits=3, verbose=2, add_dense=False):
    with open("../data/ards_ihm_v2/train_raw.pkl", "rb") as f:
        train_raw = pickle.load(f)
        train_raw = (np.asarray(train_raw["data"][0]).astype('float32'), np.asarray(train_raw["data"][1]).astype('float32'))

    output_dir = "."

    if balance:
        weight_for_0 = (1 / len(train_raw[1])) * (len(train_raw[1]) / 2.0)
        weight_for_1 = (1 / sum(train_raw[1])) * (len(train_raw[1]) / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
    else:
        class_weight = None

    if mode == "train":
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        inputs = train_raw[0]
        targets = np.asarray(train_raw[1])
        splits_history = []

        for train, val in kfold.split(inputs, targets):
            # model = build_model(depth, units, dropout, recurrent_droput, 70, 0.00)
            # model.load_weights(load_state).expect_partial()
            model = keras.models.load_model(load_state)
            if freeze_layers:
                for layer in model.layers[:-2]:
                    layer.trainable = False

            if pop_layers:
                new_model = tf.keras.models.Sequential(model.layers[:-2])
                if add_dense:
                    new_model.add(Dense(16, activation="relu"))
                new_model.add(Dense(1, activation="sigmoid"))
                for layer in new_model.layers:
                    print(layer.trainable)
                model = new_model

            # Prepare training
            model.final_name = "bla"
            path = os.path.join(output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')
            os.makedirs(path, exist_ok=True)
            saver = ModelCheckpoint(path, verbose=verbose, period=save_every)
            keras_logs = os.path.join(output_dir, 'keras_logs')
            os.makedirs(keras_logs, exist_ok=True)
            csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                                   append=True, separator=';')

            model.compile(optimizer=Adam(learning_rate=lr),
                          loss="binary_crossentropy",
                          metrics=[AUC(), AUC(curve="PR"), BinaryAccuracy(), Precision(), Recall()])

            # metrics_callback = metrics.InHospitalMortalityMetrics(train_data=(inputs[train], targets[train]),
            #                                                   val_data=(inputs[val], targets[val]),
            #                                                   batch_size=batch_size,
            #                                                   verbose=verbose)
            history = model.fit(x=inputs[train],
                      y=targets[train],
                      validation_data=(inputs[val], targets[val]),
                      epochs=epochs,
                      shuffle=True,
                      callbacks=[saver, csv_logger],
                      verbose=verbose,
                      batch_size=batch_size,
                      class_weight=class_weight)
            model.summary()
            splits_history.append(history)

        final_performances = pd.DataFrame(splits_history[0].history["val_loss"])
        for i in range(1, n_splits):
            final_performances = pd.concat([final_performances, pd.DataFrame(splits_history[i].history["val_loss"])], axis=1)
        final_performances["mean"] = final_performances.mean(axis=1)
        final_performances.to_csv(str(datetime.now().strftime("%Y%m%d-%H%M%S")) + ".csv", index=False)
        # print(final_performances)
        return final_performances[final_performances["mean"] == final_performances["mean"].min()]

    elif mode == "train_full":
        model = keras.models.load_model(load_state)
        if freeze_layers is not None:
            for layer in model.layers[:-freeze_layers]:
                layer.trainable = False

        if pop_layers is not None:
            new_model = tf.keras.models.Sequential(model.layers[:-pop_layers])
            new_model.add(Dense(16, activation="relu"))
            new_model.add(Dense(1, activation="sigmoid"))
            for layer in new_model.layers:
                print(layer.trainable)
            model = new_model

        # Prepare training
        model.final_name = "bla"
        path = os.path.join(output_dir, 'keras_states/' + model.final_name + str(random.randint(1, 10000)) + '.epoch{epoch}_trainfull.state')
        os.makedirs(path, exist_ok=True)
        saver = ModelCheckpoint(path, verbose=verbose, period=save_every)
        keras_logs = os.path.join(output_dir, 'keras_logs')
        os.makedirs(keras_logs, exist_ok=True)
        csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                               append=True, separator=';')

        model.compile(optimizer=Adam(learning_rate=lr),
                      loss="binary_crossentropy",
                      metrics=[AUC(), AUC(curve="PR"), BinaryAccuracy(), Precision(), Recall()])

        print("Initial Performance")
        eval = model.evaluate(train_raw[0], train_raw[1], batch_size=batch_size)
        print(eval)
        history = model.fit(x=train_raw[0],
                            y=train_raw[1],
                            epochs=epochs,
                            shuffle=True,
                            callbacks=[saver, csv_logger],
                            verbose=verbose,
                            batch_size=batch_size,
                            class_weight=class_weight)
        return model

    elif mode == "test":
        with open("../data/ards_ihm_v2/test.pkl", "rb") as f:
            test = pickle.load(f)
        data = np.asarray(test["data"][0]).astype('float32')
        labels = np.asarray(test["data"][1]).astype('float32')
        model = keras.models.load_model(load_state)

        # Execute Random Sampling
        for i in tqdm(range(10000)):
            data_sample, labels_sample = resample(data, labels, replace=True,
                                                  random_state=np.random.RandomState(42 + i))
            predictions = model.predict(data_sample, batch_size=batch_size, verbose=verbose)
            predictions = np.array(predictions)[:, 0]
            results = metrics.print_metrics_binary(labels_sample, predictions, verbose=verbose)
            results = [log_loss(labels_sample, predictions), results["auroc"], results["auprc"], results["acc"],
                       results["prec1"], results["rec1"]]
            if i == 0:
                eval_df = pd.DataFrame([results])
                eval_df.columns = ["loss", "auroc", "auprc", "acc", "prec1", "rec1"]
            else:
                eval_df.loc[i] = results
        eval_df["f1"] = 2 * (eval_df["prec1"] * eval_df["rec1"]) / (eval_df["prec1"] + eval_df["rec1"])
        os.makedirs("./cv_test/", exist_ok=True)
        model_name = load_state.split("\\")[-1]
        eval_df.to_csv("./cv_test/transfer_" + model_name + ".csv", index=False)


def hyperparameter_tuning():
    train_state = r"..\model\keras_states\k_lstm.n64.d0.4.rd0.3.dep.lr0.01_bs32_ts1.0.epoch12.test0.2959420382976532.state"
    mode = "train"
    units, depth, batch_size, dropout, recurrent_dropout = 64, 3, 32, 0.4, 0.3
    lrs = [0.01]
    freeze_layers_list = [None, 1, 3]
    pop_layers_list = [None, 1]
    add_dense_list = [True, False]
    balance_list = [True, False]

    for lr in lrs:
        for freeze_layers in freeze_layers_list:
            for pop_layers in pop_layers_list:
                for add_dense in add_dense_list:
                    for balance in balance_list:
                        min_value = fine_tune(mode="train", batch_size=batch_size, load_state=train_state, lr=lr,
                                              epochs=30, freeze_layers=freeze_layers, pop_layers=pop_layers,
                                              balance=balance, units=units, dropout=dropout, add_dense=add_dense,
                                              recurrent_droput=recurrent_dropout, depth=depth, verbose=0)
                        print(";".join([str(lr), str(freeze_layers), str(pop_layers), str(add_dense), str(balance)]))
                        print(min_value)


if __name__ == "__main__":
    hyperparameter_tuning()
    # train_state = r"..\model\keras_states\k_lstm.n64.d0.4.rd0.3.dep.lr0.01_bs32_ts1.0.epoch12.test0.2959420382976532.state"
    # test_state = r".\keras_states\bla2386.epoch18_trainfull.state"
    # mode = "train"
    # units, depth, batch_size, lr, dropout, recurrent_dropout = 64, 3, 32, 0.01, 0.4, 0.3
    # freeze_layers, pop_layers = 2, None
    # balance = False
    # epochs = 20
    #
    # if mode == "train":
    #     fine_tune(mode="train", batch_size=batch_size, load_state=train_state, lr=lr, epochs=50,
    #               freeze_layers=freeze_layers, pop_layers=pop_layers, balance=balance, units=units, dropout=dropout,
    #               recurrent_droput=recurrent_dropout, depth=depth, verbose=0)
    # elif mode == "train_full":
    #     fine_tune(mode="train_full", batch_size=batch_size, load_state=train_state, lr=lr, epochs=epochs,
    #               freeze_layers=freeze_layers, pop_layers=pop_layers, balance=balance, units=units, dropout=dropout,
    #               recurrent_droput=recurrent_dropout, depth=depth, verbose=1)
    # elif mode == "test":
    #     fine_tune(mode="test", batch_size=batch_size, load_state=train_state, lr=lr, epochs=epochs,
    #               freeze_layers=freeze_layers, pop_layers=pop_layers, balance=balance, units=units, dropout=dropout,
    #               recurrent_droput=recurrent_dropout, depth=depth, verbose=0)