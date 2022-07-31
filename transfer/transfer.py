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
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Precision, Recall
from datetime import datetime
import random


def build_model(depth, units, dropout, recurrent_droput, input_dim, kernel_reg):
    model = keras.Sequential()
    model.add(Input(shape=(None, input_dim)))
    for i in range(depth - 1):
        model.add(Bidirectional(LSTM(units=int(units/2), return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_droput, kernel_regularizer=L1L2(l1=kernel_reg, l2=kernel_reg),
                       recurrent_regularizer=L1L2(l1=0.00, l2=0.00), bias_regularizer=L1L2(l1=0.00, l2=0.00),
                       activity_regularizer=L1L2(l1=0.0, l2=0.0))))
    model.add(LSTM(units=units, dropout=dropout, recurrent_dropout=recurrent_droput, kernel_regularizer=L1L2(l1=kernel_reg, l2=kernel_reg),
                   recurrent_regularizer=L1L2(l1=0.00, l2=0.00), bias_regularizer=L1L2(l1=0.00, l2=0.00),
                   activity_regularizer=L1L2(l1=0.0, l2=0.0)))
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
              lr=0.001, balance=False, epochs=10, freeze_layers=False, pop_layers=False, n_splits=3):
    with open("../data/ards_ihm_v2/train_raw.pkl", "rb") as f:
        train_raw = pickle.load(f)
        train_raw = (np.asarray(train_raw["data"][0]).astype('float32'), np.asarray(train_raw["data"][1]).astype('float32'))

    output_dir = "."

    if balance:
        weight_for_0 = (1 / len(train_raw[1])) * (len(train_raw[1]) / 2.0)
        weight_for_1 = (1 / sum(train_raw[1])) * (len(train_raw[1]) / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print('Weight for class 0: {:.2f}'.format(weight_for_0))
        print('Weight for class 1: {:.2f}'.format(weight_for_1))
    else:
        class_weight = None

    if mode == "train":
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        inputs = train_raw[0]
        targets = np.asarray(train_raw[1])
        splits_history = []

        for train, val in kfold.split(inputs, targets):
            model = keras.models.load_model(load_state, custom_objects={"f1": metrics.f1})
            if freeze_layers:
                for layer in model.layers[:-2]:
                    layer.trainable = False

            if pop_layers:
                new_model = tf.keras.models.Sequential(model.layers[:-1])
                new_model.add(Dense(16, activation="relu"))
                new_model.add(Dense(1, activation="sigmoid"))
                for layer in new_model.layers:
                    print(layer.trainable)
                model = new_model

            # Prepare training
            model.final_name = "bla"
            path = os.path.join(output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')
            os.makedirs(path, exist_ok=True)
            saver = ModelCheckpoint(path, verbose=1, period=save_every)
            keras_logs = os.path.join(output_dir, 'keras_logs')
            os.makedirs(keras_logs, exist_ok=True)
            csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                                   append=True, separator=';')

            model.compile(optimizer=Adam(learning_rate=lr),
                          loss="binary_crossentropy",
                          metrics=[AUC(), AUC(curve="PR"), BinaryAccuracy(), Precision(), Recall(), metrics.f1])

            metrics_callback = metrics.InHospitalMortalityMetrics(train_data=(inputs[train], targets[train]),
                                                              val_data=(inputs[val], targets[val]),
                                                              batch_size=batch_size,
                                                              verbose=1)
            history = model.fit(x=inputs[train],
                      y=targets[train],
                      validation_data=(inputs[val], targets[val]),
                      epochs=epochs,
                      shuffle=True,
                      callbacks=[metrics_callback, saver, csv_logger],
                      verbose=1,
                      batch_size=batch_size,
                      class_weight=class_weight)
            model.summary()
            splits_history.append(history)

        final_performances = pd.DataFrame(splits_history[0].history["val_loss"])
        for i in range(1, n_splits):
            final_performances = pd.concat([final_performances, pd.DataFrame(splits_history[i].history["val_loss"])], axis=1)
        final_performances["mean"] = final_performances.mean(axis=1)
        final_performances.to_csv(str(datetime.now().strftime("%Y%m%d-%H%M%S")) + ".csv", index=False)
        print(final_performances)

    elif mode == "train_full":
        model = keras.models.load_model(load_state, custom_objects={"f1": metrics.f1})
        if freeze_layers:
            for layer in model.layers[:-2]:
                layer.trainable = False

        if pop_layers:
            new_model = tf.keras.models.Sequential(model.layers[:-1])
            new_model.add(Dense(16, activation="relu"))
            new_model.add(Dense(1, activation="sigmoid"))
            for layer in new_model.layers:
                print(layer.trainable)
            model = new_model

        # Prepare training
        model.final_name = "bla"
        path = os.path.join(output_dir, 'keras_states/' + model.final_name + str(random.randint(1, 10000)) + '.epoch{epoch}_trainfull.state')
        os.makedirs(path, exist_ok=True)
        saver = ModelCheckpoint(path, verbose=1, period=save_every)
        keras_logs = os.path.join(output_dir, 'keras_logs')
        os.makedirs(keras_logs, exist_ok=True)
        csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                               append=True, separator=';')

        model.compile(optimizer=Adam(learning_rate=lr),
                      loss="binary_crossentropy",
                      metrics=[AUC(), AUC(curve="PR"), BinaryAccuracy(), Precision(), Recall(), metrics.f1])

        inputs = train_raw[0]
        targets = np.asarray(train_raw[1])

        history = model.fit(x=inputs,
                            y=targets,
                            epochs=epochs,
                            shuffle=True,
                            callbacks=[saver, csv_logger],
                            verbose=1,
                            batch_size=batch_size,
                            class_weight=class_weight)

    elif mode == "test":
        model = keras.models.load_model(load_state, custom_objects={"f1": metrics.f1})
        with open("../data/ards_ihm_v2/test.pkl", "rb") as f:
            test = pickle.load(f)
        data = np.asarray(test["data"][0]).astype('float32')
        labels = np.asarray(test["data"][1]).astype('float32')

        predictions = model.predict(data, batch_size=batch_size, verbose=1)
        predictions = np.array(predictions)[:, 0]
        metrics.print_metrics_binary(labels, predictions)

        eval = model.evaluate(data, labels, batch_size=batch_size)
        print(eval)


if __name__ == "__main__":
    train_state = "./best_model/"
    test_state = r".\keras_states\bla9049.epoch37_trainfull.state"
    mode = "train"
    if mode == "train":
        fine_tune(mode="train", batch_size=256, load_state=train_state, lr=0.001, epochs=100, freeze_layers=True,
                  pop_layers=True)
    if mode == "train_full":
        fine_tune(mode="train_full", lr=0.001, batch_size=64, epochs=37, pop_layers=True, freeze_layers=True,
                  load_state=train_state)
    if mode == "test":
        fine_tune(mode="test", batch_size=64, lr=0.001, load_state=train_state)
