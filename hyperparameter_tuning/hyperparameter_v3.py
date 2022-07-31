from __future__ import absolute_import
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import LSTM, Input, Dropout, Dense, Masking, Bidirectional
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Precision, Recall
import keras_tuner as kt


class MyHyperModel(kt.HyperModel):

    def build(self, hp):
        hp_depth = hp.Int("depth",  min_value=1, max_value=8, step=1)
        hp_units = hp.Int("units", min_value=32, max_value=128, step=32)
        hp_dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
        hp_rec_dropout = hp.Float("rec_dropout", min_value=0.0, max_value=0.5, step=0.1)
        hp_lr = hp.Choice("lr", values=[1e-2, 1e-3, 1e-4])

        model = keras.Sequential()
        model.add(Input(shape=(48, 70)))
        for i in range(hp_depth - 1):
            model.add(Bidirectional(LSTM(units=int(hp_units/2), return_sequences=True, dropout=hp_dropout, recurrent_dropout=hp_rec_dropout)))
        model.add(LSTM(units=hp_units, dropout=hp_dropout, recurrent_dropout=hp_rec_dropout))
        model.add(Dropout(rate=hp_dropout))
        model.add(Dense(1, activation="sigmoid"))

        optimizer = Adam(learning_rate=hp_lr)
        model.compile(optimizer=optimizer, loss="binary_crossentropy",
                      metrics=[AUC(), AUC(curve="PR"), BinaryAccuracy(), Precision(), Recall()])

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", values=[32, 64]),
            **kwargs
        )


stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner = kt.Hyperband(
    MyHyperModel(),
    objective=kt.Objective("val_loss", direction="min"),
    max_epochs=100,
    hyperband_iterations=5,
    directory="./tuning/v7_2")

data = "../data/in-hospital-mortality_v7/"

with open(data + "/train_raw_balanced.pkl", "rb") as f:
    train_raw = pickle.load(f)
    train_raw = (np.asarray(train_raw["data"][0]).astype('float32'), np.asarray(train_raw["data"][1]).astype('float32'))
with open(data + "/val_raw.pkl", "rb") as f:
    val_raw = pickle.load(f)
    val_raw = (np.asarray(val_raw["data"][0]).astype('float32'), np.asarray(val_raw["data"][1]).astype('float32'))

tuner.search(x=train_raw[0], y=train_raw[1], validation_data=val_raw, verbose=1, callbacks=[stop_early])
