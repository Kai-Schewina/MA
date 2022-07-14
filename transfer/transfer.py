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


def fine_tune(mode="train", depth=2, units=64, dropout=0.3, kernel_reg=0.02, load_state="", save_every=1, batch_size=16,
              lr=0.001, balance=False):
    with open("../data/ards_ihm/train_raw.pkl", "rb") as f:
        train_raw = pickle.load(f)
        train_raw = (np.asarray(train_raw[0]).astype('float32'), np.asarray(train_raw[1]).astype('float32'))
    input_dim = train_raw[0].shape[2]

    model = build_model(depth, units, dropout, input_dim, kernel_reg)
    model.load_weights(load_state).expect_partial()
    model.summary()

    n_trained_chunks = int(re.match(".*epoch([0-9]+).*", load_state).group(1))
    output_dir = "."

    if balance:
        weight_for_0 = (1 / len(train_raw[1])) * (len(train_raw[1]) / 2.0)
        weight_for_1 = (1 / sum(train_raw[1])) * (len(train_raw[1]) / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print('Weight for class 0: {:.2f}'.format(weight_for_0))
        print('Weight for class 1: {:.2f}'.format(weight_for_1))
    else:
        class_weight = None

    model.final_name = "bla"
    # Prepare training
    path = os.path.join(output_dir, 'keras_states/' + model.final_name + '.epoch{epoch}.test{val_loss}.state')
    os.makedirs(path, exist_ok=True)
    saver = ModelCheckpoint(path, verbose=1, period=save_every)
    keras_logs = os.path.join(output_dir, 'keras_logs')
    os.makedirs(keras_logs, exist_ok=True)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="binary_crossentropy")

    max_epochs = 10
    final_performances = []
    if mode == "train":
        for i in range(1, max_epochs):
            n_splits = 5
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)

            inputs = train_raw[0]
            targets = np.asarray(train_raw[1])
            splits_history = []

            for train, val in kfold.split(inputs, targets):
                metrics_callback = metrics.InHospitalMortalityMetrics(train_data=(inputs[train], targets[train]),
                                                                  val_data=(inputs[val], targets[val]),
                                                                  batch_size=batch_size,
                                                                  verbose=1)
                history = model.fit(x=inputs[train],
                          y=targets[train],
                          validation_data=(inputs[val], targets[val]),
                          epochs=n_trained_chunks + i,
                          initial_epoch=n_trained_chunks,
                          shuffle=True,
                          callbacks=[metrics_callback, saver, csv_logger],
                          verbose=1,
                          batch_size=batch_size,
                          class_weight=class_weight)
                splits_history.append(history)

            val_sum = 0
            for e in splits_history:
                for auprc in e.history["val_auprc"]:
                    val_sum += auprc
            final_performances.append((val_sum / (i * n_splits)))
        print(final_performances)

    elif mode == "test":
        with open("../data/ards_ihm/test.pkl", "rb") as f:
            test = pickle.load(f)
        data = np.asarray(test["data"][0]).astype('float32')
        labels = np.asarray(test["data"][1]).astype('float32')
        names = np.asarray(test["names"])

        predictions = model.predict(data, batch_size=batch_size, verbose=1)
        predictions = np.array(predictions)[:, 0]
        metrics.print_metrics_binary(labels, predictions)

        eval = model.evaluate(data, labels, batch_size=batch_size)
        print(eval)


if __name__ == "__main__":
    train_state = r"..\model\keras_states\k_lstm.n16.d0.3.dep2_bs8_ts1.0.epoch6.test0.35124555230140686.state"
    test_state = r".\keras_states\bla.epoch10.test0.036394696682691574.state"

    fine_tune(mode="test", depth=2, units=16, dropout=0.3, kernel_reg=0.00, batch_size=8, load_state=test_state)
