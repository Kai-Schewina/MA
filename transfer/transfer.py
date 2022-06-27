from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers
from tensorflow import keras
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import re
import os
from model import metrics
import numpy as np


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


state = r"C:\Users\Masterarbeit\PycharmProjects\MA_v3\model\keras_states\k_lstm.n16.d0.3.dep2_bs8_ts1.0.epoch9.test0.26745611214968024.state"
with open("../data/ards_ihm/train_raw.pkl", "rb") as f:
    train_raw = pickle.load(f)
input_dim = train_raw[0].shape[2]

model = build_model(2, 16, 0.3, input_dim)
model.load_weights(state)
model.summary()

model.summary()
#%%
optimizer_config = {'class_name': "adam",
                    'config': {'lr': 0.001,
                               'beta_1': 0.9}}
n_trained_chunks = int(re.match(".*epoch([0-9]+).*", state).group(1))
output_dir = "."
save_every = 1
batch_size = 16
verbose = 10
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

model.compile(optimizer=optimizer_config,
              loss="binary_crossentropy")

max_epochs = 10
final_performances = []

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
                                                          verbose=verbose)
        history = model.fit(x=inputs[train],
                  y=targets[train],
                  validation_data=[inputs[val], targets[val]],
                  epochs=n_trained_chunks + i,
                  initial_epoch=n_trained_chunks,
                  shuffle=True,
                  callbacks=[metrics_callback, saver, csv_logger],
                  verbose=verbose,
                  batch_size=batch_size,
                  class_weight=class_weight)
        splits_history.append(history)

    val_sum = 0
    for e in splits_history:
        for auprc in e.history["val_auprc"]:
            val_sum += auprc
    final_performances.append((val_sum / (i * n_splits)))
print(final_performances)