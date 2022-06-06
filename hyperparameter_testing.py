from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import pickle


class MyHyperModel(kt.HyperModel):

    def build(self, hp):
        def weighted_bincrossentropy(y_true, y_pred, weight_zero=0.5, weight_one=4):
            bin_crossentropy = keras.backend.binary_crossentropy(y_true, y_pred)

            weights = y_true * weight_one + (1. - y_true) * weight_zero
            weighted_bin_crossentropy = weights * bin_crossentropy

            return keras.backend.mean(weighted_bin_crossentropy)

        units = hp.Int("units", min_value=16, max_value=128, step=16)
        depth = hp.Int("depth", min_value=2, max_value=8, step=1)
        dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05)
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model = keras.Sequential()
        model.add(layers.Input(shape=(None, 63)))
        # model.add(layers.Masking())
        # for i in range(depth - 1):
        #     model.add(layers.Bidirectional(layers.LSTM(units=int(units/2), return_sequences=True,
        #                                                dropout=dropout, recurrent_dropout=dropout)))
        # model.add(layers.LSTM(units=units))
        for i in range(depth - 1):
            model.add(layers.CuDNNLSTM(units=int(units), return_sequences=True))
        model.add(layers.CuDNNLSTM(units=units))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1, activation="sigmoid"))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=weighted_bincrossentropy, metrics=[keras.metrics.AUC(curve="PR")])
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [16, 32, 64, 128, 256]),
            **kwargs,
        )


tuner = kt.Hyperband(
    MyHyperModel(),
    objective=kt.Objective("val_auc", direction="max"),
    max_epochs=100,
    hyperband_iterations=5,
    directory="./try_10/")

with open("./data/in-hospital-mortality_v3/train_raw.pkl", "rb") as f:
    train_raw = pickle.load(f)

with open("./data/in-hospital-mortality_v3/val_raw.pkl", "rb") as f:
    val_raw = pickle.load(f)

tuner.search(train_raw[0], train_raw[1], validation_data=(val_raw[0], val_raw[1]), callbacks=[EarlyStopping('val_loss', patience=3)])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
