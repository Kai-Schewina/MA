from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from sklearn import metrics
from tensorflow.keras.callbacks import Callback
import keras.backend as K


class InHospitalMortalityMetrics(Callback):
    def __init__(self, train_data, val_data, batch_size=32, verbose=2):
        super(InHospitalMortalityMetrics, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.verbose = verbose
        self.train_history = []
        self.val_history = []

    def calc_metrics(self, data, history, dataset, logs):
        y_true = []
        predictions = []
        B = self.batch_size
        for i in range(0, len(data[0]), B):
            if self.verbose == 1:
                print("\tdone {}/{}".format(i, len(data[0])), end='\r')
            (x, y) = (data[0][i:i + B], data[1][i:i + B])
            outputs = self.model.predict(x, batch_size=B)
            predictions += list(np.array(outputs).flatten())
            y_true += list(np.array(y).flatten())
        print('\n')
        predictions = np.array(predictions)
        predictions = np.stack([1 - predictions, predictions], axis=1)
        ret = print_metrics_binary(y_true, predictions)
        for k, v in ret.items():
            logs[dataset + '_' + k] = v
        history.append(ret)

    def on_epoch_end(self, epoch, logs={}):
        print("\n==>predicting on train")
        self.calc_metrics(self.train_data, self.train_history, 'train', logs)
        print("\n==>predicting on validation")
        self.calc_metrics(self.val_data, self.val_history, 'val', logs)


def print_metrics_binary(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])
    f1 = metrics.f1_score(y_true, predictions[:, 1].round())

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))
        print("F1 of = {}".format(f1))

    return {"acc": acc,
            "prec0": prec0,
            "prec1": prec1,
            "rec0": rec0,
            "rec1": rec1,
            "auroc": auroc,
            "auprc": auprc,
            "minpse": minpse,
            "f1": f1}


# https://aakashgoel12.medium.com/how-to-add-user-defined-function-get-f1-score-in-keras-metrics-3013f979ce0d
def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
