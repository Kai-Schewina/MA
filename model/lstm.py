from __future__ import absolute_import
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, Dropout, CuDNNLSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed


class Network(Model):
    def __init__(self, dim, batch_norm, dropout, rec_dropout, task,
                 num_classes=1,
                 depth=1, input_dim=76, **kwargs):

        print("==> not used params in network class:", kwargs.keys())

        self.dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth

        final_activation = 'sigmoid'

        # Input layers and masking
        X = Input(shape=(None, input_dim), name='X')
        inputs = [X]
        mX = Masking()(X)

        # Configurations
        is_bidirectional = True

        # Main part of the network
        for i in range(depth - 1):
            num_units = dim
            if is_bidirectional:
                num_units = num_units // 2

            lstm = LSTM(units=num_units,
                        activation='tanh',
                        return_sequences=True,
                        recurrent_dropout=rec_dropout,
                        dropout=dropout)

            if is_bidirectional:
                mX = Bidirectional(lstm)(mX)
            else:
                mX = lstm(mX)

        # Output module of the network
        L = LSTM(units=dim,
                 activation='tanh',
                 dropout=dropout,
                 recurrent_dropout=rec_dropout)(mX)

        if dropout > 0:
            L = Dropout(dropout)(L)

        y = Dense(num_classes, activation=final_activation)(L)
        outputs = [y]

        super(Network, self).__init__(inputs=inputs, outputs=outputs)

    def say_name(self):
        return "{}.n{}{}{}{}.dep{}".format('k_lstm',
                                           self.dim,
                                           ".bn" if self.batch_norm else "",
                                           ".d{}".format(self.dropout) if self.dropout > 0 else "",
                                           ".rd{}".format(self.rec_dropout) if self.rec_dropout > 0 else "",
                                           self.depth)
