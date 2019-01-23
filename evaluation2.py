#https://neurowhai.tistory.com/142?category=605395

# -*- coding: utf-8 -*-

import numpy as np
import keras
from keras import layers, models, datasets
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics
from sklearn.preprocessing import MinMaxScaler
import os


class Data:
    def __init__(self, max_features=20000, maxlen=80):
        (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(
            num_words=max_features)
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test


class RNN_LSTM(models.Model):
    def __init__(self, max_features, maxlen):
        x = layers.Input((maxlen,))
        h = layers.Embedding(max_features, 128)(x)
        h = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(h)
        y = layers.Dense(1, activation='sigmoid')(h)
        super().__init__(x, y)

        self.compile(loss='binary_crossentropy',
                     optimizer='adam', metrics=['accuracy'])


class Machine:
    def __init__(self, max_features=20000, maxlen=80):
        self.data = Data(max_features, maxlen)
        self.model = RNN_LSTM(max_features, maxlen)

    def run(self, epochs=3, batch_size=32):
        data = self.data
        model = self.model

        print('Training stage')
        model.fit(data.x_train, data.y_train,
                  batch_size=batch_size, epochs=epochs,
                  validation_data=(data.x_test, data.y_test),
                  verbose=2)

        loss, acc = model.evaluate(data.x_test, data.y_test,
                                   batch_size=batch_size, verbose=2)

        print('Test performance: accuracy={0}, loss={1}'.format(acc, loss))


def main():
    m = Machine()
    m.run()


if __name__ == '__main__':
    main()

