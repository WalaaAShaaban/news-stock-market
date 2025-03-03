import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb


class LSTM:
    def __init__(self):
        print("LSTM")

    def fit(self, X_train, y_train):
        self.model = keras.Sequential()
        self.model.add(Embedding(input_dim=10000, output_dim=128))
        self.model.add(LSTM(units=64))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=3, batch_size=32)

    def predict(self, X_test):
        return self.model.predict(X_test)