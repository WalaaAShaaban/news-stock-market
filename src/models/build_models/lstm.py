import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

class LSTMModel:
    def __init__(self, input_dim=10000, output_dim=128, lstm_units=64, dropout_rate=0.5):
        print("Initializing LSTM Model...")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            Embedding(input_dim=self.input_dim, output_dim=self.output_dim),
            LSTM(units=self.lstm_units),
            Dropout(self.dropout_rate),
            Dense(units=1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train, epochs=3, batch_size=32):
        with mlflow.start_run():  # Start MLflow tracking
            # Log hyperparameters
            mlflow.log_param("input_dim", self.input_dim)
            mlflow.log_param("output_dim", self.output_dim)
            mlflow.log_param("lstm_units", self.lstm_units)
            mlflow.log_param("dropout_rate", self.dropout_rate)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)

            # Train model
            history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

            # Log metrics
            final_accuracy = history.history['accuracy'][-1]
            final_loss = history.history['loss'][-1]
            mlflow.log_metric("final_accuracy", final_accuracy)
            mlflow.log_metric("final_loss", final_loss)

            print(f"Final Accuracy: {final_accuracy:.4f}, Final Loss: {final_loss:.4f}")

            # Save model to MLflow
            mlflow.tensorflow.log_model(self.model, "lstm_model")
            print("Model saved to MLflow.")

    def predict(self, X_test):
        return self.model.predict(X_test)
