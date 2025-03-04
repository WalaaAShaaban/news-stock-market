import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

class RNNModel:
    def __init__(self, max_features=10000, max_len=200, rnn_units=64, embedding_dim=128):
        print("Initializing RNN Model...")
        self.max_features = max_features
        self.max_len = max_len
        self.rnn_units = rnn_units
        self.embedding_dim = embedding_dim
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            Embedding(input_dim=self.max_features, output_dim=self.embedding_dim),
            SimpleRNN(units=self.rnn_units),
            Dense(units=1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train, epochs=3, batch_size=32):
        with mlflow.start_run():  # Start MLflow tracking
            # Log hyperparameters
            mlflow.log_param("max_features", self.max_features)
            mlflow.log_param("max_len", self.max_len)
            mlflow.log_param("rnn_units", self.rnn_units)
            mlflow.log_param("embedding_dim", self.embedding_dim)
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
            mlflow.tensorflow.log_model(self.model, "rnn_model")
            print("Model saved to MLflow.")

    def predict(self, X_test):
        return self.model.predict(X_test)
