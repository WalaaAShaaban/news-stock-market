import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from sklearn.metrics import accuracy_score

class RNNModel:
    def __init__(self, max_features=10000, max_len=200):
        print("Initializing RNN...")
        self.max_features = max_features  
        self.max_len = max_len  

    def fit(self, X_train, y_train, epochs=3, batch_size=32):
        with mlflow.start_run():  # Start MLflow tracking
            self.model = keras.Sequential()
            self.model.add(Embedding(input_dim=self.max_features, output_dim=128))
            self.model.add(SimpleRNN(units=64))
            self.model.add(Dense(units=1, activation='sigmoid'))
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Log hyperparameters
            mlflow.log_param("max_features", self.max_features)
            mlflow.log_param("max_len", self.max_len)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)

            # Train model
            history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

            # Log loss & accuracy from training
            mlflow.log_metric("train_loss", history.history["loss"][-1])
            mlflow.log_metric("train_accuracy", history.history["accuracy"][-1])

            print("Model trained successfully.")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = (self.predict(X_test) > 0.5).astype("int32")  # Convert probabilities to class labels

        acc = accuracy_score(y_test, y_pred)

        # Log accuracy to MLflow
        mlflow.log_metric("test_accuracy", acc)

        print(f"Test Accuracy: {acc:.4f}")

        # Save the trained model to MLflow
        mlflow.tensorflow.log_model(self.model, "rnn_model")
        print("Model saved to MLflow.")

        return acc
