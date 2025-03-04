import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class KNNModel:
    def __init__(self, n_neighbors=3):
        print("Initializing KNN Classifier...")
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def fit(self, X_train, y_train):
        with mlflow.start_run():  # Start MLflow tracking
            self.knn.fit(X_train, y_train)

            # Log hyperparameters
            mlflow.log_param("n_neighbors", self.n_neighbors)

            print("Model trained successfully.")

    def predict(self, X_test):
        return self.knn.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)

        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        # Save the trained model to MLflow
        mlflow.sklearn.log_model(self.knn, "knn_model")
        print("Model saved to MLflow.")

        return acc, precision, recall, f1
