import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        print("Initializing Random Forest Classifier...")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, 
                                            max_depth=self.max_depth, 
                                            random_state=self.random_state)

    def fit(self, X_train, y_train):
        with mlflow.start_run():  # Start MLflow tracking
            # Log hyperparameters
            mlflow.log_param("n_estimators", self.n_estimators)
            mlflow.log_param("max_depth", self.max_depth)
            mlflow.log_param("random_state", self.random_state)

            # Train model
            self.model.fit(X_train, y_train)
            print("Model trained successfully.")

    def predict(self, X_test):
        return self.model.predict(X_test)

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
        mlflow.sklearn.log_model(self.model, "random_forest_model")
        print("Model saved to MLflow.")

        return acc, precision, recall, f1
