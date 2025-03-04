import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class DecisionTreeModel:
    def __init__(self):
        print("Initializing Decision Tree Classifier...")
        self.dtc = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)

    def fit(self, X_train, y_train):
        with mlflow.start_run():  # Start MLflow tracking
            self.dtc.fit(X_train, y_train)

            # Log hyperparameters
            mlflow.log_param("criterion", self.dtc.criterion)
            mlflow.log_param("max_depth", self.dtc.max_depth)
            mlflow.log_param("random_state", 42)

            print("Model trained successfully.")

    def predict(self, X_test):
        return self.dtc.predict(X_test)

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
        mlflow.sklearn.log_model(self.dtc, "decision_tree_model")
        print("Model saved to MLflow.")

        return acc, precision, recall, f1
