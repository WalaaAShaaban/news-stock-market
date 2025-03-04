import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class XGBoostModel:
    def __init__(self, objective="multi:softmax", num_class=3, eval_metric="mlogloss"):
        print("Initializing XGBoost Model...")
        self.objective = objective
        self.num_class = num_class
        self.eval_metric = eval_metric
        self.model = xgb.XGBClassifier(objective=self.objective, 
                                       num_class=self.num_class, 
                                       eval_metric=self.eval_metric, 
                                       use_label_encoder=False)

    def fit(self, X_train, y_train):
        with mlflow.start_run():  # Start MLflow tracking
            # Log hyperparameters
            mlflow.log_param("objective", self.objective)
            mlflow.log_param("num_class", self.num_class)
            mlflow.log_param("eval_metric", self.eval_metric)

            # Train model
            self.model.fit(X_train, y_train)
            print("Model trained successfully.")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)

        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        # Save the trained model to MLflow
        mlflow.sklearn.log_model(self.model, "xgboost_model")
        print("Model saved to MLflow.")

        return acc, precision, recall, f1
