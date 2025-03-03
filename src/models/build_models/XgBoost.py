import xgboost as xgb

class XGBoost:
    def __init__(self):
        print("XGBoost")

    def fit(self, X_train, y_train):
        self.xgb = xgb.XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss", use_label_encoder=False)
        self.xgb.fit(X_train, y_train)

    def predict(self, X_test):
        return self.xgb.predict(X_test)