from sklearn.linear_model import LogisticRegression

class LogisticRegression:
    def __init__(self):
        print("LogisticRegression")

    def fit(self, X_train, y_train):
        self.logistic_regression = LogisticRegression(max_iter=200)
        self.logistic_regression.fit(X_train, y_train)

    def predict(self, X_test):
        return self.logistic_regression.predict(X_test)