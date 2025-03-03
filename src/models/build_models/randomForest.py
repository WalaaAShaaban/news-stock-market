from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self):
        print("RandomForest")

    def fit(self, X_train, y_train):
            self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
            self.random_forest.fit(X_train, y_train)


    def predict(self, X_test):
        return self.random_forest.predict(X_test)