from sklearn.tree import DecisionTreeClassifier

class DecisionTreeClassifier:
    def __init__(self):
        print("Decision Tree Classifier")

    def fit(self, X_train, y_train):
        self.dtc = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
        self.dtc.fit(X_train, y_train)

    def predict(self, X_test):
            return self.dtc.predict(X_test)