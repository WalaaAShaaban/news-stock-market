from sklearn.naive_bayes import MultinomialNB

class NaiveBase:
    def __init__(self):
        print("NaiveBase")

    def fit(self, X_train, y_train):
        self.naive_base = MultinomialNB()
        self.naive_base.fit(X_train, y_train)

    def predict(self, X_test):
        return self.naive_base.predict(X_test)
