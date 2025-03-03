from sklearn.neighbors import KNeighborsClassifier
class KNN:
    def __init__(self):
        print("KNN")
        

    def fit(self, X_train, y_train):
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(X_train, y_train)

    def predict(self, X_test):
        return self.knn.predict(X_test)