from sklearn.svm import SVC

class SVM:
    def __init__(self):
        print("SVM")

    def fit(self, X_train, y_train):
        self.svm = SVC(kernel='linear', C=1.0)
        self.svm.fit(X_train, y_train)  

    def predict(self, X_test):
        return self.svm.predict(X_test)