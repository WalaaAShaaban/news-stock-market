from sklearn.neighbors import KNeighborsClassifier
from src.models.train_model import TrainModel
class KNN:
    def __init__(self):
        print("KNN")
        

    def fit(self):
        model = TrainModel()
        model.split_data()  # Load & split data

        X_train, y_train = model.get_train_data()
        
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(X_train, y_train)

    def predict(self, X):
        pass