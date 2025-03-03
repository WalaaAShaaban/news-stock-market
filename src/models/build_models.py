
from src.models.train_model import TrainModel
from src.models.build_models.LogisticRegression import LogisticRegression
from src.models.build_models.KNN import KNN

class BuildModels:

    def __init__(self):
        self.train_model = TrainModel()
        self.train_model.split_data()
        self.X_train, self.y_train = self.train_model.get_train_data()
        self.X_test, self.y_test = self.train_model.get_test_data()

    def build_models(self):
        self.KNN = KNN()
        self.KNN.fit(self.X_train, self.y_train)
        

