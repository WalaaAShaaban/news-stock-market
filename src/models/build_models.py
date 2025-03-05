
# Logistic Regression
from src.models.build_models.logisticRegression import LogisticRegressionModel
from src.models.train_model import TrainModel
class BuildModles:
    def __init__(self):
        pass
    
    def build_model(self):
        trainModel = TrainModel()
        trainModel.split_data()
        X_train, y_train = trainModel.get_train_data()
        X_test, y_test = trainModel.get_test_data()
        logisticRegressionModel = LogisticRegressionModel()
        logisticRegressionModel.fit(X_train, y_train)
        logisticRegressionModel.evaluate(X_test, y_test)

