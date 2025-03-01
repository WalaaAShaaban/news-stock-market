from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

class TrainModel:
    def __init__(self):
        pass


    def split_data(self):
        df = pd.read_csv('data/processed-data/clean_news.csv')[:40000]

        df['news'] = df['news'].fillna('')
        X = df['news']
        y = df['sentiment']
    # TF-IDF vectorization
        tf_vectorizer = TfidfVectorizer()
        X = tf_vectorizer.fit_transform(X).toarray()

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return (X_train.shape, X_test.shape, y_train.shape, y_test.shape)
