from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TrainModel:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None


    def split_data(self):
        df = pd.read_csv('data/processed-data/clean_news.csv')[:1000]

        df['news'] = df['news'].fillna('')
        X = df['news']
        y = df['sentiment']
    # TF-IDF vectorization
        tf_vectorizer = TfidfVectorizer()
        X = tf_vectorizer.fit_transform(X).toarray()

        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        

    def get_train_data(self):
        return (self.X_train, self.y_train)
    
    def get_test_data(self):
        return self.X_test, self.y_test
    