import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

class PreprocessingData:
    df:pd.DataFrame = None

    def __init__(self) -> None:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4') 

    def cleanNews(self):
        corpus=[]
        stop_words = stopwords.words("english")
        stop_words.remove("not")
        pos = PorterStemmer()
        lem = WordNetLemmatizer()
        for i in range(0, len(self.news)):
            new = re.sub('[^a-zA-Z0-9]', " ", self.news[i])
            new = new.lower()
            new = [lem.lemmatize(word) for word in nltk.word_tokenize(new) if word not in stop_words]
            new = " ".join(new)
            corpus.append(new)
        return corpus

    def read_data(self):
        self.df = pd.read_csv('/home/walaa-shaban/Documents/Training Qafza/news-stock-market/input/news.csv')
        self.news = self.df['news'].values
        self.df['news'] = self.cleanNews()
    
    def save_data(self):
        self.df.to_csv('/home/walaa-shaban/Documents/Training Qafza/news-stock-market/output/clean_news.csv', index=False)