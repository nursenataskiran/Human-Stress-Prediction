from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),     # 👉 unigram + bigram
            min_df=5, # en az 5 belgede geçen kelimeler alınsın
            max_df=0.85 # en fazla %85 belgede geçen kelimeler atılsın
        )

    def fit_transform(self, X_train):
        return self.vectorizer.fit_transform(X_train)

    def transform(self, X):
        return self.vectorizer.transform(X)