from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),     
            min_df=5, # Include only the words that appear in at least 5 documents.
            max_df=0.85 # Remove the words that appear in more than 85% of the documents.
        )

    def fit_transform(self, X_train):
        return self.vectorizer.fit_transform(X_train)

    def transform(self, X):
        return self.vectorizer.transform(X)
