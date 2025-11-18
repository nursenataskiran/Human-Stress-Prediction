import os
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# NLTK setup
nltk_data_path = r"C:\Users\nurse\OneDrive\Masaüstü\stress prediction\nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.download("stopwords", download_dir=nltk_data_path)
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("wordnet", download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)

class TextCleaner:
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r"[^a-zA-Z\s]", '', text)
        text = re.sub(r"\s+", ' ', text).strip()
        words = word_tokenize(text)
        filtered_words = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word not in self.stop_words
        ]
        return ' '.join(filtered_words)
