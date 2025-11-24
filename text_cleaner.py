import os
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Use project-level nltk_data directory
project_path = Path(__file__).resolve().parent
nltk_data_path = project_path / "nltk_data"

# Create directory if missing
os.makedirs(nltk_data_path, exist_ok=True)

# Download NLTK resources into project folder
nltk.download("stopwords", download_dir=nltk_data_path)
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("wordnet", download_dir=nltk_data_path)

# Add path so NLTK knows where to look
nltk.data.path.append(str(nltk_data_path))

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
