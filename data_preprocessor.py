from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, cleaner, min_confidence=0.7):
        self.cleaner = cleaner
        self.min_confidence = min_confidence

    def process(self, df):
        df_filtered = df[df['confidence'] >= self.min_confidence].copy()
        df_filtered['clean_text'] = df_filtered['text'].apply(self.cleaner.clean)
        X = df_filtered['clean_text']
        y = df_filtered['label']
        return train_test_split(X, y, test_size=0.2, random_state=42)