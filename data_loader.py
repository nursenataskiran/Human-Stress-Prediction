import pandas as pd

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_path)
        # Gerekli sütunların olup olmadığını kontrol et
        expected_columns = {'text', 'label', 'confidence'}
        if not expected_columns.issubset(df.columns):
            raise ValueError(f"CSV dosyası şu sütunları içermeli: {expected_columns}")
        return df