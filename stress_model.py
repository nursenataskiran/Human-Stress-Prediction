from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd

class StressModel:
    def __init__(self):
        self.model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

    def train(self, X_train_vec, y_train):
        self.model.fit(X_train_vec, y_train)

    def evaluate(self, X_test_vec, y_test):
        y_pred = self.model.predict(X_test_vec)
        report = classification_report(y_test, y_pred, output_dict=True)
        return pd.DataFrame(report).transpose()

    def predict(self, X_vec):
        return self.model.predict(X_vec)

    def predict_proba(self, X_vec):  # ⭐️ EKLENEN FONKSİYON
        return self.model.predict_proba(X_vec)
