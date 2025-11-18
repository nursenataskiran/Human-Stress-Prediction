class StressPredictor:
    def __init__(self, model, extractor, cleaner):
        self.model = model
        self.extractor = extractor
        self.cleaner = cleaner

    def predict_from_text(self, raw_text: str) -> str:
        cleaned = self.cleaner.clean(raw_text)
        vectorized = self.extractor.transform([cleaned])
        probas = self.model.predict_proba(vectorized)[0]
        prediction = self.model.predict(vectorized)[0]
        stress_prob = probas[1]  # class 1: stresli

        label = "Stresli" if prediction == 1 else "Stresli değil"
        percent = round(stress_prob * 100, 2)

        return f"{label} (%{percent} stres olasılığı)"
