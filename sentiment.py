from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class Sentiment:
    def __init__(self):
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=None
        )

    def score_sentence(self, sentence: str) -> float:
        """
        Returns numerical sentiment score:
        Positive → 3 + 2*prob
        Neutral  → 3
        Negative → 1 + 2*prob
        """
        if not sentence.strip():
            return 3.0
        try:
            res = self.pipe(sentence)
        except Exception:
            return 3.0

        if isinstance(res, list) and res and isinstance(res[0], list):
            res_list = res[0]
        elif isinstance(res, list):
            res_list = res
        else:
            res_list = [res]

        try:
            best = max(res_list, key=lambda x: x.get("score", 0.0))
        except Exception:
            best = res_list[0]

        label = best.get("label", "").lower()
        raw_score = float(best.get("score", 0.0))

        if label in ("positive", "pos"):
            return 3 + 2 * raw_score
        elif label in ("neutral", "neu"):
            return 3.0
        else:  # negative
            return 1 + 2 * raw_score
