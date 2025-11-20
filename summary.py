from transformers import pipeline

class Summary:
    def __init__(self, max_input_len=3000):
        self.max_input_len = max_input_len
        self.pipe = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6"
        )

    def summarize(self, sentences):
        """
        sentences: list of sentences or a single string.
        Returns a short summary string.
        """

        if isinstance(sentences, list):
            text = " ".join(sentences)
        else:
            text = str(sentences)

        text = text[:self.max_input_len]

        if len(text.split()) < 5:
            return text

        try:
            summary = self.pipe(
                text,
                max_length=80,
                min_length=10,
                do_sample=False
            )[0]["summary_text"]
        except Exception:
            # fallback if summarizer fails
            return (text[:200] + "...") if len(text) > 200 else text

        return summary
