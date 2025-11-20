from transformers import pipeline
from collections import Counter

class ReviewAggregator:
    def __init__(self):
        self.aspect_sentiments = {}   # {aspect: [scores]}
        self.aspect_keywords = {}     # {aspect: Counter()}
        self.summary_pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    def add_aspect_result(self, aspect, sentence, score, keywords):
        self.aspect_sentiments.setdefault(aspect, []).append(score)
        if aspect not in self.aspect_keywords:
            self.aspect_keywords[aspect] = Counter()
        self.aspect_keywords[aspect].update(keywords)
    
    def aggregate_aspects(self):
        """
        Returns per-aspect summary, sentiment, top keywords
        """
        aspect_summary = []
        for aspect in self.aspect_sentiments:
            avg_sentiment = sum(self.aspect_sentiments[aspect]) / len(self.aspect_sentiments[aspect])
            top_keywords = [kw for kw, _ in self.aspect_keywords[aspect].most_common(5)]

            # summarize aspect sentences
            sentences_text = " ".join(self.aspect_sentences[aspect])
            try:
                summary = self.summary_pipe(
                    sentences_text,
                    max_length=30,
                    min_length=10,
                    do_sample=False
                )[0]["summary_text"]
            except Exception:
                summary = sentences_text[:3000] + "..." if len(sentences_text) > 200 else sentences_text

            aspect_summary.append({
                "aspect": aspect,
                "summary": summary,
                "sentiment": round(avg_sentiment, 2),
                "keywords": top_keywords
            })
        return aspect_summary

    def summarize_restaurant(self):
        """
        Summarize the restaurant using the aspect summaries
        """
        aspect_summary_text = " ".join([a["summary"] for a in self.aggregate_aspects()])
        try:
            overall_summary = self.summary_pipe(
                aspect_summary_text,
                max_length=80,
                min_length=20,
                do_sample=False
            )[0]["summary_text"]
        except Exception:
            overall_summary = aspect_summary_text[:3000] + "..." if len(aspect_summary_text) > 100 else aspect_summary_text

        return overall_summary
