import sys
import json
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter

from data_preprocess import data_preprocess
from detect_aspect import DetectAspect
from sentiment import Sentiment
from keyword_extract import KeywordExtractor
from summary import Summary

nltk.download("punkt")


def main():
    review_filepath = sys.argv[1]
    output_filepath = sys.argv[2]

    # Load and preprocess reviews
    reviews = data_preprocess(review_filepath)

    noise_levels = [r.get("ratings", {}).get("Noise level", "").lower() 
                    for r in reviews if r.get("ratings", {}).get("Noise level")]
    most_common_noise = Counter(noise_levels).most_common(1)[0][0] if noise_levels else None


    aspect_detector = DetectAspect()
    sentiment_model = Sentiment()
    keyword_model = KeywordExtractor()
    summary_model = Summary()

    # Store aspect info
    aspect_buckets = {}  # {aspect: {"sentences": [], "sentiments": [], "keywords": []}}

    for r in reviews:
        text = r.get("clean_text", "")
        if not text:
            continue

        sentences = sent_tokenize(text)

        for sent in sentences:
            if not sent.strip():
                continue

            aspects = aspect_detector.detect(sent)
            sent_score = sentiment_model.score_sentence(sent)

            for asp in aspects:
                kw = keyword_model.extract_keywords(sent, asp)

                if asp not in aspect_buckets:
                    aspect_buckets[asp] = {"sentences": [], "sentiments": [], "keywords": []}

                aspect_buckets[asp]["sentences"].append(sent)
                aspect_buckets[asp]["sentiments"].append(sent_score)
                aspect_buckets[asp]["keywords"].extend(kw)

    # Aggregate per aspect
    aspect_summary_list = []
    aspect_summaries_text = []

    for asp, data in aspect_buckets.items():
        if not data["sentences"]:
            continue

        avg_sentiment = sum(data["sentiments"]) / len(data["sentiments"])
        top_keywords = [kw for kw, _ in Counter(data["keywords"]).most_common(5)]

        if asp == "environment" and most_common_noise:
            top_keywords = top_keywords[:4] + [f"Noise Level: {most_common_noise}"]

        # Generate aspect summary
        aspect_summary = summary_model.summarize(data["sentences"])
        aspect_summaries_text.append(aspect_summary)

        aspect_summary_list.append({
            "aspect": asp,
            "sentiment": round(avg_sentiment, 3),
            "keywords": top_keywords
        })


    # Restaurant-level summary (based on aspect summaries)
    restaurant_summary_text = summary_model.summarize(aspect_summaries_text)

    result = {
        "aspects": aspect_summary_list,
        "restaurant_summary": restaurant_summary_text
    }

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()