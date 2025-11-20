import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class KeywordExtractor:
    def __init__(self, top_k=5, dish_file="food.csv"):
        self.nlp = spacy.load("en_core_web_sm")
        self.top_k = top_k

        df = pd.read_csv(dish_file, header=None)
        self.dish_list = df[0].astype(str).str.lower().tolist()
        self.food_keywords = ["food", "flavor", "taste", "dish", "spicy", "fresh", "delicious", "portion", "texture"]

        self.aspect_keywords = {
            "food": self.dish_list + self.food_keywords,
            "service": ["service", "staff", "waiter", "waitress", "employee", "friendly", "attitude", "server", "manager", "rude"],
            "waiting_time": ["wait", "waiting", "slow", "fast", "line", "queue", "minute", "hour", "reservation", "busy"],
            "price": ["price", "expensive", "cheap", "cost", "value", "pricey", "bill", "worth", "dollar"],
            "environment": ["environment", "atmosphere", "clean", "dirty", "noise", "music", "space", "vibe", "bathroom", "parking"]
        }

        self.aspect_blocklist = {
            "food": ["the", "service", "price"],  
            "service": ["the", "manager", "employee"],
            "waiting_time": ["the", "noise", "waiter", "waitress"],
            "price": ["the"], 
            "environment": ["the"]
        }
        self.vectorizer = None
        self.tfidf_vocab = None

    def build_tfidf(self, all_texts):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,5))
        self.vectorizer.fit(all_texts)
        self.tfidf_vocab = self.vectorizer.vocabulary_

    def _tfidf_score(self, phrase, aspect):
        """Compute average TF-IDF score of the phrase"""
        if aspect == "price" and "$" in phrase:
            return 10.0
        if not self.vectorizer:
            return 1.0  # fallback if TF-IDF not built

        tokens = phrase.lower().split()
        scores = []
        for t in tokens:
            if t in self.tfidf_vocab:
                scores.append(self.vectorizer.idf_[self.tfidf_vocab[t]])
        return sum(scores) / len(scores) if scores else 0.0
    
    def extract_keywords(self, sentence: str, aspect: str):
        doc = self.nlp(sentence.lower())
        candidates = set()

        # Noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:
                candidates.add(chunk.text)

        tokens = [t for t in doc if not t.is_punct and not t.is_stop]
        for i in range(len(tokens)-1):
            # Adj + Noun
            if tokens[i].pos_ in {"ADJ"} and tokens[i+1].pos_ in {"NOUN", "PROPN"}:
                candidates.add(f"{tokens[i].text} {tokens[i+1].text}")
            # Noun + Noun
            if tokens[i].pos_ in {"NOUN", "PROPN"} and tokens[i+1].pos_ in {"NOUN", "PROPN"}:
                candidates.add(f"{tokens[i].text} {tokens[i+1].text}")

        for i in range(len(tokens)-2):
            # Adj + N + N, N + N + N
            if tokens[i].pos_ in {"ADJ", "NOUN", "PROPN"} and tokens[i+1].pos_ in {"NOUN", "PROPN"} and tokens[i+2].pos_ in {"NOUN", "PROPN"}:
                candidates.add(f"{tokens[i].text} {tokens[i+1].text} {tokens[i+2].text}")
            # Adjective + Adjective + Noun
            if tokens[i].pos_ == "ADJ" and tokens[i+1].pos_ == "ADJ" and tokens[i+2].pos_ in {"NOUN", "PROPN"}:
                candidates.add(f"{tokens[i].text} {tokens[i+1].text} {tokens[i+2].text}")

        if aspect == "price":
            # e.g. "$30–50", "$20", "$$", "$15"
            price_matches = re.findall(r"\$+\s*\d+(?:[-–]\d+)?", sentence)
            for pm in price_matches:
                candidates.add(pm.strip())

        # Remove stopwords
        cleaned_candidates = []
        for c in candidates:
            words = c.split()
            if not all(self.nlp.vocab[w].is_stop for w in words):
                cleaned_candidates.append(c)
        candidates = cleaned_candidates

        if aspect != "price":
            candidates = [c for c in candidates if "$" not in c]

        block_words = self.aspect_blocklist.get(aspect, [])
        filtered_candidates = []
        for c in candidates:
            if not any(bw in c.lower().split() for bw in block_words):
                filtered_candidates.append(c)
        candidates = filtered_candidates


        asp_kws = self.aspect_keywords.get(aspect, [])
        aspect_filtered = []
        for c in candidates:
            if aspect == "price" and "$" in c:
                aspect_filtered.append(c)
                continue
            for kw in asp_kws:
                if kw in c.lower():
                    aspect_filtered.append(c)
                    break

        if not aspect_filtered:
            return []

        # Score by global TF-IDF
        scored = [(c, self._tfidf_score(c, aspect)) for c in aspect_filtered]
        scored = [c for c in scored if c[1] > 0]  # remove zero-score noise
        
        if not scored:
            return []

        # Select top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:self.top_k]]
