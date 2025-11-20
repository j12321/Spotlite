import json
import re


def data_preprocess(review_filepath):
    with open(review_filepath, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    pattern = r"(Food|Service|Atmosphere|Noise level)\s*:\s*([^\n]+?)\s*(?=\bFood|\bService|\bAtmosphere|\bNoise level|$)"
        
    for r in reviews:
        text = r.get("text", "") or ""

        matches = re.findall(pattern, text)
        ratings = {key: value.strip() for key, value in matches}

        cleaned_text = re.sub(pattern, "", text)

        cleaned_text = cleaned_text.replace("\n", " ")
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        r["ratings"] = ratings
        r["clean_text"] = cleaned_text
    
    return reviews