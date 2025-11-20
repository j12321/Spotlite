import pandas as pd

class DetectAspect:
    def __init__(self, dish_file="food.csv"):
        # load dishes
        df = pd.read_csv(dish_file, header=None)
        self.dish_list = df[0].astype(str).str.lower().tolist()

        self.food_keyword_list = [
            "food", "flavor", "taste", "dish",
            "spicy", "fresh", "delicious", "portion", "texture"
        ]

        # Aspect keyword dictionary
        self.ASPECT_KEYWORDS = {
            "food": self.dish_list + self.food_keyword_list,
            "service": ["service", "staff", "waiter", "waitress",
                        "employee", "friendly", "attitude",
                        "server", "manager", "rude"],
            "waiting_time": ["wait", "waiting", "slow", "fast",
                             "line", "queue", "minute", "hour",
                             "reservation", "busy"],
            "price": ["price", "expensive", "cheap", "cost",
                      "value", "pricey", "bill", "worth", "dollar"],
            "environment": ["environment", "atmosphere", "clean",
                            "dirty", "noise", "music", "space",
                            "vibe", "bathroom", "parking"]
        }

    def detect(self, sentence: str):
        """
        Detect ALL aspects that appear in a sentence.
        Returns a list of aspects.
        """
        s = sentence.lower()
        found_aspects = []

        for aspect, keywords in self.ASPECT_KEYWORDS.items():
            if any(k in s for k in keywords):
                found_aspects.append(aspect)

        # fallback: if nothing detected, treat as food
        if not found_aspects:
            return ["food"]

        return found_aspects

