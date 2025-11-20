import nltk
from nltk.corpus import wordnet
from collections import defaultdict
import csv


# Download necessary NLTK components
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    # Note: 'vader_lexicon', 'punkt', 'stopwords' are not needed for just the word bank

def get_hyponyms(synset, depth=3):
    """
    Recursively fetches all specific terms (hyponyms) below a given synset 
    up to the specified depth.
    """
    hyponym_set = set()
    queue = [(synset, 0)]

    while queue:
        current_synset, current_depth = queue.pop(0)
        
        # Stop at the max depth
        if current_depth > depth: 
            continue

        # Add all lemma names from the current synset
        for lemma in current_synset.lemmas():
            hyponym_set.add(lemma.name().lower().replace('_', ' '))

        # Add the immediate hyponyms to the queue for the next level
        for hyponym in current_synset.hyponyms():
            queue.append((hyponym, current_depth + 1))
            
    return hyponym_set

# WORD BANK GENERATION FUNCTION

def generate_refined_food_word_bank(max_depth_general=4, max_depth_specific=3):
    """
    Generates a comprehensive food word bank by combining:
    1. General 'food' concepts from WordNet (English-centric).
    2. Targeted expansion using international dish synsets.
    3. Manual augmentation for common international dishes (e.g., 'ramen').
    """
    
    # 1. Start with the general food concept
    # food.n.02 refers to edible substances
    food_synset = wordnet.synset('food.n.02')
    food_terms = get_hyponyms(food_synset, depth=max_depth_general)
    
    # 2. Add specific, high-level international concepts
    key_international_synsets = [
        wordnet.synset('pizza.n.01'),
        wordnet.synset("pasta.n.01"),
        wordnet.synset('sushi.n.01'),
        wordnet.synset('noodle.n.01'),
        wordnet.synset('curry.n.01'),
        wordnet.synset('taco.n.01'),
        wordnet.synset('seafood.n.01'),
        wordnet.synset('rice.n.01'),
        wordnet.synset('tortilla.n.01'),
        wordnet.synset('dumpling.n.01'),
        wordnet.synset('cheese.n.01'),
        wordnet.synset('yogurt.n.01'),
        wordnet.synset('bread.n.01'),
        wordnet.synset('cake.n.01'),
        wordnet.synset('pastry.n.01'),
        wordnet.synset("dessert.n.01"),
        wordnet.synset("barbecue.n.01"),
        wordnet.synset("grill.n.01"),
        wordnet.synset("barbecue.n.02"),
        wordnet.synset("soup.n.01"),
        wordnet.synset("salad.n.01"),
        wordnet.synset("sandwich.n.01"),
        wordnet.synset("burger.n.01"),
        wordnet.synset("stew.n.01"),
        wordnet.synset("meat.n.01"),
        wordnet.synset("fish.n.01"),
        wordnet.synset("shellfish.n.01"),
        wordnet.synset("poultry.n.01"),
        wordnet.synset("beverage.n.01")
    ]
    
    for syn in key_international_synsets:
        # Use a shallower depth for specific terms
        food_terms.update(get_hyponyms(syn, depth=max_depth_specific)) 

    # 3. Augment with a custom list of high-frequency international dishes
    # This covers known gaps in WordNet's coverage
    augmentation_list = {
        'ramen', 'pho', 'pad thai', 'kimchi', 'dim sum', 'gyro', 'kebab', 
        'burrito', 'enchilada', 'risotto', 'paella', 'croissant', 'samosa',
        'wagyu', 'tandoori', 'biryani', 'malatang', 'falafel', 'hummus', 
        'shawarma', 'ceviche', 'empanada', 'gelato', 'laksa', 'banh mi', 'mochi', 
        'churro', 'tiramisu', 'pierogi'
    }
    food_terms.update(augmentation_list)
    
    # Remove obvious noise words that sometimes slip into the hierarchy
    noise_words = {'food', 'dish', 'item', 'place', 'eat', 'dinner', 'lunch', 'breakfast', 'meal'}
    food_terms = food_terms - noise_words
    
    return food_terms

# EXECUTION

# Generate the word bank
FOOD_WORD_BANK = generate_refined_food_word_bank()

# Write data to csv
food_word_bank_data = list(FOOD_WORD_BANK)
with open('food.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for item in food_word_bank_data:
        writer.writerow([item])

