# Spotlite Review Analysis Pipeline

This project implements an end-to-end NLP pipeline for analyzing text reviews. It includes preprocessing, aspect detection, sentiment analysis, keyword extraction, and summarization.  


---

## Pipeline Workflow

1. **Generate Food Word Bank**  

   The pipeline requires a `food.csv` file. You can either use the pre-provided `food.csv` or generate it by running:

   ```bash
   python food_word_bank.py

2. **Install Dependencies**

  Install all required packages:

  ```bash
  pip install -r requirements.txt

3. **Run the Main Pipeline**

  Run the main script with an input data file and an output file path:

  ```bash
  python main.py <data_filepath> <output_filepath>

  The pipeline will automatically execute all modules in order:

    Data preprocessing
    
    Aspect detection
    
    Sentiment analysis
    
    Keyword extraction
    
    Summarization
