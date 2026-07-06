# intelligent-rumor-detection-nlp
Intelligent rumor detection system using BERT &amp; multilingual BERT with LIME explainability, built on the PHEME dataset.

# Intelligent Rumor Detector

## Why Intelligent Rumor Detector

Rumors and misinformation spread rapidly on social media, often faster than they can be manually verified. By the time a claim is fact-checked, it may have already reached thousands of people.

Intelligent Rumor Detector uses NLP and transformer-based models to automatically classify text as rumor or non-rumor in real time, while also explaining *why* it made that decision — something most black-box detection systems fail to do.

Automatically classifying and explaining rumors allows users to quickly assess the credibility of content without needing to manually cross-check every claim, saving time and reducing the spread of misinformation.

## How it Works

1. User enters a piece of text (tweet, claim, or news snippet)
2. Text is cleaned and tokenized
3. A fine-tuned BERT (or multilingual BERT) model classifies it as **Rumor** or **Non-Rumor**
4. LIME generates a word-level explanation showing which terms influenced the prediction
5. Result, confidence score, and explanation are displayed on the website

## App Features

1. **Rumor Classification:** Classifies input text as rumor or non-rumor using a fine-tuned BERT model
2. **Multilingual Support:** Additional multilingual BERT (mBERT) model for detecting rumors in regional languages
3. **Explainable Predictions:** Uses LIME to highlight the specific words that influenced the model's decision, instead of a black-box output
4. **Confidence Score:** Displays how confident the model is in its prediction
5. **Interactive UI:** Built with Streamlit for a simple, clean interface

## Installation

System: Python 3.8+

1. `git clone https://github.com/yourusername/intelligent-rumor-detection-nlp.git`
2. `cd intelligent-rumor-detection-nlp`
3. `pip install -r requirements.txt`
4. Download the trained models from the links below and unzip them into the `app/` folder
5. `streamlit run app/app.py`

## Trained Models

Model weights are too large for GitHub. Download here:
- [rumor_model.zip (BERT)](PASTE_YOUR_GOOGLE_DRIVE_LINK)
- [rumor_model_multilingual.zip (mBERT)](PASTE_YOUR_GOOGLE_DRIVE_LINK)

## App Usage

1. **Enter Text:** Paste the tweet, claim, or news snippet you want to check into the text box
2. **Select Model:** Choose between the English BERT model or the Multilingual BERT model
3. **Click Predict:** The app classifies the text and displays the result
4. **View Result:** See the prediction (Rumor / Non-Rumor) along with a confidence score
5. **View Explanation:** See which words in the text influenced the prediction, highlighted by LIME

## Dataset

Trained on the **PHEME dataset for Rumour Detection and Veracity Classification** (Kochkina, Liakata, Zubiaga) — [figshare](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)

## Results

| Model | Accuracy |
|---|---|
| BERT (English) | ~87.5% |
| mBERT (Multilingual) | ~87.3% |
