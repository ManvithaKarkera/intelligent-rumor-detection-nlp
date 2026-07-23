# Intelligent Rumor Detection System Using NLP Techniques

An automated system to classify social media text and images as **Rumor** or **Non-Rumor**, using fine-tuned transformer models (BERT, multilingual BERT) and a multimodal fusion model (BERT + ResNet50), with explainable predictions (LIME, Grad-CAM) and automatic multilingual routing for Indian regional languages.

---

## Why This Project

Rumors and misinformation spread rapidly on social media, often faster than they can be manually fact-checked. Existing detection systems tend to be:

- Black boxes with no explanation of why they flagged something
- Largely English-only, ignoring India's multilingual reality
- Focused only on text, ignoring image-based misinformation

This project builds a system that addresses all three gaps in one pipeline — **classification, explainability, and multilingual/multimodal support** — with a specific focus on Indian languages and India-relevant content.

---

## How It Works

1. User enters text (and optionally an image)
2. **Script detection** automatically routes Indic-script text (Hindi, Kannada, Tamil, Telugu, Malayalam, Bengali, Punjabi, Gujarati, Odia) to the multilingual model; English routes to the fine-tuned English BERT
3. If both text and image are provided, the **multimodal (BERT + ResNet50)** model runs instead
4. The model classifies the input as **Rumor** or **Non-Rumor**
5. **LIME** (text) and **Grad-CAM** (image) generate explanations showing which words/regions drove the prediction
6. A confidence-based **domain-shift warning** flags predictions that may fall outside the model's training distribution

---

## Features

| Feature | Details |
|---|---|
| Text classification | Fine-tuned BERT (`bert-base-uncased`) on PHEME + IFND datasets |
| Multilingual support | Fine-tuned mBERT (`bert-base-multilingual-cased`), automatic script-based routing, no manual language selection |
| India-specific detection | IFND dataset integration — Hindi fake news, WhatsApp forwards, regional claims |
| Multimodal classification | BERT + ResNet50 fusion trained on Fakeddit (text + image pairs) |
| Text-Only baseline | BERT only, no image — ablation comparison |
| Image-Only baseline | ResNet50 only, no text — ablation comparison |
| Explainability (text) | LIME word-level highlighting + narrative plain-language explanation |
| Explainability (image) | Grad-CAM heatmap overlaid on uploaded image |
| Domain-shift safeguard | Flags low-confidence or out-of-domain inputs instead of silently guessing |
| NSFW protection | Subreddit blocklist + metadata filtering on Fakeddit image downloads |
| Interactive UI | Gradio: live status indicators, animated inference report, tabbed Explainability view |

---

## Results

### PHEME Dataset — Text Models

| Model | Accuracy | F1-Score | Precision | Recall |
|---|---|---|---|---|
| BERT (English, `bert-base-uncased`) | **86.6%** | 80.8% | 79.1% | 82.5% |
| mBERT (Multilingual, `bert-base-multilingual-cased`) | **85.4%** | 78.8% | 77.9% | 79.7% |

### Fakeddit Dataset — Multimodal Ablation

| Model Configuration | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Text-Only (BERT) | 81.0% | 85.2% | 82.1% | 83.6% |
| Image-Only (ResNet50) | 75.0% | 83.7% | 71.4% | 77.1% |
| **Multimodal (BERT + ResNet50)** | **83.6%** | **89.6%** | **81.7%** | **85.4%** |

> The multimodal model consistently outperforms both single-modality baselines, confirming that combining text and image signals improves rumor detection.

---

## Datasets

### PHEME — Rumour Detection and Veracity Classification
- **Source:** Kochkina, E., Liakata, M., Zubiaga, A. — [figshare](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078)
- **Content:** 6,425 tweets across 9 real news events (Charlie Hebdo, Ferguson, Ottawa shooting, Sydney siege, Germanwings crash, Gurlitt, Putin missing, Prince Toronto, Ebola Essien)
- **Used for:** English BERT and multilingual mBERT training

### IFND — Indian Fake News Dataset
- **Source:** Sonal Garg — [Kaggle](https://www.kaggle.com/datasets/sonalgarg174/ifnd-dataset)
- **Content:** Indian fake news statements in English covering domestic politics, health misinformation, WhatsApp forwards, and regional claims
- **Used for:** India-specific fine-tuning and testing
- **Label format:** 0 = Fake, 1 = Real

### Fakeddit — Multimodal Fake News Dataset
- **Source:** Reddit-sourced text + image pairs
- **Content:** 564,000 total records; 5,000 used for training (image download limit)
- **Used for:** Text-Only, Image-Only, and Multimodal (BERT + ResNet50) ablation models
- **Safety:** All image downloads filtered via NSFW metadata column before use

---

## Tech Stack

| Component | Technology |
|---|---|
| Text models | BERT (`bert-base-uncased`), mBERT (`bert-base-multilingual-cased`) |
| Image model | ResNet50 (torchvision, pretrained on ImageNet) |
| Multimodal fusion | BERT + ResNet50 late fusion |
| Explainability | LIME (text), Grad-CAM (image) |
| Frameworks | PyTorch, Hugging Face Transformers, torchvision |
| UI | Gradio |
| Training environment | Kaggle Notebooks (GPU — Tesla T4/P100) |
| Languages supported | English, Hindi, Kannada, Tamil, Telugu, Malayalam, Bengali, Punjabi, Gujarati, Odia |

---

## Installation

**Requirements:** Python 3.8+

```bash
git clone https://github.com/yourusername/intelligent-rumor-detection-nlp.git
cd intelligent-rumor-detection-nlp
pip install -r requirements.txt
```

**Download trained model checkpoints** and place in the project root:

- `rumor_model/`-English BERT checkpoint
- `rumor_model_multilingual/`-mBERT checkpoint
- `multimodal_artifacts/best_multimodal_model.pth`-Multimodal fusion model
- `multimodal_artifacts/best_text_only.pth`-Text-Only baseline
- `multimodal_artifacts/best_image_only.pth`-Image-Only baseline

> Model download links: *(https://drive.google.com/file/d/10pgqhGIgU3joqXr3o-BDoWWAWNUd3iPs/view?usp=drive_link,https://drive.google.com/file/d/10agq9H92d88z5G_5gLIi4tY_I5_aYl3T/view?usp=drive_link)*

---

## Usage

### Option 1 — Run the full notebook (training + inference)
```bash
jupyter notebook rumor-detection-nlp.ipynb
```
Run cells in order: data loading → cleaning → BERT/mBERT training → multimodal training → inference backend → Gradio UI.

### Option 2-Run inference only (models already trained)

Run only the last 2 cells (inference backend + Gradio UI cell). The app launches at a local URL.

### Using the Web Interface

1. Enter any text (English or Indian language) in the text box
2. Optionally upload an image for multimodal analysis
3. Click **Analyze** (or press Enter)
4. View the **Inference Report** — prediction, confidence, probability bars
5. Switch to the **Explainability** tab to see:
   - Plain-language explanation of why the model decided this
   - LIME word highlights (red = pushes toward Rumor, green = toward Non-Rumor)
   - Grad-CAM heatmap (if image was uploaded)

---

## Example Queries

### English
```
BREAKING: New studies suggest eating garlic prevents virus contraction immediately.
Cash withdrawal limit at ATMs will be reduced to ₹2000 per day starting tomorrow, RBI has not confirmed this.
ISRO successfully launches Chandrayaan-4 mission ahead of schedule, confirms official press release.
```

### Hindi
```
कोरोना वैक्सीन लेने से 5 साल के अंदर मौत हो सकती है
व्हाट्सएप पर वायरल: नया आधार कार्ड नियम आज रात से लागू, तुरंत अपडेट करें वरना खाता बंद
```

### Kannada
```
ಭಾರತ ಸರ್ಕಾರ ಇಂದು ಹೊಸ ಶಿಕ್ಷಣ ನೀತಿಯನ್ನು ಘೋಷಿಸಿತು
```

### Tamil
```
ரயில் கட்டணம் நாளை முதல் 50% அதிகரிக்கும் என அரசு அறிவித்தது என்று வதந்தி பரவுகிறது
```

### Telugu
```
రాష్ట్రంలో రేపటి నుంచి అన్ని పాఠశాలలకు వేసవి సెలవులు ప్రకటన
```

### Malayalam
```
ന്യൂഡൽഹി: കേന്ദ്ര സർക്കാർ പുതിയ കാർഷിക നയം പ്രഖ്യാപിച്ചു
```

---

## Multilingual Support — Important Note

The multilingual model was fine-tuned on PHEME (primarily English) and IFND (Indian English). Hindi/Kannada/Tamil/Telugu/Malayalam predictions demonstrate mBERT's **zero-shot cross-lingual transfer** capability-its shared sub-word vocabulary covers 104 languages, allowing reasonable generalization without direct training on labeled regional-language data.

Fine-tuning on dedicated Indian-language datasets (HinFakeNews, Dravidian Fake dataset) is noted as future scope and would significantly improve regional language accuracy.

---

## Project Structure

```
intelligent-rumor-detection-nlp/
│
├── rumor-detection-nlp.ipynb     ← Main notebook (32 cells)
├── app.py                        ← Gradio web app (for HF Spaces deployment)
├── requirements.txt
├── README.md
│
├── rumor_model/                  ← English BERT checkpoint (after training)
├── rumor_model_multilingual/     ← mBERT checkpoint (after training)
└── multimodal_artifacts/
    ├── best_multimodal_model.pth
    ├── best_text_only.pth
    └── best_image_only.pth
```

---

## Notebook Structure (32 Cells)

| Cells | Content |
|---|---|
| 1–3 | Imports, PHEME dataset loading, text cleaning |
| 4–8 | Tokenization, dataset preparation, BERT training |
| 9–13 | BERT evaluation, model saving |
| 14–18 | mBERT training and evaluation |
| 19–21 | IFND dataset loading and India-specific fine-tuning |
| 22–24 | Fakeddit loading, NSFW filtering, image downloading |
| 25–27 | Multimodal model architecture, training, ablation evaluation |
| 28–30 | Inference backend (run_inference, LIME, Grad-CAM) |
| 31–32 | Gradio UI |

---

## Future Scope

- Fine-tuning on labeled Indian-language rumor datasets (HinFakeNews, Dravidian Fake, BanFakeNews)
- Real-time social media integration (Twitter/X API streaming)
- Claim verification against fact-checking databases (BOOM, AltNews, FactChecker.in)
- Propagation-based signals (retweet patterns, user credibility scores)
- Lightweight mobile-friendly model (DistilBERT) for low-resource deployment

---


