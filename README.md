# 🛡️ Intelligent Multimodal Rumor Detection System

Rumors spread faster than fact-checkers can keep up. This project uses transformer-based NLP and CNN-based computer vision to automatically classify social media claims — text alone, images alone, or text+image together — as **Rumor** or **Non-Rumor**, and explains *why* for every prediction instead of returning a black-box label.

---

## Why this exists

By the time a viral claim gets manually fact-checked, it's often already reached thousands of people. This system classifies content in real time and pairs every prediction with a visual explanation (which words mattered, which parts of the image mattered), so a user can sanity-check the model's reasoning instead of blindly trusting a label.

---

## What's inside

The project has two stages, both in the notebook (`rumor-detection-nlp-added_multimodal.ipynb`):

### 1. Text-only rumor detection (PHEME dataset)
- Fine-tuned **BERT** (`bert-base-uncased`) binary classifier (Rumor / Non-Rumor)
- Fine-tuned **mBERT** (`bert-base-multilingual-cased`) for regional/multilingual text
- Trained and evaluated on the [PHEME dataset](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078) (Kochkina, Liakata, Zubiaga)
- **LIME** explanations highlighting which words pushed the prediction toward Rumor/Non-Rumor

### 2. Multimodal extension (Fakeddit dataset)
Extends the project with an image modality and a proper ablation study, trained on **Fakeddit** (title + image pairs):

| Model | Architecture | Purpose |
|---|---|---|
| `TextOnlyRumorDetector` | BERT-base → 256-d projection → MLP classifier | Text-only baseline |
| `ImageOnlyRumorDetector` | ResNet50 (ImageNet pretrained) → 256-d projection → MLP classifier | Image-only baseline |
| `MultimodalRumorDetector` | BERT + ResNet50, fused via concatenation + element-wise product (768-d) → MLP classifier | Joint text+image model |

**Why BERT + ResNet50 instead of CLIP or BERT+ViT:** the notebook includes a documented architecture comparison. ResNet50's convolutional layers give stable, direct Grad-CAM gradients, it reuses the same BERT text encoder as the text-only baseline (so the ablation comparison is apples-to-apples), and it fits comfortably on a single Kaggle T4 GPU.

**Training setup:** backbones frozen initially (transfer learning), AdamW optimizer, `ReduceLROnPlateau` scheduler, early stopping on validation loss, best checkpoint saved per model.

**Explainability:**
- **LIME** (text) — word-level contribution scores, rendered as HTML
- **Grad-CAM** (image) — custom hook-based implementation on `resnet.layer4`, heatmap overlaid on the original image, with a dynamic layer-detection helper so it isn't hardcoded to one architecture

**Evaluation:** accuracy, precision, recall, F1, confusion matrix, ROC/PR curves for all three ablation models side by side, so you can see exactly how much the image modality adds over text alone.

---

## Deployment: interactive Gradio app

The interface is a **Gradio** app (`gr.Blocks`), launched directly from the final notebook cell — there is no separate Streamlit app or `app/` folder; everything runs from the notebook.

**What it does:**
- Accepts text only, image only, or both
- Routes to the appropriate model (text-only / image-only / multimodal) automatically based on what's provided
- Returns prediction, confidence score, which model was used, processing time
- Shows an ablation comparison (how the unimodal models would have scored the same input)
- Renders LIME text explanations and Grad-CAM image overlays side by side

To run it yourself (see [Installation](#installation) and [Usage](#usage) below), execute the notebook end-to-end; the last cell calls `demo.launch(share=True)` and gives you a public URL.

> If you want a persistent, non-notebook deployment (e.g. on Hugging Face Spaces or a server), see [Deploying outside the notebook](#deploying-outside-the-notebook) below.

---

## Installation

**Requirements:** Python 3.8+, a CUDA-capable GPU recommended (CPU works but is slow for BERT/ResNet inference).

```bash
git clone https://github.com/yourusername/intelligent-rumor-detection-nlp.git
cd intelligent-rumor-detection-nlp
pip install -r requirements.txt
```

Or, matching what the notebook installs directly:

```bash
pip install transformers datasets scikit-learn lime torchvision gradio opencv-python pillow matplotlib seaborn
```

**Trained model weights** (too large for GitHub — host these on Hugging Face Hub, Google Drive, or Kaggle Datasets and link them here):
- `rumor_model.zip` — text-only BERT (PHEME)
- `rumor_model_multilingual.zip` — mBERT (PHEME)
- `multimodal_artifacts/best_text_only.pth`
- `multimodal_artifacts/best_image_only.pth`
- `multimodal_artifacts/best_multimodal_model.pth`

Unzip/place these so the paths in the notebook's `load_saved_model()` calls resolve correctly (by default it expects a `multimodal_artifacts/` folder alongside the notebook).

---

## Usage

1. Open and run `rumor-detection-nlp-added_multimodal.ipynb` top to bottom (or just the model-loading + Gradio cells if you already have checkpoints saved).
2. The last cell launches the Gradio app and prints a local + public (`share=True`) URL.
3. In the app:
   - Paste a claim/headline into the text box, and/or upload an image
   - Click **Analyze Content**
   - Review the prediction, confidence score, ablation comparison, and the LIME/Grad-CAM explanation panels

---

## Results

| Model | Dataset | Accuracy |
|---|---|---|
| BERT (English, text-only) | PHEME | ~87.5% |
| mBERT (multilingual, text-only) | PHEME | ~87.3% |
| Text-only / Image-only / Multimodal | Fakeddit | see the notebook's ablation table (Section 5) for exact per-model Accuracy/Precision/Recall/F1 from your latest training run |

The multimodal ablation study in the notebook regenerates this table (and confusion matrices / ROC curves) every time you retrain, so treat the numbers above as a snapshot rather than a fixed benchmark.

---

## ⚠️ Scope and limitations

This is a **pattern-based classifier**, not a fact-checker. It has no world knowledge and cannot verify whether a claim is objectively true — it only recognizes linguistic and visual patterns that correlated with "rumor" labels in its training data (PHEME event-based tweets, Fakeddit Reddit posts).

Practical consequences:
- **Out-of-domain text will be classified confidently but unreliably.** A factual, uncontroversial statement about a topic outside the training distribution (e.g. current office-holders, recent events, topics/regions not represented in PHEME or Fakeddit) may be labeled "Rumor" purely because of sentence structure, not content accuracy.
- There is no built-in "I don't know" — the model always forces a Rumor/Non-Rumor decision with a confidence score, even on nonsense or unrelated input.
- Use the LIME/Grad-CAM panels to sanity-check *why* a prediction was made before trusting it, especially for content stylistically or topically different from PHEME/Fakeddit.
- If you extend this project, consider adding a confidence-threshold "uncertain / out-of-scope" fallback rather than presenting every prediction with equal authority.

---

## Deploying outside the notebook

To turn this into a standalone service instead of a notebook cell:
1. Extract the model-loading, inference (`run_inference`/`gradio_predict`), and Gradio UI code from the final cells into a standalone `app.py`.
2. Load model checkpoints from a fixed local/relative path (not Kaggle-specific paths like `/kaggle/working`).
3. Replace `demo.launch(share=True)` with `demo.launch(server_name="0.0.0.0", server_port=7860)` for containerized/server deployment, or deploy directly to a [Hugging Face Space](https://huggingface.co/spaces) (Gradio SDK) for a free public demo.
4. If deploying via Docker, bundle model weights in the image or download them at container startup from wherever you host them (Hugging Face Hub, S3, etc.).

---

## Project structure
