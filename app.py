import os
import re
import time

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    BertModel,
)
from huggingface_hub import snapshot_download
from lime.lime_text import LimeTextExplainer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Intelligent Rumor Detection",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

HF_REPO_ID = "ManvithaKarkera/rumor-detection-models"

CLASS_NAMES = ["Non-Rumor", "Rumor"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DOWNLOAD MODELS FROM HUGGING FACE
# ============================================================

@st.cache_resource
def download_model_repository():
    hf_token = os.getenv("HF_TOKEN")

    kwargs = {
        "repo_id": HF_REPO_ID,
        "repo_type": "model",
    }

    if hf_token:
        kwargs["token"] = hf_token

    return snapshot_download(**kwargs)


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

class ImageOnlyRumorDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        resnet = models.resnet50(weights=None)

        self.image_encoder = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        self.image_proj = nn.Linear(2048, 256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, image):
        features = self.image_encoder(image)
        features = features.view(features.size(0), -1)
        x = self.image_proj(features)
        return self.classifier(x)


class MultimodalRumorDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.text_encoder = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        self.text_proj = nn.Linear(768, 256)

        resnet = models.resnet50(weights=None)

        self.image_encoder = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        self.image_proj = nn.Linear(2048, 256)

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask, image):
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_embedding = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_proj(cls_embedding)

        image_features = self.image_encoder(image)
        image_features = image_features.view(
            image_features.size(0), -1
        )
        image_features = self.image_proj(image_features)

        prod_features = text_features * image_features

        fused = torch.cat(
            [
                text_features,
                image_features,
                prod_features,
            ],
            dim=-1,
        )

        return self.classifier(fused)


# ============================================================
# LOAD ALL MODELS
# ============================================================

@st.cache_resource
def load_all_models():
    model_dir = download_model_repository()

    # BERT
    text_path = os.path.join(model_dir, "rumor_model")

    rumor_tokenizer = BertTokenizerFast.from_pretrained(text_path)
    rumor_model = BertForSequenceClassification.from_pretrained(text_path)
    rumor_model.to(DEVICE)
    rumor_model.eval()

    # mBERT
    multi_path = os.path.join(
        model_dir, "rumor_model_multilingual"
    )

    multi_tokenizer = BertTokenizerFast.from_pretrained(
        multi_path
    )
    multi_model = BertForSequenceClassification.from_pretrained(
        multi_path
    )
    multi_model.to(DEVICE)
    multi_model.eval()

    # Image-only
    image_checkpoint = os.path.join(
        model_dir,
        "multimodal_artifacts",
        "best_image_only.pth",
    )

    image_model = ImageOnlyRumorDetector(num_classes=2)

    image_state = torch.load(
        image_checkpoint,
        map_location=DEVICE,
    )

    image_model.load_state_dict(image_state)
    image_model.to(DEVICE)
    image_model.eval()

    # Multimodal
    multimodal_checkpoint = os.path.join(
        model_dir,
        "multimodal_artifacts",
        "best_multimodal_model.pth",
    )

    multimodal_model = MultimodalRumorDetector(num_classes=2)

    multimodal_state = torch.load(
        multimodal_checkpoint,
        map_location=DEVICE,
    )

    multimodal_model.load_state_dict(multimodal_state)
    multimodal_model.to(DEVICE)
    multimodal_model.eval()

    return (
        rumor_tokenizer,
        rumor_model,
        multi_tokenizer,
        multi_model,
        image_model,
        multimodal_model,
    )


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# LANGUAGE DETECTION
# ============================================================

INDIC_SCRIPT_PATTERN = re.compile(
    "[\u0900-\u097F"
    "\u0980-\u09FF"
    "\u0A00-\u0A7F"
    "\u0A80-\u0AFF"
    "\u0B00-\u0B7F"
    "\u0B80-\u0BFF"
    "\u0C00-\u0C7F"
    "\u0C80-\u0CFF"
    "\u0D00-\u0D7F]"
)


def is_indic_script(text):
    return bool(INDIC_SCRIPT_PATTERN.search(text))


# ============================================================
# TEXT PREDICTION
# ============================================================

def predict_text_proba(texts, tokenizer, model):
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )

    enc = {
        key: value.to(DEVICE)
        for key, value in enc.items()
    }

    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=1).cpu().numpy()

    return probs


# ============================================================
# IMAGE CONVERSION
# ============================================================

def to_pil_rgb(input_image):
    if isinstance(input_image, Image.Image):
        return input_image.convert("RGB")

    arr = np.array(input_image)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]

    return Image.fromarray(arr.astype("uint8"), "RGB")


# ============================================================
# GRAD-CAM
# ============================================================

class ResNetGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def forward_hook(module, input_data, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(
            backward_hook
        )

    def generate_heatmap(self, image_tensor, class_idx=None):
        self.model.eval()

        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        image_tensor.requires_grad = True

        logits = self.model(image_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=-1).item()

        score = logits[0, class_idx]

        self.model.zero_grad()
        score.backward()

        gradients = (
            self.gradients.detach().cpu().numpy()[0]
        )

        activations = (
            self.activations.detach().cpu().numpy()[0]
        )

        weights = np.mean(gradients, axis=(1, 2))

        heatmap = np.zeros(
            activations.shape[1:],
            dtype=np.float32,
        )

        for i, weight in enumerate(weights):
            heatmap += weight * activations[i]

        heatmap = np.maximum(heatmap, 0)

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap


def get_gradcam_target_layer(model):
    for child in reversed(
        list(model.image_encoder.children())
    ):
        if isinstance(child, nn.Sequential):
            return child[-1]

    raise ValueError(
        "Could not find Grad-CAM target layer."
    )


def create_gradcam_image(model, image, class_idx):
    import cv2

    image_tensor = image_transform(image)

    target_layer = get_gradcam_target_layer(model)

    gradcam = ResNetGradCAM(
        model,
        target_layer,
    )

    heatmap = gradcam.generate_heatmap(
        image_tensor,
        class_idx,
    )

    original = np.array(
        image.resize((224, 224))
    )

    heatmap_resized = cv2.resize(
        heatmap,
        (224, 224),
    )

    heatmap_colored = np.uint8(
        255 * heatmap_resized
    )

    heatmap_colored = cv2.applyColorMap(
        heatmap_colored,
        cv2.COLORMAP_JET,
    )

    original_bgr = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR,
    )

    overlayed = cv2.addWeighted(
        original_bgr,
        0.6,
        heatmap_colored,
        0.4,
        0,
    )

    overlayed_rgb = cv2.cvtColor(
        overlayed,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(overlayed_rgb)


# ============================================================
# LIME
# ============================================================

lime_explainer = LimeTextExplainer(
    class_names=CLASS_NAMES,
    split_expression=r"\s+",
    bow=False,
)


def render_lime_html(word_scores):
    if not word_scores:
        return ""

    max_abs = max(
        abs(score)
        for _, score in word_scores
    ) or 1.0

    html_parts = [
        '<div style="line-height:2.2;">'
    ]

    for word, score in word_scores:
        intensity = min(
            abs(score) / max_abs,
            1.0,
        )

        if score > 0:
            background = (
                "rgba(220,38,38,"
                f"{max(0.08, intensity * 0.45):.2f})"
            )
        else:
            background = (
                "rgba(8,127,121,"
                f"{max(0.08, intensity * 0.45):.2f})"
            )

        safe_word = (
            str(word)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        html_parts.append(
            f"""
            <span class="word-pill"
                  style="
                    background:{background};
                    border:1px solid rgba(148,163,184,0.25);
                  ">
                {safe_word}
            </span>
            """
        )

    html_parts.append("</div>")

    return "".join(html_parts)


def generate_plain_explanation(
    word_scores,
    prediction_label,
    confidence,
):
    if not word_scores:
        return (
            "No significant words were identified "
            "for this prediction."
        )

    sorted_scores = sorted(
        word_scores,
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    top_word, top_score = sorted_scores[0]

    explanation = (
        f"The model leaned toward **{prediction_label}** "
        f"primarily because of the word **{top_word}**, "
        "which had the strongest influence on this prediction. "
    )

    if confidence >= 0.85:
        explanation += (
            "Overall, the model is highly confident "
            "in this classification."
        )
    elif confidence >= 0.65:
        explanation += (
            "Overall, the model is reasonably confident, "
            "though some ambiguity remains."
        )
    else:
        explanation += (
            "Overall, the model's confidence is relatively "
            "low, so this result should be treated cautiously."
        )

    return explanation


# ============================================================
# DOMAIN SHIFT WARNING
# ============================================================

def domain_shift_warning(text, confidence):
    messages = []

    if confidence < 0.65:
        messages.append(
            "Low model confidence. The input may differ "
            "from the training domain."
        )

    if len(text.split()) < 4:
        messages.append(
            "Very short input. LIME may provide a less "
            "reliable explanation."
        )

    return " ".join(messages)


# ============================================================
# MAIN INFERENCE
# ============================================================

def run_inference(text, uploaded_image, models_data):
    (
        rumor_tokenizer,
        rumor_model,
        multi_tokenizer,
        multi_model,
        image_model,
        multimodal_model,
    ) = models_data

    has_text = (
        text is not None
        and len(text.strip()) > 0
    )

    has_image = uploaded_image is not None

    if not has_text and not has_image:
        return None

    start_time = time.time()

    # TEXT ONLY
    if has_text and not has_image:
        text = text.strip()

        if is_indic_script(text):
            tokenizer = multi_tokenizer
            model = multi_model
            model_name = "Multilingual BERT (mBERT)"
        else:
            tokenizer = rumor_tokenizer
            model = rumor_model
            model_name = "Fine-tuned BERT"

        probs = predict_text_proba(
            [text],
            tokenizer,
            model,
        )[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        explanation = lime_explainer.explain_instance(
            text,
            classifier_fn=lambda x:
                predict_text_proba(
                    x,
                    tokenizer,
                    model,
                ),
            labels=[pred_idx],
            num_features=6,
            num_samples=500,
        )

        word_scores = explanation.as_list(
            label=pred_idx
        )

        return {
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": confidence,
            "model": model_name,
            "processing_time": (
                time.time() - start_time
            ) * 1000,
            "non_rumor": float(probs[0]),
            "rumor": float(probs[1]),
            "lime_html": render_lime_html(word_scores),
            "plain_explanation": generate_plain_explanation(
                word_scores,
                CLASS_NAMES[pred_idx],
                confidence,
            ),
            "gradcam": None,
            "domain_warning": domain_shift_warning(
                text,
                confidence,
            ),
        }

    # IMAGE ONLY
    if has_image and not has_text:
        image = to_pil_rgb(uploaded_image)

        image_tensor = image_transform(image)

        with torch.no_grad():
            logits = image_model(
                image_tensor.unsqueeze(0).to(DEVICE)
            )

            probs = F.softmax(
                logits,
                dim=1,
            ).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        gradcam_image = create_gradcam_image(
            image_model,
            image,
            pred_idx,
        )

        return {
            "prediction": (
                f"{CLASS_NAMES[pred_idx]} (Image Only)"
            ),
            "confidence": confidence,
            "model": "Image-Only (ResNet50)",
            "processing_time": (
                time.time() - start_time
            ) * 1000,
            "non_rumor": float(probs[0]),
            "rumor": float(probs[1]),
            "lime_html": None,
            "plain_explanation": (
                "No text was provided, so no text-based "
                "explanation is available."
            ),
            "gradcam": gradcam_image,
            "domain_warning": (
                "Rumor detection is generally more reliable "
                "when textual context accompanies the image."
            ),
        }

    # TEXT + IMAGE
    text = text.strip()
    image = to_pil_rgb(uploaded_image)
    image_tensor = image_transform(image)

    tokens = rumor_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    tokens = {
        key: value.to(DEVICE)
        for key, value in tokens.items()
    }

    with torch.no_grad():
        logits = multimodal_model(
            tokens["input_ids"],
            tokens["attention_mask"],
            image_tensor.unsqueeze(0).to(DEVICE),
        )

        probs = F.softmax(
            logits,
            dim=1,
        ).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    explanation = lime_explainer.explain_instance(
        text,
        classifier_fn=lambda x:
            predict_text_proba(
                x,
                rumor_tokenizer,
                rumor_model,
            ),
        labels=[pred_idx],
        num_features=6,
        num_samples=500,
    )

    word_scores = explanation.as_list(
        label=pred_idx
    )

    gradcam_image = create_gradcam_image(
        image_model,
        image,
        pred_idx,
    )

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "model": "Multimodal (BERT + ResNet50)",
        "processing_time": (
            time.time() - start_time
        ) * 1000,
        "non_rumor": float(probs[0]),
        "rumor": float(probs[1]),
        "lime_html": render_lime_html(word_scores),
        "plain_explanation": generate_plain_explanation(
            word_scores,
            CLASS_NAMES[pred_idx],
            confidence,
        ),
        "gradcam": gradcam_image,
        "domain_warning": domain_shift_warning(
            text,
            confidence,
        ),
    }


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #eef2f7;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    .app-header {
        background: linear-gradient(
            100deg,
            #078b87 0%,
            #405bd1 70%,
            #5b57ed 100%
        );
        color: white;
        padding: 13px 20px;
        border-radius: 10px 10px 0 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 15px;
        font-weight: 700;
    }

    .header-title {
        display: flex;
        align-items: center;
        gap: 7px;
    }

    .shield {
        font-size: 17px;
    }

    .language-pills {
        display: flex;
        gap: 5px;
        flex-wrap: wrap;
    }

    .language-pill {
        border: 1px solid rgba(255,255,255,0.45);
        background: rgba(255,255,255,0.14);
        border-radius: 12px;
        padding: 3px 8px;
        font-size: 9px;
        font-weight: 500;
    }

    .status-container {
        background: white;
        border: 1px solid #d8e0ea;
        padding: 9px;
        margin-top: 22px;
        border-radius: 7px;
        display: flex;
        justify-content: center;
        gap: 7px;
        flex-wrap: wrap;
    }

    .status-pill {
        border: 1px solid #dce3eb;
        background: #f8fafc;
        border-radius: 6px;
        padding: 4px 9px;
        font-size: 9px;
        color: #334155;
    }

    .status-dot {
        color: #08a86b;
        font-size: 10px;
    }

    .section-title {
        font-size: 12px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }

    .section-subtitle {
        font-size: 8px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }

    .main-card,
    .result-card,
    .explanation-card {
        background: white;
        border: 1px solid #d9e2ec;
        border-radius: 10px;
        padding: 14px;
        box-shadow: 0 1px 3px rgba(15,23,42,0.05);
    }

    .result-card {
        min-height: 330px;
    }

    .inner-card {
        background: #f8fafc;
        border: 1px solid #d9e2ec;
        border-radius: 9px;
        padding: 12px;
        margin-bottom: 10px;
    }

    .live-badge {
        float: right;
        background: #ecfdf5;
        border: 1px solid #86efac;
        color: #059669;
        border-radius: 10px;
        padding: 3px 9px;
        font-size: 8px;
        font-weight: 600;
    }

    .result-title {
        font-size: 13px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 15px;
    }

    .metric-box {
        background: #f8fafc;
        border: 1px solid #d9e2ec;
        border-radius: 7px;
        padding: 9px 10px;
        min-height: 60px;
    }

    .metric-label {
        font-size: 8px;
        color: #64748b;
        text-transform: uppercase;
    }

    .metric-value {
        font-size: 15px;
        font-weight: 700;
        margin-top: 4px;
        color: #111827;
    }

    .rumor-value {
        color: #dc2626;
    }

    .normal-value {
        color: #087f79;
    }

    .prob-title {
        font-size: 8px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-top: 15px;
        margin-bottom: 8px;
    }

    .prob-row {
        margin-bottom: 8px;
    }

    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 9px;
        color: #334155;
        margin-bottom: 3px;
    }

    .prob-track {
        height: 6px;
        background: #dce3eb;
        border-radius: 5px;
        overflow: hidden;
    }

    .prob-rumor {
        height: 100%;
        background: #dc2626;
        border-radius: 5px;
    }

    .prob-normal {
        height: 100%;
        background: #087f79;
        border-radius: 5px;
    }

    .domain-ok {
        background: #f0fdf4;
        border: 1px solid #86efac;
        color: #15803d;
        border-radius: 6px;
        padding: 7px 9px;
        font-size: 8px;
        margin-top: 10px;
    }

    .domain-warning {
        background: #fff7ed;
        border: 1px solid #fdba74;
        color: #c2410c;
        border-radius: 6px;
        padding: 7px 9px;
        font-size: 8px;
        margin-top: 10px;
    }

    .word-pill {
        display: inline-block;
        padding: 4px 8px;
        margin: 3px;
        border-radius: 5px;
        font-size: 11px;
        color: #1e293b;
    }

    div.stButton > button {
        border-radius: 6px;
        border: 1px solid #d1d9e2;
        font-size: 11px;
        font-weight: 600;
        min-height: 38px;
    }

    div.stButton > button[kind="primary"] {
        background: #14b8a6;
        color: white;
        border: none;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #0d9488;
        color: white;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 8px;
    }

    #MainMenu,
    footer,
    header {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_text" not in st.session_state:
    st.session_state.last_text = ""

if "run_example" not in st.session_state:
    st.session_state.run_example = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <div class="header-title">
            <span class="shield">🛡️</span>
            <span>Intelligent Rumor Detection System</span>
        </div>

        <div class="language-pills">
            <span class="language-pill">EN EN</span>
            <span class="language-pill">हि HI</span>
            <span class="language-pill">ಕಂ KN</span>
            <span class="language-pill">த TA</span>
            <span class="language-pill">మా TE</span>
            <span class="language-pill">മല ML</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODELS
# ============================================================

with st.spinner("Loading trained models..."):
    try:
        models_data = load_all_models()
        models_loaded = models_data is not None
    except Exception as e:
        models_data = None
        models_loaded = False

st.markdown(
    """
    <div class="status-container">
        <span class="status-pill">
            <span class="status-dot">●</span>
            English BERT: Ready
        </span>
        <span class="status-pill">
            <span class="status-dot">●</span>
            Multilingual (mBERT): Ready
        </span>
        <span class="status-pill">
            <span class="status-dot">●</span>
            Image (ResNet50): Ready
        </span>
        <span class="status-pill">
            <span class="status-dot">●</span>
            Multimodal Fusion: Ready
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


if not models_loaded:
    st.error(
        "The trained models could not be loaded. "
        "Check the Streamlit deployment logs and make sure "
        "the Hugging Face repository is public and contains "
        "the four uploaded model artifacts."
    )
    st.stop()


# ============================================================
# TABS
# ============================================================

tab_analyze, tab_explain = st.tabs(
    ["Analyze Content", "Explainability"]
)


# ============================================================
# EXAMPLES
# ============================================================

english_examples = [
    "BREAKING: New studies suggest eating garlic prevents virus completely.",
    "Government launches new education policy changes for secondary schools.",
    "Cash withdrawal limit at ATMs will be reduced to ₹2000 per day.",
    "ISRO successfully launches Chandrayaan-4 mission ahead of schedule.",
]

indian_examples = [
    "कोरोना वैक्सीन लेने से 5 साल के अंदर मौत हो सकती है",
    "व्हाट्सएप पर वायरल: नया आधार कार्ड नियम आज से लागू",
    "ಎಲ್ಲಾ ಬ್ಯಾಂಕುಗಳು ಇಂದಿನಿಂದ ಹೊಸ ನಿಯಮ ಜಾರಿಗೆ ತರಲಿವೆ",
    "தமிழ்நாட்டில் நாளை முதல் பள்ளிகளுக்கு புதிய விதிமுறைகள்",
    "ആശുപത്രിയിൽ കൊവിഡ് മരുന്നുകൾ സൗജന്യമായി നൽകും",
]


# ============================================================
# ANALYZE TAB
# ============================================================

with tab_analyze:
    left_col, right_col = st.columns(
        [1.03, 1],
        gap="medium",
    )

    with left_col:
        st.markdown(
            """
            <div class="main-card">
                <div class="inner-card">
                    <div class="section-title">
                        📊 Input Analysis Canvas
                    </div>
                    <div class="section-subtitle">
                        Content Payload
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        lang_cols = st.columns(6)

        for col, lang in zip(
            lang_cols,
            ["EN", "HI", "KN", "TA", "TE", "ML"],
        ):
            with col:
                st.markdown(
                    f"""
                    <div style="
                        text-align:center;
                        border:1px solid #cbd5e1;
                        background:#f8fafc;
                        border-radius:5px;
                        padding:4px;
                        font-size:9px;
                        color:#334155;
                    ">
                        {lang}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        text_input = st.text_area(
            "Content",
            value=st.session_state.last_text,
            height=115,
            label_visibility="collapsed",
            placeholder=(
                "Paste rumor text, social media headlines, "
                "or claims here. Multi-script detection "
                "(Hindi, Kannada, etc.) is active..."
            ),
            key="text_input",
        )

        st.markdown(
            """
            <div class="section-subtitle"
                 style="margin-top:8px;">
                Multimodal Context (Optional)
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_image = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="uploaded_image",
        )

        if uploaded_image is not None:
            try:
                preview = Image.open(uploaded_image).convert("RGB")
                st.image(
                    preview,
                    use_container_width=True,
                )
            except Exception:
                st.error("Could not read the uploaded image.")
        else:
            st.markdown(
                """
                <div style="
                    height:150px;
                    border:1px solid #cbd5e1;
                    border-radius:6px;
                    background:white;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    color:#94a3b8;
                    font-size:10px;
                ">
                    Upload an image for multimodal analysis
                </div>
                """,
                unsafe_allow_html=True,
            )

        clear_col, analyze_col = st.columns(2)

        with clear_col:
            clear_button = st.button(
                "🗑 Clear",
                use_container_width=True,
            )

        with analyze_col:
            analyze_button = st.button(
                "⚡ Analyze",
                type="primary",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # RIGHT — INFERENCE REPORT
    # ========================================================

    with right_col:
        result = st.session_state.last_result

        st.markdown(
            """
            <div class="result-card">
                <span class="live-badge">● LIVE</span>
                <div class="result-title">
                    Inference Report
                </div>
            """,
            unsafe_allow_html=True,
        )

        if result is None:
            prediction = "—"
            confidence = "—"
            model_name = "Waiting for input"
            processing_time = "—"
            rumor_probability = 0.0
            normal_probability = 0.0
        else:
            prediction = result["prediction"]
            confidence = f"{result['confidence'] * 100:.2f}%"
            model_name = result["model"]
            processing_time = (
                f"{result['processing_time'] / 1000:.1f} s"
            )
            rumor_probability = result["rumor"]
            normal_probability = result["non_rumor"]

        metric1, metric2 = st.columns(2)

        with metric1:
            prediction_class = (
                "rumor-value"
                if (
                    "Rumor" in prediction
                    and "Non" not in prediction
                )
                else "normal-value"
            )

            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">
                        Prediction
                    </div>
                    <div class="metric-value {prediction_class}">
                        {prediction}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">
                        Confidence
                    </div>
                    <div class="metric-value">
                        {confidence}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div style="
                margin-top:13px;
                padding-bottom:9px;
                border-bottom:1px solid #e2e8f0;
                font-size:9px;
            ">
                <div style="
                    display:flex;
                    justify-content:space-between;
                    margin-bottom:8px;
                ">
                    <span style="color:#64748b;">
                        Model Used
                    </span>
                    <b style="color:#0f172a;">
                        {model_name}
                    </b>
                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                ">
                    <span style="color:#64748b;">
                        Process Time
                    </span>
                    <b style="color:#0f172a;">
                        {processing_time}
                    </b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="prob-title">
                Probability Distribution
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="prob-row">
                <div class="prob-label">
                    <span>Rumor Class</span>
                    <span>{rumor_probability:.3f}</span>
                </div>
                <div class="prob-track">
                    <div class="prob-rumor"
                         style="width:{rumor_probability * 100:.2f}%">
                    </div>
                </div>
            </div>

            <div class="prob-row">
                <div class="prob-label">
                    <span>Non-Rumor Class</span>
                    <span>{normal_probability:.3f}</span>
                </div>
                <div class="prob-track">
                    <div class="prob-normal"
                         style="width:{normal_probability * 100:.2f}%">
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result is None:
            warning_text = "No analysis performed yet."
            warning_class = "domain-warning"
        elif result["domain_warning"]:
            warning_text = "⚠️ " + result["domain_warning"]
            warning_class = "domain-warning"
        else:
            warning_text = (
                "✓ No domain-shift concerns detected "
                "for this input."
            )
            warning_class = "domain-ok"

        st.markdown(
            f"""
            <div class="{warning_class}">
                {warning_text}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # EXAMPLES
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander(
        "🌐 English examples",
        expanded=False,
    ):
        st.markdown(
            '<div class="example-title">☷ Examples</div>',
            unsafe_allow_html=True,
        )

        for i, example in enumerate(english_examples):
            if st.button(
                example,
                key=f"eng_{i}",
                use_container_width=True,
            ):
                st.session_state.last_text = example
                st.rerun()

    with st.expander(
        "🌏 Hindi / Kannada / Tamil / Telugu / Malayalam examples",
        expanded=False,
    ):
        st.markdown(
            '<div class="example-title">☷ Examples</div>',
            unsafe_allow_html=True,
        )

        for i, example in enumerate(indian_examples):
            if st.button(
                example,
                key=f"ind_{i}",
                use_container_width=True,
            ):
                st.session_state.last_text = example
                st.rerun()


# ============================================================
# EXPLAINABILITY TAB
# ============================================================

with tab_explain:
    result = st.session_state.last_result

    if result is None:
        st.info(
            "Run an analysis first. LIME and Grad-CAM "
            "explanations will appear here."
        )
    else:
        if result["lime_html"]:
            st.markdown(
                """
                <div class="explanation-card">
                    <div class="result-title">
                        🔎 Text Explanation — LIME
                    </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                result["lime_html"],
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div style="
                    margin-top:10px;
                    font-size:9px;
                    color:#64748b;
                ">
                    <b style="color:#dc2626;">Red</b>
                    = pushes toward Rumor
                    &nbsp;&nbsp;&nbsp;
                    <b style="color:#087f79;">Green</b>
                    = pushes toward Non-Rumor
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if result["plain_explanation"]:
            st.markdown(
                f"""
                <div class="explanation-card">
                    <div class="result-title">
                        📝 Explanation
                    </div>
                    <div style="
                        font-size:11px;
                        color:#334155;
                        line-height:1.7;
                    ">
                        {result["plain_explanation"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if result["gradcam"] is not None:
            st.markdown(
                """
                <div class="explanation-card">
                    <div class="result-title">
                        🔥 Image Explanation — Grad-CAM
                    </div>
                """,
                unsafe_allow_html=True,
            )

            st.image(
                result["gradcam"],
                use_container_width=True,
            )

            st.markdown(
                """
                <div style="
                    font-size:9px;
                    color:#64748b;
                    margin-top:5px;
                ">
                    Highlighted regions represent image
                    areas contributing to the prediction.
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# BUTTON ACTIONS
# ============================================================

if clear_button:
    st.session_state.last_result = None
    st.session_state.last_text = ""
    st.rerun()


if analyze_button:
    if not text_input.strip() and uploaded_image is None:
        st.warning(
            "Please enter text, upload an image, "
            "or provide both."
        )
    else:
        with st.spinner("Analyzing content..."):
            try:
                result = run_inference(
                    text_input,
                    uploaded_image,
                    models_data,
                )

                st.session_state.last_result = result
                st.session_state.last_text = text_input

                st.rerun()

            except Exception as e:
                st.error(
                    "An error occurred during inference."
                )
                st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94a3b8;
        font-size:9px;
        padding:20px 0 5px 0;
    ">
        Intelligent Rumor Detection using BERT,
        mBERT, ResNet50 and Multimodal Fusion
    </div>
    """,
    unsafe_allow_html=True,
)
