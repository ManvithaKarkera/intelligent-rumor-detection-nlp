import os
import re
import time
import tempfile

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image

from transformers import BertTokenizerFast, BertForSequenceClassification, BertModel
from huggingface_hub import snapshot_download
from lime.lime_text import LimeTextExplainer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Intelligent Rumor Detection",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# Replace this with your actual Hugging Face model repository.
HF_REPO_ID = "ManvithaKarkera/intelligent-rumor-detection-nlp"

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

    if hf_token:
        model_dir = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=hf_token
        )
    else:
        model_dir = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="model"
        )

    return model_dir


# ============================================================
# MODEL ARCHITECTURES
# These match the architectures used in your notebook.
# ============================================================

class ImageOnlyRumorDetector(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        # weights=None because the trained checkpoint already
        # contains the ResNet weights.
        resnet = models.resnet50(weights=None)

        self.image_encoder = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        self.image_proj = nn.Linear(2048, 256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image):

        features = self.image_encoder(image)
        features = features.view(features.size(0), -1)

        x = self.image_proj(features)

        logits = self.classifier(x)

        return logits


class MultimodalRumorDetector(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        # TEXT ENCODER
        self.text_encoder = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        self.text_proj = nn.Linear(768, 256)

        # IMAGE ENCODER
        resnet = models.resnet50(weights=None)

        self.image_encoder = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        self.image_proj = nn.Linear(2048, 256)

        # FUSION:
        # Text = 256
        # Image = 256
        # Product = 256
        # Total = 768
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        image
    ):

        # TEXT FEATURES
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = text_outputs.last_hidden_state[:, 0, :]

        text_features = self.text_proj(cls_embedding)

        # IMAGE FEATURES
        image_features = self.image_encoder(image)

        image_features = image_features.view(
            image_features.size(0),
            -1
        )

        image_features = self.image_proj(
            image_features
        )

        # ELEMENT-WISE PRODUCT
        prod_features = (
            text_features * image_features
        )

        # CONCATENATE
        fused = torch.cat(
            [
                text_features,
                image_features,
                prod_features
            ],
            dim=-1
        )

        logits = self.classifier(fused)

        return logits


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_all_models():

    model_dir = download_model_repository()

    # --------------------------------------------------------
    # BERT
    # --------------------------------------------------------

    text_path = os.path.join(
        model_dir,
        "rumor_model"
    )

    rumor_tokenizer = BertTokenizerFast.from_pretrained(
        text_path
    )

    rumor_model = BertForSequenceClassification.from_pretrained(
        text_path
    )

    rumor_model.to(DEVICE)
    rumor_model.eval()

    # --------------------------------------------------------
    # mBERT
    # --------------------------------------------------------

    multi_path = os.path.join(
        model_dir,
        "rumor_model_multilingual"
    )

    multi_tokenizer = BertTokenizerFast.from_pretrained(
        multi_path
    )

    multi_model = BertForSequenceClassification.from_pretrained(
        multi_path
    )

    multi_model.to(DEVICE)
    multi_model.eval()

    # --------------------------------------------------------
    # IMAGE MODEL
    # --------------------------------------------------------

    image_checkpoint = os.path.join(
        model_dir,
        "multimodal_artifacts",
        "best_image_only.pth"
    )

    image_model = ImageOnlyRumorDetector(
        num_classes=2
    )

    image_state = torch.load(
        image_checkpoint,
        map_location=DEVICE
    )

    image_model.load_state_dict(
        image_state
    )

    image_model.to(DEVICE)
    image_model.eval()

    # --------------------------------------------------------
    # MULTIMODAL MODEL
    # --------------------------------------------------------

    multimodal_checkpoint = os.path.join(
        model_dir,
        "multimodal_artifacts",
        "best_multimodal_model.pth"
    )

    multimodal_model = MultimodalRumorDetector(
        num_classes=2
    )

    multimodal_state = torch.load(
        multimodal_checkpoint,
        map_location=DEVICE
    )

    multimodal_model.load_state_dict(
        multimodal_state
    )

    multimodal_model.to(DEVICE)
    multimodal_model.eval()

    return (
        rumor_tokenizer,
        rumor_model,
        multi_tokenizer,
        multi_model,
        image_model,
        multimodal_model
    )


# ============================================================
# IMAGE TRANSFORMATION
# Matches your notebook
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# INDIC LANGUAGE DETECTION
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

    return bool(
        INDIC_SCRIPT_PATTERN.search(text)
    )


# ============================================================
# TEXT PREDICTION
# ============================================================

def predict_text_proba(
    texts,
    tokenizer,
    model
):

    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    enc = {
        key: value.to(DEVICE)
        for key, value in enc.items()
    }

    with torch.no_grad():

        logits = model(
            **enc
        ).logits

        probs = F.softmax(
            logits,
            dim=1
        ).cpu().numpy()

    return probs


# ============================================================
# IMAGE CONVERSION
# ============================================================

def to_pil_rgb(input_image):

    if isinstance(input_image, Image.Image):

        return input_image.convert("RGB")

    arr = np.array(input_image)

    if arr.ndim == 2:

        arr = np.stack(
            [arr, arr, arr],
            axis=-1
        )

    if arr.shape[-1] == 4:

        arr = arr[:, :, :3]

    return Image.fromarray(
        arr.astype("uint8"),
        "RGB"
    )


# ============================================================
# GRAD-CAM
# ============================================================

class ResNetGradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        def forward_hook(
            module,
            input_data,
            output
        ):

            self.activations = output

        def backward_hook(
            module,
            grad_input,
            grad_output
        ):

            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(
            forward_hook
        )

        self.target_layer.register_full_backward_hook(
            backward_hook
        )

    def generate_heatmap(
        self,
        image_tensor,
        class_idx=None
    ):

        self.model.eval()

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        image_tensor.requires_grad = True

        logits = self.model(
            image_tensor
        )

        if class_idx is None:

            class_idx = (
                logits
                .argmax(dim=-1)
                .item()
            )

        score = logits[
            0,
            class_idx
        ]

        self.model.zero_grad()

        score.backward()

        gradients = (
            self.gradients
            .detach()
            .cpu()
            .numpy()[0]
        )

        activations = (
            self.activations
            .detach()
            .cpu()
            .numpy()[0]
        )

        weights = np.mean(
            gradients,
            axis=(1, 2)
        )

        heatmap = np.zeros(
            activations.shape[1:],
            dtype=np.float32
        )

        for i, weight in enumerate(weights):

            heatmap += (
                weight *
                activations[i]
            )

        heatmap = np.maximum(
            heatmap,
            0
        )

        if heatmap.max() > 0:

            heatmap /= heatmap.max()

        return heatmap


def get_gradcam_target_layer(model):

    for child in reversed(
        list(model.image_encoder.children())
    ):

        if isinstance(
            child,
            nn.Sequential
        ):

            return child[-1]

    raise ValueError(
        "Could not find Grad-CAM target layer."
    )


def create_gradcam_image(
    model,
    image,
    class_idx
):

    import cv2

    image_tensor = image_transform(
        image
    )

    target_layer = (
        get_gradcam_target_layer(
            model
        )
    )

    gradcam = ResNetGradCAM(
        model,
        target_layer
    )

    heatmap = gradcam.generate_heatmap(
        image_tensor,
        class_idx
    )

    original = np.array(
        image.resize((224, 224))
    )

    heatmap_resized = cv2.resize(
        heatmap,
        (224, 224)
    )

    heatmap_colored = np.uint8(
        255 * heatmap_resized
    )

    heatmap_colored = cv2.applyColorMap(
        heatmap_colored,
        cv2.COLORMAP_JET
    )

    original_bgr = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    overlayed = cv2.addWeighted(
        original_bgr,
        0.6,
        heatmap_colored,
        0.4,
        0
    )

    overlayed_rgb = cv2.cvtColor(
        overlayed,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(
        overlayed_rgb
    )


# ============================================================
# LIME
# ============================================================

lime_explainer = LimeTextExplainer(
    class_names=CLASS_NAMES,
    split_expression=r"\s+",
    bow=False
)


def render_lime_html(
    word_scores
):

    if not word_scores:

        return "No explanation available."

    spans = []

    max_abs = max(
        abs(score)
        for _, score in word_scores
    ) or 1.0

    for word, score in word_scores:

        intensity = min(
            abs(score) / max_abs,
            1.0
        )

        if score > 0:

            background = (
                f"rgba(220,38,38,{intensity:.2f})"
            )

        else:

            background = (
                f"rgba(22,163,74,{intensity:.2f})"
            )

        spans.append(
            f"""
            <span style="
                background:{background};
                padding:3px 7px;
                margin:2px;
                border-radius:5px;
                display:inline-block;
            ">
                {word}
            </span>
            """
        )

    legend = """
    <div style="
        margin-top:12px;
        font-size:12px;
        color:#64748b;
    ">
        <b>Red</b> = pushes toward Rumor
        &nbsp;&nbsp;&nbsp;
        <b>Green</b> = pushes toward Non-Rumor
    </div>
    """

    return (
        '<div style="line-height:2.2;">'
        + " ".join(spans)
        + "</div>"
        + legend
    )


def generate_plain_explanation(
    word_scores,
    prediction_label,
    confidence
):

    if not word_scores:

        return (
            "No significant words were identified "
            "for this prediction."
        )

    sorted_scores = sorted(
        word_scores,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_word, top_score = (
        sorted_scores[0]
    )

    direction = (
        "Rumor"
        if top_score > 0
        else "Non-Rumor"
    )

    explanation = (
        f"The model leaned toward "
        f"**{prediction_label}** primarily because "
        f"of the word **{top_word}**, which had "
        f"the strongest influence on this prediction. "
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

def domain_shift_warning(
    text,
    confidence
):

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

def run_inference(
    text,
    uploaded_image,
    models_data
):

    (
        rumor_tokenizer,
        rumor_model,
        multi_tokenizer,
        multi_model,
        image_model,
        multimodal_model
    ) = models_data

    has_text = (
        text is not None
        and len(text.strip()) > 0
    )

    has_image = (
        uploaded_image is not None
    )

    if not has_text and not has_image:

        return None

    start_time = time.time()

    # ========================================================
    # TEXT ONLY
    # ========================================================

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
            model
        )[0]

        pred_idx = int(
            np.argmax(probs)
        )

        confidence = float(
            probs[pred_idx]
        )

        explanation = lime_explainer.explain_instance(
            text,
            classifier_fn=lambda x:
                predict_text_proba(
                    x,
                    tokenizer,
                    model
                ),
            labels=[pred_idx],
            num_features=6,
            num_samples=500
        )

        word_scores = explanation.as_list(
            label=pred_idx
        )

        lime_html = render_lime_html(
            word_scores
        )

        plain = generate_plain_explanation(
            word_scores,
            CLASS_NAMES[pred_idx],
            confidence
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
            "lime_html": lime_html,
            "plain_explanation": plain,
            "gradcam": None,
            "domain_warning": domain_shift_warning(
                text,
                confidence
            )
        }

    # ========================================================
    # IMAGE ONLY
    # ========================================================

    if has_image and not has_text:

        image = to_pil_rgb(
            uploaded_image
        )

        image_tensor = image_transform(
            image
        )

        with torch.no_grad():

            logits = image_model(
                image_tensor
                .unsqueeze(0)
                .to(DEVICE)
            )

            probs = F.softmax(
                logits,
                dim=1
            ).cpu().numpy()[0]

        pred_idx = int(
            np.argmax(probs)
        )

        confidence = float(
            probs[pred_idx]
        )

        gradcam_image = create_gradcam_image(
            image_model,
            image,
            pred_idx
        )

        return {
            "prediction": (
                f"{CLASS_NAMES[pred_idx]} "
                "(Image Only)"
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
                "No text was provided, so "
                "no text-based explanation is available."
            ),
            "gradcam": gradcam_image,
            "domain_warning": (
                "Rumor detection is generally more reliable "
                "when textual context accompanies the image."
            )
        }

    # ========================================================
    # TEXT + IMAGE
    # ========================================================

    text = text.strip()

    image = to_pil_rgb(
        uploaded_image
    )

    image_tensor = image_transform(
        image
    )

    tokens = rumor_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    tokens = {
        key: value.to(DEVICE)
        for key, value in tokens.items()
    }

    with torch.no_grad():

        logits = multimodal_model(
            tokens["input_ids"],
            tokens["attention_mask"],
            image_tensor.unsqueeze(0).to(DEVICE)
        )

        probs = F.softmax(
            logits,
            dim=1
        ).cpu().numpy()[0]

    pred_idx = int(
        np.argmax(probs)
    )

    confidence = float(
        probs[pred_idx]
    )

    # LIME uses the text-only BERT model,
    # matching your notebook's explanation approach.
    explanation = lime_explainer.explain_instance(
        text,
        classifier_fn=lambda x:
            predict_text_proba(
                x,
                rumor_tokenizer,
                rumor_model
            ),
        labels=[pred_idx],
        num_features=6,
        num_samples=500
    )

    word_scores = explanation.as_list(
        label=pred_idx
    )

    lime_html = render_lime_html(
        word_scores
    )

    plain = generate_plain_explanation(
        word_scores,
        CLASS_NAMES[pred_idx],
        confidence
    )

    gradcam_image = create_gradcam_image(
        image_model,
        image,
        pred_idx
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
        "lime_html": lime_html,
        "plain_explanation": plain,
        "gradcam": gradcam_image,
        "domain_warning": domain_shift_warning(
            text,
            confidence
        )
    }


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def initialize():

    return load_all_models()


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🛡️ Intelligent Rumor Detection System")

st.markdown(
    """
    **BERT • mBERT • ResNet50 • Multimodal Fusion •
    LIME • Grad-CAM**
    """
)

st.info(
    "Enter text, upload an image, or provide both. "
    "The system automatically selects the appropriate model."
)


# ------------------------------------------------------------
# Load models
# ------------------------------------------------------------

with st.spinner(
    "Loading trained models from Hugging Face..."
):

    try:

        models_data = initialize()

        models_loaded = True

    except Exception as e:

        models_loaded = False

        st.error(
            "Unable to load the trained models."
        )

        st.exception(e)


# ------------------------------------------------------------
# Model status
# ------------------------------------------------------------

if models_loaded:

    st.success(
        "All four trained models are ready."
    )


# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("📝 Text Input")

    text_input = st.text_area(
        "Enter a social media post or news text",
        height=220,
        placeholder=(
            "Example: Enter the claim or "
            "social media post you want to analyze..."
        )
    )


with col2:

    st.subheader("🖼️ Image Input")

    image_input = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    preview_image = None

    if image_input is not None:

        preview_image = Image.open(
            image_input
        ).convert("RGB")

        st.image(
            preview_image,
            caption="Uploaded image",
            use_container_width=True
        )


# ------------------------------------------------------------
# Analyze
# ------------------------------------------------------------

st.divider()

analyze_button = st.button(
    "🔍 Analyze Content",
    type="primary",
    use_container_width=True
)


if analyze_button:

    if not models_loaded:

        st.error(
            "Models could not be loaded. "
            "Please check the Hugging Face repository "
            "and deployment logs."
        )

    elif (
        not text_input.strip()
        and preview_image is None
    ):

        st.warning(
            "Please enter text, upload an image, "
            "or provide both."
        )

    else:

        with st.spinner(
            "Analyzing content..."
        ):

            try:

                result = run_inference(
                    text_input,
                    preview_image,
                    models_data
                )

                # ============================================
                # RESULT
                # ============================================

                st.subheader(
                    "📊 Inference Report"
                )

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    prediction = result[
                        "prediction"
                    ]

                    if "Rumor" in prediction and "Non" not in prediction:

                        st.error(
                            f"Prediction: {prediction}"
                        )

                    else:

                        st.success(
                            f"Prediction: {prediction}"
                        )

                with result_col2:

                    st.metric(
                        "Confidence",
                        f"{result['confidence'] * 100:.2f}%"
                    )

                # ============================================
                # METADATA
                # ============================================

                meta1, meta2 = st.columns(2)

                with meta1:

                    st.write(
                        "**Model Used:**",
                        result["model"]
                    )

                with meta2:

                    st.write(
                        "**Processing Time:**",
                        f"{result['processing_time']:.1f} ms"
                    )

                # ============================================
                # PROBABILITIES
                # ============================================

                st.subheader(
                    "Probability Distribution"
                )

                p1, p2 = st.columns(2)

                with p1:

                    st.metric(
                        "Non-Rumor",
                        f"{result['non_rumor'] * 100:.2f}%"
                    )

                    st.progress(
                        float(result["non_rumor"])
                    )

                with p2:

                    st.metric(
                        "Rumor",
                        f"{result['rumor'] * 100:.2f}%"
                    )

                    st.progress(
                        float(result["rumor"])
                    )

                # ============================================
                # DOMAIN WARNING
                # ============================================

                if result["domain_warning"]:

                    st.warning(
                        result["domain_warning"]
                    )

                # ============================================
                # TEXT EXPLANATION
                # ============================================

                if result["lime_html"]:

                    st.subheader(
                        "🔎 Text Explanation — LIME"
                    )

                    st.markdown(
                        result["lime_html"],
                        unsafe_allow_html=True
                    )

                    st.subheader(
                        "📝 Explanation"
                    )

                    st.markdown(
                        result["plain_explanation"]
                    )

                # ============================================
                # IMAGE EXPLANATION
                # ============================================

                if result["gradcam"] is not None:

                    st.subheader(
                        "🔥 Image Explanation — Grad-CAM"
                    )

                    st.image(
                        result["gradcam"],
                        caption=(
                            "Grad-CAM visualization "
                            "showing regions influencing "
                            "the prediction."
                        ),
                        use_container_width=True
                    )

            except Exception as e:

                st.error(
                    "An error occurred while processing "
                    "the input."
                )

                st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Intelligent Rumor Detection using BERT, "
    "mBERT, ResNet50 and Multimodal Fusion"
)
