import os, re, time
import numpy as np
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
import gradio as gr

# ── Device ──────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Non-Rumor", "Rumor"]
HF_REPO_ID  = "ManvithaKarkera/rumor-detection-models"

# ── Download models from your HF repo ───────────────
print("Downloading models...")
model_dir = snapshot_download(repo_id=HF_REPO_ID, repo_type="model")
print(f"Models at: {model_dir}")

# ── Model architectures ──────────────────────────────
class ImageOnlyRumorDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_proj    = nn.Linear(2048, 256)
        self.classifier    = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
    def forward(self, image):
        f = self.image_encoder(image).view(image.size(0), -1)
        return self.classifier(self.image_proj(f))

class MultimodalRumorDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.text_proj    = nn.Linear(768, 256)
        resnet = models.resnet50(weights=None)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_proj    = nn.Linear(2048, 256)
        self.classifier    = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, input_ids, attention_mask, image):
        cls  = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        tf   = self.text_proj(cls)
        imf  = self.image_proj(self.image_encoder(image).view(image.size(0), -1))
        fused = torch.cat([tf, imf, tf * imf], dim=-1)
        return self.classifier(fused)

# ── Image transform ──────────────────────────────────
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Load all models ──────────────────────────────────
text_model_loaded  = False
multi_model_loaded = False
image_model_loaded = False
mm_model_loaded    = False

try:
    rumor_tokenizer = BertTokenizerFast.from_pretrained(os.path.join(model_dir, "rumor_model"))
    rumor_model     = BertForSequenceClassification.from_pretrained(os.path.join(model_dir, "rumor_model")).to(DEVICE).eval()
    text_model_loaded = True
    print("English BERT loaded ✓")
except Exception as e:
    print(f"English BERT failed: {e}")

try:
    multi_tokenizer = BertTokenizerFast.from_pretrained(os.path.join(model_dir, "rumor_model_multilingual"))
    multi_model     = BertForSequenceClassification.from_pretrained(os.path.join(model_dir, "rumor_model_multilingual")).to(DEVICE).eval()
    multi_model_loaded = True
    print("mBERT loaded ✓")
except Exception as e:
    print(f"mBERT failed: {e}")

try:
    image_model = ImageOnlyRumorDetector()
    image_model.load_state_dict(torch.load(
        os.path.join(model_dir, "multimodal_artifacts", "best_image_only.pth"),
        map_location=DEVICE
    ))
    image_model.to(DEVICE).eval()
    image_model_loaded = True
    print("Image model loaded ✓")
except Exception as e:
    print(f"Image model failed: {e}")

try:
    mm_model = MultimodalRumorDetector()
    mm_model.load_state_dict(torch.load(
        os.path.join(model_dir, "multimodal_artifacts", "best_multimodal_model.pth"),
        map_location=DEVICE
    ))
    mm_model.to(DEVICE).eval()
    mm_model_loaded = True
    print("Multimodal model loaded ✓")
except Exception as e:
    print(f"Multimodal model failed: {e}")

# ── Helpers ──────────────────────────────────────────
INDIC = re.compile(
    "[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF"
    "\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F]"
)
def is_indic(text): return bool(INDIC.search(text))

def to_pil_rgb(img):
    if isinstance(img, Image.Image): return img.convert("RGB")
    arr = np.array(img)
    if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 4: arr = arr[:,:,:3]
    return Image.fromarray(arr.astype("uint8"), "RGB")

def predict_text_proba(texts):
    enc = rumor_tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return F.softmax(rumor_model(**enc).logits, dim=1).cpu().numpy()

def predict_multi_proba(texts):
    enc = multi_tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return F.softmax(multi_model(**enc).logits, dim=1).cpu().numpy()

lime_explainer = LimeTextExplainer(class_names=CLASS_NAMES, split_expression=r'\s+', bow=False)

def render_lime_html(word_scores):
    if not word_scores: return "<i>No explanation available.</i>"
    max_abs = max(abs(s) for _, s in word_scores) or 1.0
    spans = []
    for word, score in word_scores:
        intensity = min(abs(score)/max_abs, 1.0)
        color = f"rgba(220,38,38,{intensity:.2f})" if score > 0 else f"rgba(22,163,74,{intensity:.2f})"
        spans.append(f'<span style="background:{color};padding:2px 6px;margin:2px;border-radius:4px;display:inline-block;">{word}</span>')
    legend = ('<div style="margin-top:10px;font-size:12px;color:#64748b;">'
              '<span style="background:rgba(220,38,38,0.5);padding:1px 6px;border-radius:3px;">red = Rumor</span>&nbsp;&nbsp;'
              '<span style="background:rgba(22,163,74,0.5);padding:1px 6px;border-radius:3px;">green = Non-Rumor</span></div>')
    return '<div style="line-height:2.2;">' + " ".join(spans) + '</div>' + legend

def generate_plain_explanation(word_scores, label, confidence):
    if not word_scores: return "No significant words identified."
    top = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)
    top_word = top[0][0]
    explanation = f"The model predicted **{label}** mainly because of **{top_word}**. "
    if confidence >= 0.85: explanation += "The model is highly confident."
    elif confidence >= 0.65: explanation += "The model is reasonably confident."
    else: explanation += "Confidence is low — treat this result cautiously."
    return explanation

def domain_shift_warning(text, confidence):
    msgs = []
    if confidence < 0.65: msgs.append("Low confidence — input may be outside training domain.")
    if len(text.split()) < 4: msgs.append("Very short input — LIME may be less reliable.")
    return " ".join(msgs)

# ── Grad-CAM ─────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.grads = self.acts = None
        target_layer.register_forward_hook(lambda m,i,o: setattr(self,'acts',o))
        target_layer.register_full_backward_hook(lambda m,gi,go: setattr(self,'grads',go[0]))

    def generate(self, img_tensor, class_idx):
        self.model.eval()
        t = img_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)
        logits = self.model(t)
        self.model.zero_grad()
        logits[0, class_idx].backward()
        w   = self.grads.detach().cpu().numpy()[0].mean(axis=(1,2))
        act = self.acts.detach().cpu().numpy()[0]
        hm  = np.maximum(np.einsum('c,chw->hw', w, act), 0)
        if hm.max() > 0: hm /= hm.max()
        return hm

def make_gradcam_image(model, pil_img, class_idx):
    import cv2
    target_layer = list(model.image_encoder.children())[-1]
    if isinstance(target_layer, nn.Sequential): target_layer = target_layer[-1]
    cam    = GradCAM(model, target_layer)
    hm     = cam.generate(image_transform(pil_img), class_idx)
    orig   = np.array(pil_img.resize((224,224)))
    hm_r   = cv2.resize(hm, (224,224))
    colored = cv2.applyColorMap(np.uint8(255*hm_r), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(orig, cv2.COLOR_RGB2BGR), 0.6, colored, 0.4, 0)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

# ── Main inference ────────────────────────────────────
def _empty(msg=""): return {"Prediction":"—","Confidence Score":"—","Model Used":"—","Processing Time":"—","Ablation Comparison":"","Domain Note":msg,"Text Explanation (LIME)":"","Text Explanation (Plain)":"","Image Explanation (Grad-CAM)":None}
def _error(e):      return {"Prediction":"Error","Confidence Score":"-","Model Used":"-","Processing Time":"-","Ablation Comparison":"","Domain Note":f"⚠️ {e}","Text Explanation (LIME)":"","Text Explanation (Plain)":"","Image Explanation (Grad-CAM)":None}

def run_inference(input_text, input_image):
    try:
        start = time.time()
        has_text  = input_text  is not None and str(input_text).strip()
        has_image = input_image is not None

        if not has_text and not has_image:
            return _empty()

        if has_text and not has_image:
            txt = str(input_text).strip()
            if is_indic(txt):
                if not multi_model_loaded: return _empty("mBERT not loaded.")
                probs = predict_multi_proba([txt])[0]; model_name = "Multilingual (mBERT)"
                fn = predict_multi_proba
            else:
                if not text_model_loaded: return _empty("BERT not loaded.")
                probs = predict_text_proba([txt])[0]; model_name = "Text-Only (BERT)"
                fn = predict_text_proba
            pred_idx   = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            exp        = lime_explainer.explain_instance(txt, fn, labels=[pred_idx], num_features=6, num_samples=500)
            ws         = exp.as_list(label=pred_idx)
            return {
                "Prediction": CLASS_NAMES[pred_idx],
                "Confidence Score": f"{confidence*100:.2f}%",
                "Model Used": model_name,
                "Processing Time": f"{(time.time()-start)*1000:.1f} ms",
                "Ablation Comparison": f"Rumor: {probs[1]:.2f} | Non-Rumor: {probs[0]:.2f}",
                "Domain Note": domain_shift_warning(txt, confidence),
                "Text Explanation (LIME)": render_lime_html(ws),
                "Text Explanation (Plain)": generate_plain_explanation(ws, CLASS_NAMES[pred_idx], confidence),
                "Image Explanation (Grad-CAM)": None
            }

        if has_image and not has_text:
            if not image_model_loaded: return _empty("Image model not loaded.")
            img    = to_pil_rgb(input_image)
            tensor = image_transform(img)
            with torch.no_grad():
                probs = F.softmax(image_model(tensor.unsqueeze(0).to(DEVICE)), dim=1).cpu().numpy()[0]
            pred_idx   = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            gradcam    = make_gradcam_image(image_model, img, pred_idx)
            return {
                "Prediction": f"{CLASS_NAMES[pred_idx]} (Image Only)",
                "Confidence Score": f"{confidence*100:.2f}%",
                "Model Used": "Image-Only (ResNet50)",
                "Processing Time": f"{(time.time()-start)*1000:.1f} ms",
                "Ablation Comparison": f"Non-Rumor: {probs[0]:.2f} | Rumor: {probs[1]:.2f}",
                "Domain Note": "Most reliable when text accompanies the image.",
                "Text Explanation (LIME)": "<i>No text provided.</i>",
                "Text Explanation (Plain)": "No text provided.",
                "Image Explanation (Grad-CAM)": gradcam
            }

        # Text + Image
        if not (mm_model_loaded and text_model_loaded and image_model_loaded):
            return _empty("One or more models not loaded.")
        txt    = str(input_text).strip()
        img    = to_pil_rgb(input_image)
        tensor = image_transform(img)
        tokens = rumor_tokenizer(txt, padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(mm_model(tokens['input_ids'], tokens['attention_mask'], tensor.unsqueeze(0).to(DEVICE)), dim=1).cpu().numpy()[0]
        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        exp        = lime_explainer.explain_instance(txt, predict_text_proba, labels=[pred_idx], num_features=6, num_samples=500)
        ws         = exp.as_list(label=pred_idx)
        gradcam    = make_gradcam_image(image_model, img, pred_idx)
        return {
            "Prediction": CLASS_NAMES[pred_idx],
            "Confidence Score": f"{confidence*100:.2f}%",
            "Model Used": "Multimodal (BERT + ResNet50)",
            "Processing Time": f"{(time.time()-start)*1000:.1f} ms",
            "Ablation Comparison": f"Multimodal — Non-Rumor: {probs[0]:.2f}, Rumor: {probs[1]:.2f}",
            "Domain Note": domain_shift_warning(txt, confidence),
            "Text Explanation (LIME)": render_lime_html(ws),
            "Text Explanation (Plain)": generate_plain_explanation(ws, CLASS_NAMES[pred_idx], confidence),
            "Image Explanation (Grad-CAM)": gradcam
        }
    except Exception as e:
        return _error(str(e))

# ── Status bar ────────────────────────────────────────
def status_bar_html():
    def chip(name, loaded):
        dot = "dot-on" if loaded else "dot-off"
        state = "Ready" if loaded else "Not loaded"
        return f'<div class="status-chip"><span class="{dot}"></span>{name}: {state}</div>'
    return (
        '<div id="status-bar">'
        + chip("English BERT", text_model_loaded)
        + chip("Multilingual mBERT", multi_model_loaded)
        + chip("Image ResNet50", image_model_loaded)
        + chip("Multimodal Fusion", mm_model_loaded)
        + '</div>'
    )

# ── Report card renderer ──────────────────────────────
def parse_probs(ablation_text):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", ablation_text or "")
    try:
        rumor_p = float(nums[0]); non_p = float(nums[-1])
    except Exception:
        rumor_p, non_p = 0.5, 0.5
    return non_p, rumor_p

def live_badge(status):
    if status == "loading": return '<span class="badge-pending mono">⏳ ANALYZING…</span>'
    if status == "done":    return '<span class="live-badge mono">🟢 LIVE</span>'
    return '<span class="badge-idle mono">⚪ IDLE</span>'

def render_report_card(prediction, confidence_str, model_used, proc_time, ablation_text, domain_note, status="idle"):
    is_rumor   = "Rumor" in str(prediction) and "Non" not in str(prediction)
    pred_class = "v-pending" if status == "loading" else ("v-rumor" if is_rumor else "v-nonrumor")

    if status == "loading":
        body = '<div class="stat-row"><div class="stat-box"><div class="k">Prediction</div><div class="v v-pending">Analyzing…</div></div><div class="stat-box"><div class="k">Confidence</div><div class="v v-pending">—</div></div></div>'
    else:
        non_p, rumor_p = parse_probs(ablation_text)
        domain_html = (f'<div class="domain-warning">⚠️ {domain_note}</div>' if domain_note and domain_note.strip()
                       else '<div class="domain-clean">✅ No domain-shift concerns detected.</div>')
        body = f"""
        <div class="stat-row">
            <div class="stat-box"><div class="k">Prediction</div><div class="v {pred_class}">{prediction}</div></div>
            <div class="stat-box"><div class="k">Confidence</div><div class="v">{confidence_str}</div></div>
        </div>
        <div class="meta-line"><span class="k">Model Used</span><span class="v">{model_used}</span></div>
        <div class="meta-line"><span class="k">Process Time</span><span class="v">{proc_time}</span></div>
        <div class="prob-block">
            <div class="field-label">Probability Distribution</div>
            <div class="prob-row">
                <div class="lbl"><span>Rumor</span><span>{rumor_p:.3f}</span></div>
                <div class="prob-track"><div class="prob-fill" style="width:{rumor_p*100:.1f}%;background:#dc2626;"></div></div>
            </div>
            <div class="prob-row">
                <div class="lbl"><span>Non-Rumor</span><span>{non_p:.3f}</span></div>
                <div class="prob-track"><div class="prob-fill" style="width:{non_p*100:.1f}%;background:#0f766e;"></div></div>
            </div>
        </div>
        {domain_html}"""

    return f'<div class="report-card"><div class="report-head"><h3>Inference Report</h3>{live_badge(status)}</div>{body}</div>'

# ── Gradio UI ─────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.gradio-container { background: #f1f5f9 !important; }
#navbar { background: linear-gradient(90deg,#0d9488 0%,#6366f1 100%); padding:18px 28px; border-radius:16px 16px 0 0; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; }
#navbar h1 { color:white; font-size:19px; font-weight:800; margin:0; }
.badge-row { display:flex; gap:6px; flex-wrap:wrap; }
.badge { background:rgba(255,255,255,0.2); border:1px solid rgba(255,255,255,0.35); padding:4px 11px; border-radius:999px; font-size:11px; font-weight:600; color:white; }
div[role="tablist"] { background:linear-gradient(90deg,#0f766e 0%,#4f46e5 100%) !important; border:none !important; padding:0 24px !important; margin:0 0 20px 0 !important; border-radius:0 0 16px 16px !important; }
div[role="tablist"] button[role="tab"] { color:rgba(255,255,255,0.7) !important; font-weight:600 !important; border:none !important; background:transparent !important; padding:12px 6px !important; }
div[role="tablist"] button[role="tab"].selected { color:#ffffff !important; border-bottom:2.5px solid #ffffff !important; }
#status-bar { display:flex; gap:10px; flex-wrap:wrap; justify-content:center; padding:10px; margin-bottom:18px; background:white; border:1px solid #e2e8f0; border-radius:12px; }
.status-chip { display:flex; align-items:center; gap:6px; font-size:12px; font-weight:600; padding:4px 10px; border-radius:8px; background:#f8fafc; border:1px solid #e2e8f0; color:#334155; }
.dot-on { width:7px; height:7px; border-radius:50%; background:#16a34a; display:inline-block; }
.dot-off { width:7px; height:7px; border-radius:50%; background:#dc2626; display:inline-block; }
.panel-card { background:white; border:1px solid #e2e8f0; border-radius:16px; padding:20px; }
.field-label { font-size:11px; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; color:#64748b; margin:14px 0 6px; }
.gr-button-primary { background:linear-gradient(135deg,#0d9488,#6366f1) !important; border:none !important; font-weight:700 !important; border-radius:10px !important; }
.report-card { background:white; border:1px solid #e2e8f0; border-radius:16px; padding:20px; }
.report-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; }
.report-head h3 { font-size:15px; font-weight:700; color:#0f172a; margin:0; }
.live-badge { background:#ecfdf5; color:#047857; border:1px solid #a7f3d0; font-size:10.5px; font-weight:800; padding:3px 10px; border-radius:999px; }
.badge-pending { background:#fffbeb; color:#b45309; border:1px solid #fcd34d; font-size:10.5px; font-weight:800; padding:3px 10px; border-radius:999px; animation:pulse 1.1s ease-in-out infinite; }
.badge-idle { background:#f1f5f9; color:#64748b; border:1px solid #cbd5e1; font-size:10.5px; font-weight:800; padding:3px 10px; border-radius:999px; }
@keyframes pulse { 0%,100%{opacity:1}50%{opacity:0.45} }
.stat-row { display:flex; gap:12px; margin-bottom:14px; }
.stat-box { flex:1; background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:12px 14px; }
.stat-box .k { font-size:10.5px; font-weight:700; text-transform:uppercase; color:#94a3b8; }
.stat-box .v { font-size:21px; font-weight:800; margin-top:3px; }
.v-rumor { color:#b91c1c; } .v-nonrumor { color:#15803d; } .v-pending { color:#94a3b8; }
.meta-line { display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #f1f5f9; font-size:13px; }
.meta-line .k { color:#64748b; } .meta-line .v { color:#1e293b; font-weight:600; }
.prob-block { margin-top:16px; }
.prob-row { margin-bottom:10px; }
.prob-row .lbl { display:flex; justify-content:space-between; font-size:12.5px; font-weight:600; margin-bottom:4px; }
.prob-track { width:100%; height:9px; background:#e2e8f0; border-radius:999px; overflow:hidden; }
.prob-fill { height:100%; border-radius:999px; }
.domain-warning { background:#fffbeb; border:1.5px solid #f59e0b; border-radius:10px; padding:11px 14px; color:#92400e; font-size:12.5px; margin-top:14px; }
.domain-clean { background:#f0fdf4; border:1.5px solid #16a34a; border-radius:10px; padding:9px 14px; color:#166534; font-size:12px; margin-top:14px; }
footer { visibility:hidden; }
#custom-footer { text-align:center; padding:16px; color:#94a3b8; font-size:12px; margin-top:22px; border-top:1px solid #e2e8f0; }
"""

def gradio_predict_ui(text, image):
    yield (render_report_card("","","","","","",status="loading"), "<i>⏳ Running LIME…</i>", "*⏳ Generating…*", None, live_badge("loading"))
    res = run_inference(text, image)
    yield (
        render_report_card(res["Prediction"], res["Confidence Score"], res["Model Used"], res["Processing Time"], res["Ablation Comparison"], res.get("Domain Note",""), status="done"),
        res["Text Explanation (LIME)"],
        res.get("Text Explanation (Plain)",""),
        res["Image Explanation (Grad-CAM)"],
        live_badge("done"),
    )

with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="indigo"), css=custom_css) as demo:
    gr.HTML("""<div id="navbar"><h1>🛡️ Intelligent Rumor Detection System</h1>
        <div class="badge-row">
            <span class="badge">🇬🇧 EN</span><span class="badge">🇮🇳 HI</span>
            <span class="badge">🇮🇳 KN</span><span class="badge">🇮🇳 TA</span>
            <span class="badge">🇮🇳 TE</span><span class="badge">🇮🇳 ML</span>
        </div></div>""")

    gr.HTML(status_bar_html())

    with gr.Tabs():
        with gr.Tab("Analyze Content"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="panel-card"):
                        gr.HTML('<div class="field-label">Content Payload</div>')
                        input_text = gr.Textbox(show_label=False, placeholder="Paste rumor text here. Hindi, Kannada, Tamil, Telugu, Malayalam also supported...", lines=4)
                        gr.HTML('<div class="field-label">Image (optional)</div>')
                        input_image = gr.Image(show_label=False, type="pil")
                        with gr.Row():
                            btn_clear   = gr.Button("🗑️ Clear", variant="secondary")
                            btn_analyze = gr.Button("⚡ Analyze", variant="primary")

                    with gr.Accordion("🇬🇧 English examples", open=True):
                        gr.Examples(examples=[
                            ["BREAKING: New studies suggest eating garlic prevents virus contraction immediately.", None],
                            ["Government launches new education policy changes for secondary schools nationwide.", None],
                            ["Cash withdrawal limit at ATMs reduced to ₹2000 per day starting tomorrow, RBI has not confirmed.", None],
                            ["ISRO successfully launches Chandrayaan-4 ahead of schedule, confirms official press release.", None],
                        ], inputs=[input_text, input_image])

                    with gr.Accordion("🇮🇳 Indian language examples", open=False):
                        gr.Examples(examples=[
                            ["कोरोना वैक्सीन लेने से 5 साल के अंदर मौत हो सकती है", None],
                            ["व्हाट्सएप पर वायरल: नया आधार कार्ड नियम आज रात से लागू, तुरंत अपडेट करें वरना खाता बंद", None],
                            ["ಭಾರತ ಸರ್ಕಾರ ಇಂದು ಹೊಸ ಶಿಕ್ಷಣ ನೀತಿಯನ್ನು ಘೋಷಿಸಿತು", None],
                            ["ரயில் கட்டணம் நாளை முதல் 50% அதிகரிக்கும் என அரசு அறிவித்தது என்று வதந்தி பரவுகிறது", None],
                            ["రాష్ట్రంలో రేపటి నుంచి అన్ని పాఠశాలలకు వేసవి సెలవులు ప్రకటన", None],
                            ["ന്യൂഡൽഹി: കേന്ദ്ര സർക്കാർ പുതിയ കാർഷിക നയം പ്രഖ്യാപിച്ചു", None],
                        ], inputs=[input_text, input_image])

                with gr.Column(scale=1):
                    out_report = gr.HTML(render_report_card("","","","","","",status="idle"))

        with gr.Tab("Explainability"):
            out_explain_status = gr.HTML(visible=False)
            with gr.Row():
                with gr.Column():
                    with gr.Group(elem_classes="panel-card"):
                        gr.Markdown("#### 📝 Plain Explanation")
                        out_plain_explanation = gr.Markdown()
                        gr.Markdown("#### 🔬 LIME Word Highlights")
                        out_lime = gr.HTML()
                with gr.Column():
                    with gr.Group(elem_classes="panel-card"):
                        gr.Markdown("#### 🖼️ Grad-CAM Heatmap")
                        out_gradcam = gr.Image(type="pil", interactive=False)

    gr.HTML('<div id="custom-footer">Built with BERT · mBERT · ResNet50 · LIME · Grad-CAM | Intelligent Rumor Detection — AJIET Mangaluru</div>')

    outputs_list = [out_report, out_lime, out_plain_explanation, out_gradcam, out_explain_status]
    btn_analyze.click(fn=gradio_predict_ui, inputs=[input_text, input_image], outputs=outputs_list)
    input_text.submit(fn=gradio_predict_ui, inputs=[input_text, input_image], outputs=outputs_list)
    btn_clear.click(fn=lambda: (None, None, render_report_card("","","","","","",status="idle"), "", "", None, live_badge("idle")),
                    inputs=None, outputs=[input_text, input_image] + outputs_list)

demo.launch()
