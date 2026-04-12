"""
Streamlit chat mockup demo for food moderation.

Tests both ViT (video) and EfficientNet (image) models in a chat UI.
Users upload an image (or video), the selected model classifies it,
and the moderation decision is shown inline.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Whispr Food Moderation Demo", page_icon="🍔", layout="wide")

CLASSES_16 = [
    "fruits", "vegetables", "salads", "seafood", "grilled_meat", "grain_bowls",
    "soups", "smoothies", "burgers", "pizza", "fried_food", "desserts",
    "candy_sweets", "salty_snacks", "sugary_drinks", "not_food",
]

HEALTH_LABELS = {
    "fruits": "healthy", "vegetables": "healthy", "salads": "healthy",
    "seafood": "healthy", "grilled_meat": "healthy", "grain_bowls": "healthy",
    "soups": "healthy", "smoothies": "healthy",
    "burgers": "unhealthy", "pizza": "unhealthy", "fried_food": "unhealthy",
    "desserts": "unhealthy", "candy_sweets": "unhealthy",
    "salty_snacks": "unhealthy", "sugary_drinks": "unhealthy",
    "not_food": "not_food",
}

HEALTH_EMOJI = {"healthy": "🥗", "unhealthy": "🍔", "not_food": "❓"}

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Model loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_efficientnet_model():
    """Load the EfficientNet TFLite model."""
    import tensorflow as tf
    tflite_path = MODELS_DIR / "efficientnet_food.tflite"
    if not tflite_path.exists():
        return None, None
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details()[0]["dtype"]


@st.cache_resource
def load_vit_model():
    """Load the ViT TorchScript model."""
    try:
        import torch
    except ImportError:
        return None
    pt_path = MODELS_DIR / "vit_food.pt"
    if not pt_path.exists():
        return None
    model = torch.jit.load(str(pt_path), map_location="cpu")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_efficientnet(image: Image.Image) -> tuple[str, float, np.ndarray]:
    """Run EfficientNet TFLite inference on a single image."""
    interpreter, input_dtype = load_efficientnet_model()
    if interpreter is None:
        return "(model not loaded)", 0.0, np.zeros(16)

    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], img_batch.astype(input_details[0]["dtype"]))
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    idx = int(np.argmax(preds))
    return CLASSES_16[idx] if idx < len(CLASSES_16) else f"class_{idx}", float(preds[idx]), preds


def predict_vit(image: Image.Image) -> tuple[str, float, np.ndarray]:
    """Run ViT TorchScript inference -- duplicates the image 8x for the video input."""
    try:
        import torch
        from torchvision import transforms
    except ImportError:
        return "(torch not installed)", 0.0, np.zeros(16)

    model = load_vit_model()
    if model is None:
        return "(model not loaded)", 0.0, np.zeros(16)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    frame = transform(image.convert("RGB"))
    video_tensor = torch.stack([frame] * 8).unsqueeze(0)  # (1, 8, 3, 224, 224)

    with torch.no_grad():
        logits = model(video_tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()

    idx = int(np.argmax(probs))
    return CLASSES_16[idx] if idx < len(CLASSES_16) else f"class_{idx}", float(probs[idx]), probs


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🍔 Whispr Food Moderation Demo")
st.caption("Test the ViT and EfficientNet food classifiers in a chat mockup.")

with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.radio("Model", ["EfficientNet (TFLite)", "ViT (TorchScript)"], index=0)
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.4, 0.05)

    st.divider()
    st.subheader("📦 Models")
    eff_exists = (MODELS_DIR / "efficientnet_food.tflite").exists()
    vit_exists = (MODELS_DIR / "vit_food.pt").exists()
    st.write(f"{'✅' if eff_exists else '❌'} `models/efficientnet_food.tflite`")
    st.write(f"{'✅' if vit_exists else '❌'} `models/vit_food.pt`")
    if not eff_exists or not vit_exists:
        st.info("Place trained models in `app_mockup_demo/models/` -- see README.md")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey 👋 Upload a food image and I'll classify it for you. Images of unhealthy food get flagged."}
    ]

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "image" in msg:
            st.image(msg["image"], width=300)
        if msg.get("content"):
            st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Image upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

if uploaded is not None:
    img_bytes = uploaded.read()
    image = Image.open(io.BytesIO(img_bytes))

    # Append user message
    st.session_state.messages.append({
        "role": "user",
        "content": f"📎 sent image `{uploaded.name}`",
        "image": image,
    })

    with st.chat_message("user"):
        st.image(image, width=300)
        st.markdown(f"📎 sent image `{uploaded.name}`")

    # Inference
    with st.chat_message("assistant"):
        with st.spinner(f"Analyzing with {model_choice}..."):
            start = time.perf_counter()
            if model_choice.startswith("EfficientNet"):
                cls, conf, probs = predict_efficientnet(image)
            else:
                cls, conf, probs = predict_vit(image)
            latency_ms = (time.perf_counter() - start) * 1000.0

        health = HEALTH_LABELS.get(cls, "unknown")
        emoji = HEALTH_EMOJI.get(health, "❓")

        if conf < threshold:
            response = f"⚠️ **Low confidence** ({conf:.1%}). Best guess: `{cls}` [{health}]. Can you send a clearer photo?"
        elif health == "unhealthy":
            response = (
                f"{emoji} **Flagged as unhealthy**\n\n"
                f"- Detected: `{cls}` ({conf:.1%})\n"
                f"- Health group: **{health}**\n"
                f"- Latency: {latency_ms:.0f} ms\n\n"
                f"🚫 This content would be flagged by the moderation system."
            )
        elif health == "healthy":
            response = (
                f"{emoji} **Looks healthy!**\n\n"
                f"- Detected: `{cls}` ({conf:.1%})\n"
                f"- Health group: **{health}**\n"
                f"- Latency: {latency_ms:.0f} ms\n\n"
                f"✅ This content passes moderation."
            )
        else:
            response = (
                f"{emoji} **Not food**\n\n"
                f"- Detected: `{cls}` ({conf:.1%})\n"
                f"- Health group: **{health}**\n"
                f"- Latency: {latency_ms:.0f} ms\n\n"
                f"ℹ️ No food detected in this image."
            )

        st.markdown(response)

        # Top-3 predictions
        with st.expander("🔍 Top predictions"):
            top_indices = np.argsort(probs)[::-1][:5]
            for i in top_indices:
                if i < len(CLASSES_16):
                    name = CLASSES_16[i]
                    bar_len = int(probs[i] * 30)
                    st.text(f"{name:15s} {probs[i]:.1%}  {'█' * bar_len}")

    st.session_state.messages.append({"role": "assistant", "content": response})

st.divider()
st.caption(
    f"💡 Using **{model_choice}** with threshold {threshold:.0%}. "
    f"16 fine-grained classes → rolled up to healthy / unhealthy / not_food."
)
