import streamlit as st
import joblib
from PIL import Image
from googletrans import Translator
from lime.lime_text import LimeTextExplainer
import numpy as np
import os

# -------------------------
# Streamlit config must be first Streamlit command
# -------------------------
st.set_page_config(page_title="Cyberbullying Detector", page_icon="🛡️")

# -------------------------
# Helper: load model + vectorizer (cached)
# -------------------------
@st.cache_resource
def load_files():
    model = joblib.load("cyber_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf

model, tfidf = load_files()

# -------------------------
# UI: optional logo
# -------------------------
if os.path.exists("assets/logo.png"):
    try:
        logo = Image.open("assets/logo.png")
        st.image(logo, width=110)
    except Exception:
        pass

# -------------------------
# Small custom CSS for polish
# -------------------------
st.markdown(
    """
    <style>
    .stApp {{
        background-color: #fbfdff;
    }}
    .big-title {{
        font-size:28px;
        color: #1f3b5a;
        font-weight:600;
    }}
    .explain-box {{
        background: #ffffff;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Title & description
# -------------------------
st.markdown('<div class="big-title">🛡️ Cyberbullying Comment Detector</div>', unsafe_allow_html=True)
st.write(
    "Paste any social-media comment below. The model predicts whether it may be cyberbullying "
    "and shows an explanation of which words influenced the prediction."
)

# -------------------------
# Input box
# -------------------------
user_input = st.text_area("✍️ Enter a comment:", height=160)

# Language translator object (not cached due to occasional issues with objects across sessions)
translator = Translator()

# Button area
if st.button("🚀 Predict"):

    if not user_input or not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        original_text = user_input.strip()

        # 1) Translation: detect and translate if not English
        translated_flag = False
        try:
            detected = translator.detect(original_text)
            detected_lang = detected.lang
            if detected_lang != "en":
                translated = translator.translate(original_text, src=detected_lang, dest="en")
                user_input = translated.text
                translated_flag = True
                st.info(f"🌐 Detected language: `{detected_lang}` — translated to English for analysis.")
        except Exception:
            # If translation fails, just proceed with original text
            st.warning("⚠️ Translation service unavailable — proceeding with original text.")

        # 2) Preprocessing: the TF-IDF vectorizer expects raw text similar to training input
        X_vec = tfidf.transform([user_input])

        # 3) Prediction and confidence
        try:
            predicted_label = model.predict(X_vec)[0]
            if hasattr(model, "predict_proba"):
                confidence = float(model.predict_proba(X_vec).max())
            else:
                # fallback: use decision_function if available
                confidence = 0.0
        except Exception as e:
            st.error(f"Model prediction error: {e}")
            raise

        # 4) Display result
        st.markdown("---")
        if "not" in str(predicted_label).lower():
            st.success(f"✅ Result: **Not Cyberbullying** — ({confidence:.2%} confidence)")
        else:
            st.error(f"🚫 Result: **Cyberbullying Detected** — ({confidence:.2%} confidence)")

        st.progress(min(max(confidence, 0.0), 1.0))
        st.markdown("---")

        # 5) Explainability using LIME
        st.markdown("### 🔍 Why the model predicted this (LIME explanation)")
        try:
            # LIME expects a predict_proba function taking list[str] -> array
            # We need wrapper functions if our model pipeline expects vectorized input;
            # since LIME passes raw text, we define a wrapper that vectorizes then calls model.predict_proba
            def predict_proba_for_lime(texts):
                vect_texts = tfidf.transform(texts)
                return model.predict_proba(vect_texts)

            explainer = LimeTextExplainer(class_names=[str(c) for c in model.classes_])
            exp = explainer.explain_instance(user_input, predict_proba_for_lime, num_features=6)

            # Show explanation as HTML via Streamlit components (scrollable)
            import streamlit.components.v1 as components
            explanation_html = exp.as_html()
            components.html(explanation_html, height=380, scrolling=True)
        except Exception as e:
            st.warning("⚠️ LIME explanation failed or is not available: " + str(e))

        # 6) Show original text and translation note
        st.markdown("#### Original comment")
        st.info(original_text)
        if translated_flag:
            st.caption("Note: Input was translated to English before prediction.")
