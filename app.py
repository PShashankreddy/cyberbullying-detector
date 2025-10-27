import streamlit as st
import joblib
from lime.lime_text import LimeTextExplainer
import numpy as np
import pandas as pd
from deep_translator import GoogleTranslator
from datetime import datetime
import matplotlib.pyplot as plt
import os

# ----------------------------------------
# Streamlit App Configuration
# ----------------------------------------
st.set_page_config(
    page_title="Cyberbullying Detector",
    page_icon="🛡️",
    layout="centered"
)

# ----------------------------------------
# Load Model and TF-IDF Vectorizer
# ----------------------------------------
@st.cache_resource
def load_files():
    model = joblib.load("cyber_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf

model, tfidf = load_files()

# ----------------------------------------
# App Title and Description
# ----------------------------------------
st.title("🛡️ Cyberbullying Comment Detector")
st.markdown("""
Detect and explain potentially harmful online comments — now with:
- 🌐 Translation (auto to English)
- 🔍 Explainability (LIME)
- 💾 Prediction Logging
""")

# ----------------------------------------
# User Input
# ----------------------------------------
user_input = st.text_area("✍️ Enter a comment:", height=150)

col1, col2 = st.columns(2)
translate = col1.toggle("🌐 Translate to English (auto)")
explain = col2.toggle("🔍 Show Explainability (LIME)")

# ----------------------------------------
# Prediction Logic
# ----------------------------------------
if st.button("🚀 Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        # Step 1: Translation
        text_to_analyze = user_input
        if translate:
            try:
                text_to_analyze = GoogleTranslator(source="auto", target="en").translate(user_input)
                st.info(f"🌍 Translated Text: `{text_to_analyze}`")
            except Exception as e:
                st.error("Translation failed. Proceeding with original text.")

        # Step 2: Prediction
        X = tfidf.transform([text_to_analyze])
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X).max()

        st.markdown("---")
        if "not" in prediction.lower():
            st.success(f"✅ Safe comment ({confidence:.2%} confidence)")
        else:
            st.error(f"🚫 Cyberbullying Detected ({confidence:.2%} confidence)")

        # Confidence progress bar
        st.progress(float(confidence))
        st.markdown("---")

        # Step 3: Explainability (LIME)
        if explain:
            explainer = LimeTextExplainer(class_names=["Not Bullying", "Bullying"])

            def predict_proba_wrapper(texts):
                X = tfidf.transform(texts)
                return model.predict_proba(X)

            exp = explainer.explain_instance(text_to_analyze, predict_proba_wrapper, num_features=8)

            st.markdown("### 🔍 Words Impacting Prediction")
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)

        # Step 4: Logging to CSV
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "comment": user_input,
            "translated": text_to_analyze,
            "prediction": prediction,
            "confidence": round(confidence, 4),
        }

        os.makedirs("logs", exist_ok=True)
        log_file = "logs/predictions.csv"

        # Append or create CSV file
        if os.path.exists(log_file):
            pd.DataFrame([log_data]).to_csv(log_file, mode="a", header=False, index=False)
        else:
            pd.DataFrame([log_data]).to_csv(log_file, index=False)

        st.success("💾 Prediction logged successfully!")

# ----------------------------------------
# Footer
# ----------------------------------------
st.caption("Built with ❤️ using Streamlit, Scikit-Learn, and LIME")
