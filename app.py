import streamlit as st
import joblib

# Must be first Streamlit command
st.set_page_config(page_title="Cyberbullying Detector", page_icon="🛡️")

# Load model and vectorizer
@st.cache_resource
def load_files():
    model = joblib.load('cyber_model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    return model, tfidf

model, tfidf = load_files()

# Title & Description
st.title("🛡️ Cyberbullying Comment Detector")
st.markdown("""
This web app uses a machine learning model to detect **cyberbullying** in online comments.  
It was trained using TF-IDF + Logistic Regression on a real dataset of social media posts.
""")

# Input
user_input = st.text_area("✍️ Enter a comment:", height=150)

# Prediction logic
if st.button("🚀 Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        X = tfidf.transform([user_input])
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X).max()

        st.markdown("---")
        if "not" in prediction.lower():
            st.success(f"✅ **Safe comment** ({confidence:.2%} confidence)")
        else:
            st.error(f"🚫 **Cyberbullying Detected** ({confidence:.2%} confidence)")
        st.markdown("---")

        # Optional confidence bar
        st.progress(float(confidence))

st.caption("Built with ❤️ using Streamlit & Scikit-Learn")
