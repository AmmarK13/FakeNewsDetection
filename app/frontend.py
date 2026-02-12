import streamlit as st
import joblib

# Load model & vectorizer
tfidf_vectorizer = joblib.load(r"/root/Cyber/FakeNewsDetection/notebook/tfidf_vector.pkl")
model = joblib.load(r"/root/Cyber/FakeNewsDetection/notebook/fake_news_model.pkl")

# Streamlit UI
st.title("Fake News Detector 📰")
st.write("Paste a news headline or article below:")

# User input
user_input = st.text_area("News text")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Transform input
        input_vect = tfidf_vectorizer.transform([user_input])
        prediction = model.predict(input_vect)[0]

        # Display result
        if prediction == "FAKE":
            st.error("🚨 This news is likely FAKE!")
        else:
            st.success("✅ This news is likely REAL.")
