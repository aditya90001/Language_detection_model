import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("tfidf.pkl", "rb"))
vectorizer = pickle.load(open("cv.pkl", "rb"))

# Streamlit UI
st.set_page_config(page_title="Language Detection", layout="centered")
st.title("🌐 Language Detection App")
st.write("Enter a sentence and I'll guess the language!")

# Text input
input_text = st.text_area("✍️ Type your sentence here:")

# Detect language button
if st.button("Detect Language"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize input text
        transformed_text = vectorizer.transform([input_text])
        
        # Predict language
        prediction = model.predict(transformed_text)
        
        # Display result
        st.success(f"🌍 Detected Language: **{prediction[0]}**")
