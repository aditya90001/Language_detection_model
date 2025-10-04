import streamlit as st
import pickle

# Load the saved model and CountVectorizer
model = pickle.load(open("tfidf.pkl", "rb"))
vectorizer = pickle.load(open("cv.pkl", "rb"))

# Streamlit UI
st.set_page_config(page_title="Language Detection", layout="centered")
st.title("🌐 Language Detection App")
st.write("Enter a sentence and I'll guess the language!")

# Text input from user
input_text = st.text_area("✍️ Type your sentence here:")

# Predict button
if st.button("Detect Language"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize the input text
        transformed_text = vectorizer.transform([input_text])

        # Make prediction
        prediction = model.predict(transformed_text)

        # Show result
        st.success(f"🌍 Detected Language: **{prediction[0]}**")
