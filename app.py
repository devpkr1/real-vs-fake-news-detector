import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK resources are downloaded
def download_nltk_resources():
    resources = ["stopwords", "punkt"]
    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}" if resource == "stopwords" else f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource)

download_nltk_resources()

# Load the pre-trained model and vectorizer
model_path = "random_forest_model.pkl"  # Update with your model file name
vectorizer_path = "vectorizer.pkl"  # Update with your vectorizer file name

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(news):
    # Remove non-alphabetic characters
    news = re.sub(r'[^a-zA-Z\s]', '', news)
    # Convert to lowercase
    news = news.lower()
    # Tokenize and lemmatize
    tokens = nltk.word_tokenize(news)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(processed_tokens)

# Prediction function
def predict_news(news):
    processed_news = preprocess_text(news)
    vectorized_news = vectorizer.transform([processed_news])
    prediction = model.predict(vectorized_news)
    return "Real News 📰" if prediction[0] == 1 else "Fake News 📰"

# Streamlit app
st.title("Real vs Fake News Detector")
st.write("Enter the news article below to determine whether it is **real** or **fake**.")

# Input text from user
news_input = st.text_area("News Article", placeholder="Type or paste the news article here...")

# Predict button
if st.button("Predict"):
    if news_input.strip():  # Ensure the input is not empty
        prediction = predict_news(news_input)
        st.subheader(f"Prediction: {prediction}")
    else:
        st.error("Please enter a news article to predict.")

# Footer
st.write("---")
st.write("Developed with ❤️ using Streamlit.")
