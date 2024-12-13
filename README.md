# Real vs Fake News Detector

This project is designed to classify news articles as **Real News** or **Fake News** using a pre-trained machine learning model. The tool leverages Natural Language Processing (NLP) and a Random Forest classifier to provide predictions, helping users identify unreliable or fabricated content.

## Project Structure

- **app.py**: The main Streamlit application for user interaction and predictions.
- **random_forest_model.pkl**: The pre-trained Random Forest model used for classification.
- **vector.pkl**: The TF-IDF vectorizer for text preprocessing.
- **requirements.txt**: List of Python dependencies required to run the project.
- **real vs fake news analysis.ipynb**: Notebook containing exploratory data analysis (EDA) and model training details.

## Features

- **Interactive Input Form**: Users can input a news article to determine its authenticity.
- **Real-Time Predictions**: The app predicts whether the news is **Real** or **Fake**.
- **NLP Preprocessing**: Automatically cleans, tokenizes, and vectorizes text input for predictions.

## Dependencies

The project uses several Python libraries, as listed in `requirements.txt`. Key dependencies include:
- `streamlit`: For deploying the interactive web application.
- `nltk`: Used for text preprocessing (tokenization, stopword removal, lemmatization).
- `scikit-learn`: For machine learning model training and evaluation.
- `lightgbm` : For LightGBM Classifier Model.
- `pickle`: For serializing and loading the trained model and vectorizer.

## Setup and Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/devpkr1/real-vs-fake-news-detector.git
   cd real-vs-fake-news-detector
   ```

2. **Install dependencies**:
   Use the following command to install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK resources**:
 Run the following commands to download NLTK's stopwords and tokenizer:
 ```bash
 import nltk
 nltk.download('stopwords')
 nltk.download('punkt')
 ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Access the app**:
   Open a web browser and go to `http://localhost:8501` to interact with the app.

## Usage

1. Open the Streamlit application.
2. Enter a news article in the input field.
3. Click **Predict** to classify the article:
   - **Fake News (0):** Indicates fabricated or unreliable content.
   - **Real News (1):** Indicates authentic and credible content.

## Model Training

The model was trained using a labeled dataset containing real and fake news articles. The training process included:

- Text preprocessing using TF-IDF vectorization.
- Training a Random Forest classifier for classification.
- Evaluating the model with metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

Details of the training process and performance metrics can be found in the `real vs fake news analysis.ipynb`.

## License

This project is licensed under the MIT License.
