from flask import Flask, request, render_template
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline

# Download required NLTK data
nltk.download('vader_lexicon')

# Initialize Flask App
app = Flask(__name__)

# Load Pretrained Emotion Model (BERT)
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Function for VADER Sentiment Analysis
def analyze_sentiment_vader(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    
    if sentiment['compound'] >= 0.05:
        return "Happy"
    elif sentiment['compound'] <= -0.05:
        return "Sad"
    else:
        return "Neutral"

# Function for TextBlob Sentiment Analysis
def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0:
        return "Happy"
    elif polarity < 0:
        return "Sad"
    else:
        return "Neutral"

# Function for BERT-Based Emotion Detection
def analyze_sentiment_bert(text):
    result = emotion_model(text)[0]
    emotion_mapping = {
        "joy": "Happy 😊",
        "sadness": "Sad 😢",
        "anger": "Angry 😡",
        "fear": "Fearful 😨",
        "surprise": "Surprised 😲",
        "disgust": "Disgusted 🤢",
        "neutral": "Neutral 😐"
    }
    return emotion_mapping.get(result['label'], "Neutral 😐")

# Web UI (Styled)
@app.route('/')
def home():
    return render_template('index.html')

# API Endpoint for Emotion Analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    detected_emotion = analyze_sentiment_bert(text)

    return render_template('result.html', text=text, emotion=detected_emotion)

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)


