import os
import re
import joblib
import logging
import tweepy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO)

# Download NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load trained model and vectorizer
model = joblib.load('cyberbully_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Twitter API credentials from environment variables
api_key = os.getenv('TWITTER_API_KEY')
api_key_secret = os.getenv('TWITTER_API_SECRET')
access_token = os.getenv('TWITTER_ACCESS_TOKEN')
access_token_secret = os.getenv('TWITTER_ACCESS_SECRET')
bearer_token = os.getenv('TWITTER_BEARER_TOKEN')

# Authentication
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Preprocessing function
def preprocess_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#','', tweet)
    tweet = re.sub(r'[^A-Za-z0-9 ]+', '', tweet)
    tweet = tweet.lower()
    tweet = ' '.join([word for word in tweet.split() if word not in stop_words])
    return tweet

# Stream listener class
class CyberbullyStream(tweepy.StreamingClient):
    def on_tweet(self, tweet):
        try:
            cleaned = preprocess_tweet(tweet.text)
            transformed = vectorizer.transform([cleaned])
            prediction = model.predict(transformed)[0]
            label = "Cyberbullying" if prediction == 1 else "Not Cyberbullying"
            logging.info(f"Tweet: {tweet.text}\nPrediction: {label}")
        except Exception as e:
            logging.error(f"Error processing tweet: {e}")

# Set up stream
stream = CyberbullyStream(bearer_token)

# Delete old rules and add new rule
stream.delete_all_rules()
stream.add_rules(tweepy.StreamRule("bully OR abuse OR hate"))  # filter by keywords

# Start streaming
logging.info("Starting real-time tweet classification...")
stream.filter(tweet_fields=["text"])
