from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import pandas as pd

CLEANED_TWEETS_CSV = 'data/english_tweets_sample.csv'
SENTIMENT_TWEETS_CSV = 'data/sentiment_tweets.csv'

sentiment = SentimentIntensityAnalyzer()

def get_sentiment(text):
    """
    Returns sentiment of text using vader 
    """
    sentiment_score = sentiment.polarity_scores(text)
    sentiment_compound = sentiment_score['compound']
    # print(text, sentiment_score)
    return sentiment_score, sentiment_compound

def get_sentiment_data():
    print("Running sentiment data...")
    df = pd.read_csv(CLEANED_TWEETS_CSV, 
                    sep=';', 
                    nrows=None,
                    usecols=['timestamp','text'],         
                    dtype={'timestamp': 'str', 'text': 'str'}
                    )
    tqdm.pandas()
    df['sentiment'], df['compound'] = zip(*df['text'].progress_apply(get_sentiment))
    df.to_csv(SENTIMENT_TWEETS_CSV, index=False, sep=';')     # Save sentiment data
    return df

if __name__ == '__main__':
    get_sentiment_data()