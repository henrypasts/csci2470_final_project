from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import pandas as pd

CLEANED_TWEETS_CSV = 'data/english_tweets_sample.csv'
SENTIMENT_TWEETS_CSV = 'data/sentiment_tweets.csv'

sentiment = SentimentIntensityAnalyzer()

def get_sentiment_vader(text):
    """
    Returns sentiment of text using vader 
    """
    sentiment_raw = sentiment.polarity_scores(text)
    sentiment_compound = sentiment_raw['compound']

    if sentiment_compound >= 0.05:
        sentiment_score = 1
    elif sentiment_compound <= -0.05:
        sentiment_score = -1
    else:
        sentiment_score = 0

    # print(text, sentiment_score)
    return sentiment_score

def get_sentiment_data(method=get_sentiment_vader):
    print(f"Running sentiment data with {method.__name__}...")
    df = pd.read_csv(CLEANED_TWEETS_CSV, 
                    sep=';', 
                    nrows=None,
                    usecols=['timestamp','text'],         
                    dtype={'timestamp': 'str', 'text': 'str'}
                    )
    tqdm.pandas()
    
    df['sentiment'] = df['text'].progress_apply(method)
        
    df = df[['timestamp', 'sentiment']]
    df.to_csv(SENTIMENT_TWEETS_CSV, index=False, sep=';')     # Save sentiment data
    print(df['sentiment'].value_counts())
    return df

if __name__ == '__main__':
    get_sentiment_data()