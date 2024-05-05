import pandas as pd
from tqdm import tqdm
import datetime
from langdetect import detect   
from langid import classify     # faster than langdetect

global USING_CLD3
try:
    import cld3                     # faster than langid
    USING_CLD3 = True
except ImportError:
    USING_CLD3 = False

PROCESS_ALL = True
TWEETS_CSV = 'data/tweets.csv'
TWEETS_CSV_OUT = 'data/cleaned_tweets.csv' if PROCESS_ALL else 'data/cleaned_tweets_sample.csv'
SENTIMENT_CSV = 'data/sentiment_tweets.csv'
BTC_2019_CSV = 'data/bitcoin_1-min_data_2019.csv'   # from other directory
FINAL_CSV = 'data/final_tweets.csv' 

def detect_language(text):
    """
    Detects language of text using cld3
    """
    try:
        if bool(text.strip()) and len(text.strip()) >= 10:
            if USING_CLD3:
                return cld3.get_language(text).language
            else: 
                return classify(text)[0]
        else:
            return "invalid"
    except:
        return "exception"

def clean_data():
    """
    Cleans data by removing non-english tweets
    """
    print("Running clean data...")
    df = pd.read_csv(TWEETS_CSV,
                     sep=';',
                     nrows=None if PROCESS_ALL else 10000,
                     usecols=['timestamp', 'text'],
                     dtype={'timestamp': 'str', 'text': 'str'})

    tqdm.pandas()
    df['lang'] = df['text'].progress_apply(detect_language)
    print(df['lang'].value_counts())

    df = df[df['lang'] == 'en']
    df.to_csv(TWEETS_CSV_OUT, index=False, sep=';')  # Save cleaned data

    return df

def parser(x):
    x = float(x)
    if x >= 0.05:
        return 1
    elif x <= -0.05:
        return -1
    else:
        return 0
    
def convert_sentiment():
    print("converting...")
    df = pd.read_csv(SENTIMENT_CSV, sep=';')
    tqdm.pandas()
    df['compound'] = df['compound'].progress_apply(parser)
    print(df.head())
    print(df['compound'].value_counts())
    df.to_csv(SENTIMENT_CSV, index=False, sep=';')


def preprocess(start_date, end_date, window_sz=10):
    print("running preprocess")

    # Apply sliding window to sentiment
    sentiment_df = pd.read_csv(SENTIMENT_CSV, sep=';')
    sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'], errors='coerce')
    sentiment_df = sentiment_df[(sentiment_df['timestamp'] >= start_date) & (sentiment_df['timestamp'] <= end_date)]
    sentiment_df = sentiment_df.sort_values(by='timestamp')
    sentiment_df['sliding_sentiment'] = sentiment_df['compound'].rolling(window=window_sz).mean()

    # Calculate percent change for BTC data
    btc_df = pd.read_csv(BTC_2019_CSV)    
    btc_df['timestamp'] = pd.to_datetime(btc_df['Timestamp'], errors='coerce')
    btc_df['Percent Change'] = btc_df['Close'].pct_change()

    # Merge sentiment and BTC data
    sentiment_df.set_index('timestamp', inplace=True)
    btc_df.set_index('timestamp', inplace=True)
    # sentiment_df = sentiment_df.reindex(btc_df.index).ffill()
    sentiment_df = sentiment_df.tz_localize(None)  
    btc_df = btc_df.tz_localize(None)  
    print(sentiment_df.head())
    print(btc_df.head())

    final_df = btc_df.join(sentiment_df, how='inner')
    final_df = final_df.drop(columns=['Timestamp', 'Open', 'High', 'Low', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price', 'text', 'sentiment'])
    final_df.to_csv(FINAL_CSV, sep=';')
    print(final_df.head())


if __name__ == '__main__':
    #clean_data()
    #convert_sentiment()
    # start_date = datetime.datetime(2019, 7, 1)
    # end_date = datetime.datetime(2019, 12, 31)
    preprocess('2019-07-01', '2021-12-10', 20)
