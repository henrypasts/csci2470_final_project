import pandas as pd
from langdetect import detect
from tqdm import tqdm

TWEETS_CSV = 'data/tweets.csv'

def print_head():
    df = pd.read_csv(TWEETS_CSV, sep=';', nrows=10000, usecols=['timestamp','text'])    
    print(df.head(100))

def is_valid_text(text):
    # Check if text is not empty or too short (e.g., less than 10 characters)
    return bool(text.strip()) and len(text.strip()) >= 10

def detect_language(text):
    try:
        if is_valid_text(text):
            return detect(text)
        else:
            return "invalid"
    except:
        return "exception"

def preprocess():
    print("Running preprocess...")
    df = pd.read_csv(TWEETS_CSV, 
                    sep=';', 
                    usecols=['timestamp','text'],         
                    dtype={'timestamp': 'str', 'text': 'str'}
                    )
    tqdm.pandas()
    df['lang'] = df['text'].progress_apply(detect_language)
    print(df['lang'].value_counts())
    df = df[df['lang'] == 'en']
    df.to_csv('data/english_tweets.csv', index=False, sep=';')


if __name__ == '__main__':
    preprocess()