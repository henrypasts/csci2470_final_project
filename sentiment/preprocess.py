import pandas as pd
from tqdm import tqdm
from langdetect import detect
from langid import classify     # faster than langdetect
# import cld3                   # faster than langid

PROCESS_ALL = False
TWEETS_CSV = 'data/tweets.csv'
TWEETS_CSV_OUT = 'data/cleaned_tweets.csv' if PROCESS_ALL else 'data/cleaned_tweets_sample.csv'

def detect_language(text):
    """
    Detects language of text using cld3
    """
    try:
        if bool(text.strip()) and len(text.strip()) >= 10:
            # return detect(text)
            return classify(text)[0]
            return cld3.get_language(text).language
        else:
            return "invalid"
    except:
        return "exception"

def clean_data():
    """
    Cleans data by removing non-english tweets
    """
    print("Running clean data...")
    chunks = pd.read_csv(TWEETS_CSV,
                     sep=';',
                     nrows=None if PROCESS_ALL else 10000,
                     usecols=['timestamp', 'text'],
                     dtype={'timestamp': 'str', 'text': 'str'},
                     chunksize=1000)

    # tqdm.pandas()
    # df['lang'] = df['text'].progress_apply(detect_language)
    # print(df['lang'].value_counts())

    filtered_chunks = []
    tqdm.pandas()

    for chunk in chunks:
        chunk['lang'] = chunk['text'].progress_apply(detect_language)
        filtered_chunks.append(chunk)

    df = pd.concat(filtered_chunks)

    df = df[df['lang'] == 'en']
    df.to_csv(TWEETS_CSV_OUT, index=False, sep=';')  # Save cleaned data

    return df

if __name__ == '__main__':
    clean_data()