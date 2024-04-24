import pandas as pd
import matplotlib.pyplot as plt

SENTIMENT_TWEETS_CSV = 'data/sentiment_tweets.csv'

def plot_tweets_over_time():
    df = pd.read_csv(SENTIMENT_TWEETS_CSV, sep=';', 
                     nrows=None, 
                     usecols=['timestamp'], 
                     dtype={'timestamp': 'str'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    df = df.dropna(subset=['timestamp'])
    tweets_per_day = df.resample('D', on='timestamp').size()
    print(df['timestamp'].dt.year.unique())
    
    # Plotting
    x_start = pd.Timestamp('2016-01-01')
    x_end = pd.Timestamp('2019-12-31')
    plt.close('all')
    plt.figure(figsize=(12, 6))
    plt.plot(tweets_per_day.index, tweets_per_day.values, label='Tweets per day')
    plt.xlim(x_start, x_end)
    plt.title('Number of Tweets per Day (2016-2019)')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig('plots/tweets_per_day.png')
    plt.show()

if __name__ == '__main__':
    plot_tweets_over_time()