import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

class btcLSTM(tf.keras.Model):

    ##########################################################################################

    def __init__(self, vocab_size=1, rnn_size=64, embed_size=64):

        super().__init__()

        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embed_size = embed_size


        # self.embed = tf.keras.layers.Embedding(self.vocab_size,self.embed_size)
        self.rnn = tf.keras.layers.LSTM(self.rnn_size, return_sequences=False, return_state=True)
        self.dense = tf.keras.layers.Dense(10)
        self.dense2 = tf.keras.layers.Dense(1)


    def call(self, inputs):

        # logits = self.embed(inputs)
        logits, _, _ = self.rnn(inputs)
        logits = self.dense(logits)
        logits = self.dense2(logits)
        # logits = tf.nn.softmax(logits)
        return logits


def main():
    CLEANED_TWEETS_W_SENTIMENT = 'data/final_tweets.csv'

    data = pd.read_csv(CLEANED_TWEETS_W_SENTIMENT,sep=';')
    data = data.dropna(axis=0)

    # X = data[['Percent Change', 'compound']].values
    # y = data['Percent Change'].shift(-1).values[:-1]
    X = data[['Percent Change', 'compound']]
    y = data['Percent Change'].shift(-1)

    print(X.head(7))
    print()
    print(y.head(7))

    print(X.shape)
    print()
    print(y.shape)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    seq_length = 10  
    X_train_seq = np.array([X_train[i:i+seq_length] for i in range(len(X_train)-seq_length+1)])
    X_test_seq = np.array([X_test[i:i+seq_length] for i in range(len(X_test)-seq_length+1)])
    y_train_seq = y_train[seq_length-1:]
    y_test_seq = y_test[seq_length-1:]

    model = btcLSTM()

    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])

    model.fit(X_train_seq, y_train_seq, epochs=5, batch_size=32, validation_data=(X_test_seq, y_test_seq))

    loss = model.evaluate(X_test_seq, y_test_seq)
    print(f'Test loss: {loss}')

    y_pred = model.predict(X_test_seq)

    print(y_pred)

    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(y_test)), y_test, label='Actual', marker='o', s=10)
    plt.scatter(range(len(y_pred.flatten())), y_pred.flatten(), label='Predicted', marker='x', s=200)
    plt.title('Actual vs Predicted Percent Change')
    plt.xlabel('Time')
    plt.ylabel('Percent Change')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()