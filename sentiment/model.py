import numpy as np
from keras.layers import LSTM, Dense, BatchNormalization, Dropout, LeakyReLU
import pandas as pd
import tensorflow as tf
import plotly.graph_objs as go
import plotly.io as pio
from sklearn import preprocessing

class btcLSTM(tf.keras.Model):

    def __init__(self, lstm_size=300):
        super().__init__()

        self.rnn = LSTM(lstm_size, return_sequences=False, return_state=True)
        self.dense1 = Dense(512, activation=LeakyReLU())
        # self.batch_norm = BatchNormalization()
        # self.dropout = Dropout(0.2)
        self.dense2 = Dense(220, activation=LeakyReLU())
        self.dense3 = Dense(1, activation='linear')

    def call(self, inputs):
        logits, _, _ = self.rnn(inputs)
        logits = self.dense1(logits)
        logits = self.dense2(logits)
        # logits = self.batch_norm(logits)
        # logits = self.dropout(logits)
        logits = self.dense3(logits)
        return logits

def sign_loss(y_true, y_pred):

    absolute_error = tf.abs(y_true - y_pred)
    
    sign_penalty = tf.cast(tf.not_equal(tf.sign(y_true), tf.sign(y_pred)), dtype=tf.float32)
    
    loss = absolute_error + 10 * sign_penalty
    
    return loss

def main():
    CLEANED_TWEETS_W_SENTIMENT = 'data/final_tweets.csv'

    data = pd.read_csv(CLEANED_TWEETS_W_SENTIMENT,sep=';')
    data = data.dropna(axis=0)

    X = data[['Percent Change', 'compound']]

    y = data['Percent Change'].shift(-1)

    y = y.iloc[:-1]

    X = data[['Percent Change', 'compound']].iloc[:-1]


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    seq_length = 10
    X_train_seq = np.array([X_train.iloc[i:i+seq_length] for i in range(len(X_train)-seq_length+1)])
    X_test_seq = np.array([X_test.iloc[i:i+seq_length] for i in range(len(X_test)-seq_length+1)])
    y_train_seq = np.array([y_train.iloc[i+seq_length-1] for i in range(len(X_train)-seq_length+1)])
    y_test_seq = np.array([y_test.iloc[i+seq_length-1] for i in range(len(X_test)-seq_length+1)])

    model = btcLSTM()

    model.compile(optimizer='adam', loss=sign_loss, metrics=[tf.keras.metrics.MeanSquaredError()])

    model.fit(X_train_seq, y_train_seq, epochs=5, batch_size=32, validation_data=(X_test_seq, y_test_seq))

    loss = model.evaluate(X_test_seq, y_test_seq)
    print(f'Test loss: {loss}')

    y_pred = model.predict(X_test_seq)



    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(y_test_seq)), y=y_test_seq, mode='markers', name='Actual'))
    fig.add_trace(go.Scatter(x=np.arange(len(y_pred)), y=y_pred.flatten(), mode='markers', name='Predicted'))
    fig.update_layout(title='Actual vs Predicted Percent Change', xaxis_title='Time', yaxis_title='Percent Change')
    pio.show(fig)

    # True value - predicted value
    residual = y_test_seq - y_pred.flatten()

    # Plot residuals
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(residual)), y=residual, mode='lines', name='Residuals'))
    fig.update_layout(title='Residuals (Actual - Predicted) vs Time', xaxis_title='Time', yaxis_title='Residual')
    pio.show(fig)

if __name__ == '__main__':
    main()