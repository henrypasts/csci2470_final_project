## Introduction
Our project aims to predict the price movements of BTC using past price movements and market sentiment extracted from Twitter. Observing the shortcomings of existing deep learning approaches in accurately tracking cryptocurrency movements, we were motivated to use our own unique architecture and technical data with minute-interval changes, alongside sentiment analysis. Whereas many models aim to predict the future price of BTC, our model will be using a transformer architecture to predict future price change percentage. Albeit unconventional, we will be tokenizing intervals of price history (which is a hyperparameter) and then using transformers to classify them into ranges of price movement. 

The philosophy behind this approach is that BTC has experienced extreme price changes, so a model trying to predict future price may struggle since it will be trained on a massive potential range. Even using a minmax scale, the distribution is not stationary since the mean price is constantly changing. However, the underlying patterns in price movement are more consistent throughout the entire dataset. As mentioned before, we address this problem by focusing on percent changes, which are tracked by grouping similar price changes.

## Related Work
Since our project is 2470 level, a lot of the design decisions are novel. However, we did draw inspiration for our model architecture and, at a higher level, overall strategy from several related papers.

Paper #1: 
- Title: "Prediction of Cryptocurrency prices using Transformers and Long Short term Neural Networks," A. Tanwar and V. Kumar
- Summary: This paper attempts to predict financial time series for Bitcoin using Transformer and LSTM model architectures. It combines both architectures to capture both short-term and long-term dependencies. The results show that this combined approach leads to longer computational times, but outperforms the predictive accuracy of traditional RNN/KNN forecasting models.
- How we can improve: We aim to 1) enhance sentiment analysis and 2) expand the dataset. For sentiment analysis, we plan to extract sentiment signals from beyond the basic sentiment scores used in this paper by potentially using BERT to extract more nuanced insights (reach goal). For the dataset, we are extending the scope of the dataset to additional sources of sentiment data with Tweets and Bloomberg data. We can also investigate potential efficiency gains by experimenting with smaller Transformer variants.
- URL: Prediction of Cryptocurrency prices using Transformers and Long Short term Neural Networks

Additional related work:

- Title: "Predicting Cryptocurrency Price Movement using Generative Transformer (MinGPT)"
Summary: This paper uses a pure transformer approach, specifically minGPT, to predict cryptocurrency price movement. 
URL:https://medium.com/@bondarchukb/predicting-cryptocurrency-price-movement-using-generative-transformer-mingpt-7bc0c3a9304b

- Title: "Attention Transformer with Sentiment on Cryptocurrency Price Prediction"
Summary: This paper investigates the use of attention within Transformers for cryptocurrency price prediction, while using sentiment analysis as an additional feature. It measures the impact of sentiment on price dynamics.
URL: https://doras.dcu.ie/27010/2/Attention__Transformer_with_sentiment_on_cryptocurrency_price_prediction___Complexis_conference.pdf

## Data
To get data on past BTC price, we will be using the [“Bitcoin Historical Data”](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data) dataset, which is publicly available on Kaggle. This dataset is composed of price data at 1-minute intervals, and records activity from 2012 until 2021. 
To get data on BTC sentiment, we will be using the [“Bitcoin tweets - 16m tweets](https://www.kaggle.com/datasets/alaix14/bitcoin-tweets-20160101-to-20190329) dataset, which is also publicly available on Kaggle. The dataset is composed of tweets related to BTC, and record activity from 2016-2019.

When training a model on both BTC price and sentiment data, it would be necessary to split the datasets such that they start and end at the same timestamps - in this case, from 2016 until 2019. However, as implementing sentiment analysis into our model is more so a stretch goal, we intend on training our first few attempts on a wider range of the BTC price data initially.

Furthermore, we will have to clean the BTC price data somewhat. The dataset, while reviewed as highly useable by others, notes that it records some price points as ‘NaN’ if there were technical difficulties in collecting the data (i.e. a given exchange was down). For these cases, we will need to backfill the data. We could either substitute data from another exchange at that same time, or simply use the last known price (such that there is 0% change between the last known timestamp and the unknown timestamp).

## Methodology

We will treat the task of price prediction similar to that of language modeling. We will tokenize the percent changes of our minute data into discrete categories by using our training data. By finding the min and max minute percent changes, we can then create buckets of a step size to get ‘n’ number of categories and classify each minute’s percent change into one of those categories. This allows each category to have an embedding similar to word embeddings. The full architecture is an Encoder-Decoder Transformer with Multi-Headed Attention using positional encoding. For a look-back period, say the previous 20 minutes of price, we feed in the input into the encoder and we feed a masked target into the decoder, along with the output of the encoder to use encoder-decoder Multi-Headed Attention. For each prediction, the most important output is the final one which is used for evaluation.

## Metrics
During training, we use Sparse Categorical Cross-Entropy loss. The metric for evaluation, however, will be focused on evaluating the final prediction of the model’s prediction of the target sequence. For the final prediction, we can evaluate it using MSE or Sparse Categorical Cross Entropy Accuracy if we treat the final output as a categorical variable. If we instead want to view how well the model did at predicting above or below 0, we treat it as a binary classification problem. Using the traditional metrics of accuracy, True Positive Rate, False Positive Rate, False Negative Rate, and True Negative Rate.

In order to measure performance of our regression problem we can use Mean Squared Error (MSE), Mean Absolute Error (MAE) and R Squared to evaluate how well our model is doing to predict the actual level of future price. 

Finally, since we are concerned about how well the model would do simulating if you acted upon its predictions, we will also explore how much money the model would have made versus simply buying and holding BTC over the test period if one were to act on the buy and sell signals generated by the model (i.e. if the model predicted a positive return over the next minute, we would buy or remain invested and if predicted a negative return over the next minute, we would sell and receive 0%).

## Ethics

### Who are the major stakeholders in this problem, and what are the consequences of mistakes made by your algorithm?

When predicting BTC prices we must make sure we understand the consequences of possible mistakes on the stakeholders. The stakeholders here include different types of investors, both individual and institutional. Due to the volatile nature of crypto markets, an error in our model's prediction can lead to a substantial financial loss if the model were to be followed by the user. Ethically, it's crucial to ensure the model's accuracy and reliability, as well as maintaining transparency about the model's limitations, and warn the stakeholders against over-reliance on the model for their trading decisions. Ethically, it is very important to emphasize the fact that the predictions made by the model are not certainly correct and their associated risks to prevent stakeholders from over-relying on the model. This will safeguard the stakeholders against poor financial decision-making driven by an incomplete understanding of the model's potential impacts.

### How are you planning to quantify or measure error or success? What implications does your quantification have?

When using metrics like Sparse Categorical Cross-Entropy, Mean Squared Error (MSE), and binary classification metrics for our model, it is important to understand that these measures do not fully capture the potential financial impact that an error in the model can have on an individual. This oversight is particularly significant in the extremely volatile cryptocurrency market. This limitation can lead to an underestimation or misunderstanding by stakeholders of the risks associated with the model, potentially misleading users about its reliability. The inclusion of financial performance metrics such as those comparing predictions against other investment strategies could be a better metric as it would allow someone to compare the performance of our model with another, enabling them to see the different risks presented by both investment strategies.

## Division of Labor

John Wilkinson: Organizer, will make sure everyone is completing their tasks and that the group is on track to accomplish our goals.

Maxime Seknadje: will help with model parameter tuning and presentation/report in terms of prior work.

Stefano Chiappo: will assist with presentation/report in terms of ethics.

Manoli Angelakis: will assist with presentation/report, including introductions, background, and visuals.

Henry Pasts: will assist with presentation/report and focus on model architecture with help from other group members.


## Note:

Our Base Goal - Our model should be able to predict whether the future price of BTC will go up or down based on past prices with validation accuracy >50%. We will use a classification model trained on BTC price history. Getting a working model at all will be reasonably significant, since we are experimenting with a novel architecture.

Our Target Goal - We aim to maximize our model's validation accuracy, ideally somewhere around 55%. While 55% accuracy may seem low for other applications, in this context it would represent being able to predict positive or negative movement more often than not. 

Our Stretch Goal- We want to try and implement market sentiments within our model. Doing so will require another model to perform sentiment analysis on a dataset of BTC related tweets. From there, we will be able to line it up with our time series data and input both into the transformer architecture described before. This will hopefully give us the opportunity to better predict the price movement BTC. 

