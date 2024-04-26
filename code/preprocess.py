import pandas as pd
import numpy as np
import tensorflow as tf
from helper_functions import categorize_value
import os
import datetime
import matplotlib.pyplot as plt


def get_2019_data():
    input_path = 'bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv'
    output_path = 'bitcoin_1-min_data_2019.csv'
    chunk_size = 10000

    start_date = datetime.datetime(2019, 7, 1)
    end_date = datetime.datetime(2019, 12, 31)

    # Create an iterator to load chunks:
    reader = pd.read_csv(input_path, chunksize=chunk_size)

    # Check if the output file exists and set header accordingly:
    header = not os.path.exists(output_path)

    # Process each chunk
    for chunk in reader:
        # Ensure the 'Timestamp' column is a datetime object
        chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], unit='s')

        filtered_chunk = chunk[(chunk['Timestamp'] >= start_date) & (chunk['Timestamp'] <= end_date)]
        # Append each filtered chunk to the output CSV, managing the header
        filtered_chunk.to_csv(output_path, mode='a', index=False, header=header)
        header = False  # Ensure header is not written again


def window_data(data, data_percents, dates, window):
    x, y = [], []
    y_percents = []
    y_dates = []
    for i in range(0, len(data) - window):
        if i + window + 1 < len(data):
            x.append(data[i: i + window])
            y.append(data[i + 1: i + window + 1])
            y_percents.append(data_percents[i + window + 1])
            y_dates.append(dates[i + window + 1])
        else:
            continue
    x, y = tf.convert_to_tensor(x, dtype=tf.int32), tf.convert_to_tensor(y, dtype=tf.int32)
    y_percents = np.array(y_percents)
    y_dates = np.array(y_dates)
    return x, y, y_percents, y_dates


def preprocess(df: pd.DataFrame, window_size: int = 20):

    # drop nan rows:
    df = df.dropna()

    # get percent change:
    df["Percent Change"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # split the data in 80-10-10 for train, validation, and test
    train_df = df.iloc[:int(len(df) * 0.8)]
    validation_df = df.iloc[int(len(df) * 0.8): int(len(df) * 0.9)]
    test_df = df.iloc[int(len(df) * 0.9):]

    # print Timestamp range of train, validation, and test sets:
    print("Train Timestamp Range: ", train_df['Timestamp'].min(), train_df['Timestamp'].max())
    print("Validation Timestamp Range: ", validation_df['Timestamp'].min(), validation_df['Timestamp'].max())
    print("Test Timestamp Range: ", test_df['Timestamp'].min(), test_df['Timestamp'].max())

    plt.style.use('seaborn-darkgrid')

    # Create the histogram
    plt.figure(figsize=(12, 8))
    n, bins, patches = plt.hist(train_df['Percent Change'] * 100, bins=100, color='skyblue', edgecolor='black')
    # Set the title and labels with improved formatting
    plt.title('Histogram of Percent Change', fontsize=15)
    plt.xlabel('Percent Change (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.gca().set_xticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_xticks()])  # Format x-axis with '%' sign
    # Add a vertical line for the mean or median:
    mean_val = np.mean(train_df['Percent Change'] * 100)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1)
    plt.text(mean_val + 0.5, plt.gca().get_ylim()[1] * 0.9, f'Mean: {mean_val:.2f}%', color='red')
    plt.savefig("figs/train_histogram_percent_change.png")

    train_df, validation_df, test_df = (train_df[["Timestamp", "Percent Change"]].reset_index(drop=True),
                                        validation_df[["Timestamp", "Percent Change"]].reset_index(drop=True),
                                        test_df[["Timestamp", "Percent Change"]].reset_index(drop=True))

    # generate upper and lower bounds (Note: +/- 0.70% used based on the histogram)
    lower_bound = -0.0070  # train_df['Percent Change'].min()
    upper_bound = 0.0070  # train_df['Percent Change'].max()

    # generate the buckets:
    step_size = 0.00025  # This is 0.025% or 2.5 bps
    negative_buckets = np.arange(-step_size, lower_bound - step_size, -step_size)[::-1]
    positive_buckets = np.arange(0, upper_bound + step_size, step_size)

    # Concatenate to form the full range of buckets, ensuring 0 is included once
    bucket_ranges = np.concatenate((negative_buckets, positive_buckets))

    # The category of the bucket that has its lower bound at 0
    category_at_zero = len(negative_buckets)
    # apply the categorize_value function to the train_df, validation_df, and test_df:
    train_df['Category'] = train_df['Percent Change'].apply(lambda x: categorize_value(x, bucket_ranges))
    validation_df['Category'] = validation_df['Percent Change'].apply(lambda x: categorize_value(x, bucket_ranges))
    test_df['Category'] = test_df['Percent Change'].apply(lambda x: categorize_value(x, bucket_ranges))

    # get distribution of categories in the train, validation, and test sets and save to excel:
    train_df['Category'].value_counts().to_excel("train_category_distribution.xlsx")
    validation_df['Category'].value_counts().to_excel("validation_category_distribution.xlsx")
    test_df['Category'].value_counts().to_excel("test_category_distribution.xlsx")

    # train data:
    train = train_df['Category'].values
    train_percents = train_df['Percent Change'].values
    train_dates = train_df['Timestamp'].values

    # validation data:
    validation = validation_df['Category'].values
    validation_percents = validation_df['Percent Change'].values
    val_dates = validation_df['Timestamp'].values

    # test data:
    test = test_df['Category'].values
    test_percents = test_df['Percent Change'].values
    test_dates = test_df['Timestamp'].values

    # get the train, validation, and test window_data:
    x_train, y_train, y_train_percents, train_dates = window_data(train, train_percents, train_dates, window_size)
    x_validation, y_validation, y_validation_percents, val_dates = window_data(validation, validation_percents, val_dates, window_size)
    x_test, y_test, y_test_percents, test_dates = window_data(test, test_percents, test_dates, window_size)

    # get category size:
    category_size = len(bucket_ranges) + 1

    # plot train histogram:
    plt.figure(figsize=(12, 8))
    plt.hist(train, bins=category_size)
    plt.title('Histogram of Categories in Train Set')
    plt.xlabel('Categories')
    plt.ylabel('Frequency')
    plt.savefig("figs/train_histogram_categories.png")
    # plt.show()

    # plot validation histogram:
    plt.figure(figsize=(12, 8))
    plt.hist(validation, bins=category_size)
    plt.title('Histogram of Categories in Validation Set')
    plt.xlabel('Categories')
    plt.ylabel('Frequency')
    plt.savefig("figs/validation_histogram_categories.png")
    # plt.show()

    # plot test histogram:
    plt.figure(figsize=(12, 8))
    plt.hist(test, bins=category_size)
    plt.title('Histogram of Categories in Test Set')
    plt.xlabel('Categories')
    plt.ylabel('Frequency')
    plt.savefig("figs/test_histogram_categories.png")
    # plt.show()

    # Plot all histograms on the same plot:
    plt.figure(figsize=(12, 8))
    plt.hist([train, validation, test], bins=category_size, label=['Train', 'Validation', 'Test'], alpha=0.8)
    plt.title('Histogram of Categories Across Train, Validation, and Test Sets')
    plt.xlabel('Categories')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig("figs/train_val_test_histogram_categories.png")
    # plt.show()

    return (x_train, y_train, y_train_percents, x_validation, y_validation, y_validation_percents, x_test, y_test,
            y_test_percents, category_at_zero, category_size, train_dates, val_dates, test_dates)

