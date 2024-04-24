import pandas as pd
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from transformer import Transformer, CustomSchedule
from helper_functions import categorize_value, print_confusion_matrix_stats
from transformer_model import TimeSeriesTransformer


def main():
    # load data and apply percent change:
    # df = pd.read_excel("BTC Prices.xlsx", sheet_name="BTC")
    # df["Percent Change"] = df["Price"].pct_change()
    df = pd.read_excel("SPY.xlsx", sheet_name="Sheet1")
    df["Percent Change"] = df["Total Return Index (Gross Dividends)"].pct_change()
    df = df[["Percent Change"]]
    df.dropna(inplace=True)

    # split the data in 80-10-10 for train, validation, and test
    train_df = df.iloc[:int(len(df) * 0.8)]
    validation_df = df.iloc[int(len(df) * 0.8): int(len(df) * 0.9)]
    test_df = df.iloc[int(len(df) * 0.9):]

    # generate upper and lower bounds:
    lower_bound = train_df['Percent Change'].min()
    upper_bound = validation_df['Percent Change'].max()

    # generate the buckets:
    step_size = 0.00250  # This is 0.5%
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

    train = train_df['Category'].values
    train_percents = train_df['Percent Change'].values
    validation = validation_df['Category'].values
    validation_percents = validation_df['Percent Change'].values
    # test = test_df['Category'].values  # TODO: run on test once model is trained
    # test_percents = test_df['Percent Change'].values

    window_size = 20  # hyperparameter
    x_train, y_train = [], []
    y_train_percents = []
    for i in range(0, len(train) - window_size):
        x_train.append(train[i: i + window_size])
        if i + window_size + 1 < len(train):
            y_train.append(train[i + 1: i + window_size + 1])
            y_train_percents.append(train_percents[i + window_size + 1])
        else:
            y_train.append(train[i: i + window_size])
            y_train_percents.append(train_percents[i + window_size])
    x_train, y_train = tf.convert_to_tensor(x_train, dtype=tf.int32), tf.convert_to_tensor(y_train, dtype=tf.int32)
    y_train_percents = np.array(y_train_percents)

    x_val, y_val = [], []
    y_val_percents = []
    for i in range(0, len(validation) - window_size):
        x_val.append(validation[i: i + window_size])
        if i + window_size + 1 < len(validation):
            y_val.append(validation[i + 1: i + window_size + 1])
            y_val_percents.append(validation_percents[i + window_size + 1])
        else:
            y_val.append(validation[i: i + window_size])
            y_val_percents.append(validation_percents[i + window_size])
    x_val, y_val = tf.convert_to_tensor(x_val, dtype=tf.int32), tf.convert_to_tensor(y_val, dtype=tf.int32)
    y_val_percents = np.array(y_train_percents)

    class EnhancedTopNReturnCallback(tf.keras.callbacks.Callback):
        def __init__(self, training_data, validation_data, cat_at_zero, checkpoint_path='best_model.h5'):
            super().__init__()
            self.training_data = training_data
            self.validation_data = validation_data
            self.checkpoint_path = checkpoint_path
            self.cat_at_zero = cat_at_zero
            self.best_train = -np.inf
            self.best_val = -np.inf

        def on_epoch_end(self, epoch, logs=None):
            train_return, train_dd = self.calculate_return(self.training_data)
            val_return, val_dd = self.calculate_return(self.validation_data)

            # Save the model with the best validation top n return
            if val_return > self.best_val:
                self.best_val = val_return
                print("VALIDATION RETURN IMPROVED\n")
                print(f'\nEpoch {epoch + 1}: {train_return}% and Max Drawdown = {train_dd}%')
                print(f'\nEpoch {epoch + 1}: Return Improved to {val_return}% and Max Drawdown = {val_dd}%')
                self.model.save(self.checkpoint_path)

        def calculate_return(self, data):
            x, y, y_percents = data
            y_pred = self.model.predict((x, x))

            y_pred = np.array([pred[-1] for pred in y_pred])
            y_pred = np.argmax(y_pred, axis=1)

            portfolio_values = [10_000]
            for j in range(len(y_pred)):
                predicted_return = y_pred[j]
                if predicted_return >= self.cat_at_zero:
                    actual_return = y_percents[j]
                    portfolio_values.append(portfolio_values[-1] * (1 + actual_return))
                else:
                    portfolio_values.append(portfolio_values[-1])
            total_return = round((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100, 2)
            rolling_max = np.maximum.accumulate(np.array(portfolio_values))
            max_drawdown = round(np.max(1 - (np.array(portfolio_values) / rolling_max)) * 100, 2)
            return total_return, max_drawdown

    checkpoint_filepath = 'tmp'
    # model_checkpoint_callback = ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_weights_only=True,
    #     monitor='sparse_categorical_accuracy',
    #     mode='max',
    #     save_best_only=True)

    # Assuming x_val, y_val are your validation datasets
    total_return_callback = EnhancedTopNReturnCallback(training_data=(x_train, y_train, y_train_percents),
                                                       validation_data=(x_val, y_val, y_val_percents),
                                                       cat_at_zero=category_at_zero,
                                                       checkpoint_path=checkpoint_filepath)

    category_size = len(bucket_ranges) + 1
    # d_model = 128
    # num_heads = 8
    # dff = 512
    # rate = 0.3
    # model = TimeSeriesTransformer(category_size=category_size, d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)

    # Transformer parameters
    num_layers = 5
    d_model = 256
    num_heads = 6
    dff = 256
    input_vocab_size = len(bucket_ranges) + 1
    target_vocab_size = len(bucket_ranges) + 1
    dropout_rate = 0.4

    # Instantiate the Transformer
    model = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        rate=dropout_rate
    )

    # Define the optimizer
    learning_rate = CustomSchedule(256)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # loss fn:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False
    )

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[tf.metrics.SparseCategoricalAccuracy()])

    # Example input and target data preparation
    train_inp = x_train
    train_tar = y_train

    # Packing inputs and masks into dictionaries or using tuples
    train_inputs = (train_inp, train_tar)

    # num_samples = x_val.shape[0]
    # # Create validation targets by shifting x_val by one step in time, but making the last element 0
    # val_tar = np.zeros_like(x_val)
    # for i in range(num_samples):
    #     val_tar[i, :-1] = x_val[i, 1:]
    #     val_tar[i, -1] = input_vocab_size + 1  # Set the last element to a value that is not in the vocabulary

    val_inputs = (x_val, x_val)

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_val shape:", x_val.shape)
    print("y_val shape:", y_val.shape)

    print("Number of Buckets: ", category_size)
    print("Category at Zero: ", category_at_zero)

    # Now use fit
    history = model.fit(x=train_inputs, y=y_train, batch_size=32, epochs=3, validation_data=(val_inputs, y_val),
                        callbacks=[total_return_callback])

    # plot the training and validation loss:
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    model.load_weights(checkpoint_filepath)

    y_train_pred = model.predict(train_inputs)
    y_val_pred = model.predict(val_inputs)

    # Extract the last predictions for training and validation sets
    y_train_last_pred = np.array([y_pred[-1] for y_pred in y_train_pred])
    y_val_last_pred = np.array([y_pred[-1] for y_pred in y_val_pred])

    # Extract the last true labels for training and validation sets
    y_train_last_true = np.array([y_true[-1] for y_true in y_train])
    y_val_last_true = np.array([y_true[-1] for y_true in y_val])

    def get_category(y_pred, category_at_zero):
        # Convert softmax outputs to class indices
        y_pred_indices = np.argmax(y_pred, axis=1)
        # Classify as negative (0) or positive (1)
        y_pred_cat = np.where(y_pred_indices <= category_at_zero, 0, 1)
        return y_pred_cat

    def calculate_accuracy(y_true_cat, y_pred_cat):
        # Calculate accuracy
        accuracy = np.mean(y_true_cat == y_pred_cat)
        return accuracy

    # Get categories for predictions and true labels
    y_train_last_pred_cat = get_category(y_train_last_pred, category_at_zero)
    y_val_last_pred_cat = get_category(y_val_last_pred, category_at_zero)
    y_train_last_true_cat = np.where(y_train_last_true < category_at_zero, 0, 1)
    y_val_last_true_cat = np.where(y_val_last_true < category_at_zero, 0, 1)

    # simulate trading:
    validation_buy_hold_port_values = [10_000]
    validation_strategy_port_values = [10_000]
    validation_buy_hold_port = 10_000
    validation_strategy_port = 10_000
    buy = False
    number_of_moves = 0
    for i in range(0, len(y_val_last_pred_cat)):
        if y_val_last_pred_cat[i] == 1:
            validation_strategy_port = validation_strategy_port * (1 + y_val_percents[i])
        validation_strategy_port_values.append(validation_strategy_port)
        validation_buy_hold_port = validation_buy_hold_port * (1 + y_val_percents[i])
        validation_buy_hold_port_values.append(validation_buy_hold_port)
        if buy:
            if y_val_last_pred_cat[i] == 0:
                buy = False
                number_of_moves += 1
        else:
            if y_val_last_pred_cat[i] == 1:
                buy = True

    print("Validation Buy and Hold Return: ", round((validation_buy_hold_port - 10_000) / 10_000 * 100, 2), "%")
    print("Validation Strategy Return: ", round((validation_strategy_port - 10_000) / 10_000 * 100, 2), "%")
    print("Number of moves: ", number_of_moves)
    # now, plot the port values:
    plt.plot(validation_strategy_port_values, label="Strategy")
    plt.plot(validation_buy_hold_port_values, label="Buy and Hold")
    plt.title("Validation Port Values")
    plt.ylabel("Port Value")
    plt.xlabel("Day")
    plt.legend(loc="upper left")
    plt.show()

    print("Train Confusion Matrix:")
    print_confusion_matrix_stats(y_train_last_true_cat, y_train_last_pred_cat)
    print("\n")
    print("Validation Confusion Matrix:")
    print_confusion_matrix_stats(y_val_last_true_cat, y_val_last_pred_cat)
    print("\n")

    # Calculate accuracy for training and validation sets
    train_accuracy = calculate_accuracy(y_train_last_true_cat, y_train_last_pred_cat)
    val_accuracy = calculate_accuracy(y_val_last_true_cat, y_val_last_pred_cat)

    print(f"Training Accuracy (Positive/Negative): {train_accuracy}")
    print(f"Validation Accuracy (Positive/Negative): {val_accuracy}")

    def calculate_last_accuracy(y_true, y_pred):
        # Convert predictions to class indices
        y_pred_indices = np.argmax(y_pred, axis=1)
        # Calculate accuracy
        accuracy = np.mean(y_pred_indices == y_true)
        return accuracy

    # Calculate accuracy for the last predictions
    train_last_accuracy = calculate_last_accuracy(np.array(y_train_last_true), np.array(y_train_last_pred))
    val_last_accuracy = calculate_last_accuracy(np.array(y_val_last_true), np.array(y_val_last_pred))

    print(f"Last prediction accuracy on training set: {train_last_accuracy}")
    print(f"Last prediction accuracy on validation set: {val_last_accuracy}")

    # Assuming y_train_last_pred, y_val_last_pred, y_train_last_true, y_val_last_true are defined
    # Convert softmax outputs to class indices for predictions
    y_train_last_pred_indices = np.argmax(y_train_last_pred, axis=1)
    y_val_last_pred_indices = np.argmax(y_val_last_pred, axis=1)

    # Plotting
    plt.figure(figsize=(14, 6))

    # Plot training last predictions and true labels
    plt.subplot(1, 2, 1)
    plt.plot(y_train_last_pred_indices, label='Train Predictions', marker='o', linestyle='none', markersize=5)
    plt.plot(y_train_last_true, label='Train True', marker='x', linestyle='none', markersize=5)
    plt.title('Training Set: Last Prediction vs. True')
    plt.xlabel('Sequence Index')
    plt.ylabel('Category')
    plt.legend()

    # Plot validation last predictions and true labels
    plt.subplot(1, 2, 2)
    plt.plot(y_val_last_pred_indices, label='Validation Predictions', marker='o', linestyle='none', markersize=5)
    plt.plot(y_val_last_true, label='Validation True', marker='x', linestyle='none', markersize=5)
    plt.title('Validation Set: Last Prediction vs. True')
    plt.xlabel('Sequence Index')
    plt.ylabel('Category')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()