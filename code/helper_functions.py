from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns


def categorize_value(x, bucket_ranges):
    bucket_index = np.digitize(x, bucket_ranges, right=False)
    return bucket_index


def print_confusion_matrix_stats(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)  # Get the confusion matrix
    TN, FP, FN, TP = cm.ravel()  # Unpacking the confusion matrix

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.show()

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    TNR = TN / (TN + FP)

    print(f"% of Positive Labels Correctly Classified (TPR): {round(TPR * 100, 2)}%")
    print(f"% of Negative Labels Misclassified as Positive (FPR): {round(FPR * 100, 2)}%")
    print(f"% of Positive Labels Misclassified as Negative (FNR): {round(FNR * 100, 2)}%")
    print(f"% of Negative Labels Correctly Classified (TNR): {round(TNR * 100, 2)}%")


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


def calculate_last_accuracy(y_true, y_pred):
    # Convert predictions to class indices
    y_pred_indices = np.argmax(y_pred, axis=1)
    # Calculate accuracy
    accuracy = np.mean(y_pred_indices == y_true)
    return accuracy


def simulate_trading(dates, y_last_pred_cat, y_percents, name="Validation"):
    # Convert numeric timestamps to datetime objects
    dates = pd.to_datetime(dates)

    # Initialize portfolio values
    buy_hold_port_values = []
    strategy_port_values = []
    buy_hold_port = 10_000
    strategy_port = 10_000
    buy = False
    number_of_moves = 0

    for i in range(len(y_last_pred_cat)):
        if y_last_pred_cat[i] == 1:
            strategy_port = strategy_port * (1 + y_percents[i])
        strategy_port_values.append(strategy_port)
        buy_hold_port = buy_hold_port * (1 + y_percents[i])
        buy_hold_port_values.append(buy_hold_port)
        if buy:
            if y_last_pred_cat[i] == 0:
                buy = False
                number_of_moves += 1
        else:
            if y_last_pred_cat[i] == 1:
                buy = True

    # Output performance
    print(f"{name} Buy and Hold Return: {round((buy_hold_port - 10_000) / 10_000 * 100, 2)}%")
    print(f"{name} Strategy Return: {round((strategy_port - 10_000) / 10_000 * 100, 2)}%")
    print("Number of moves:", number_of_moves)

    # Plot the portfolio values
    plt.figure(figsize=(12, 8))
    plt.plot(dates, strategy_port_values, label='Strategy', color='blue', linewidth=2)
    plt.plot(dates, buy_hold_port_values, label='Buy and Hold', color='red', linewidth=2)
    plt.title(f"{name} Portfolio Values", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Portfolio Value ($)", fontsize=14)
    plt.legend(loc='upper left', fontsize="medium")
    plt.grid(True)

    # Set date formatting on the x-axis
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"figs/{name}_port_values.png")
    plt.show()


class EnhancedTopNReturnCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, cat_at_zero, checkpoint_path='tmp'):
        super().__init__()
        # self.training_data = training_data
        self.validation_data = validation_data
        self.checkpoint_path = checkpoint_path
        self.cat_at_zero = cat_at_zero
        self.best_train = -np.inf
        self.best_val = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        # train_return, train_dd = self.calculate_return(self.training_data)
        val_return, val_dd = self.calculate_return(self.validation_data)

        # Save the model with the best validation top n return
        if val_return > self.best_val:
            self.best_val = val_return
            print("VALIDATION RETURN IMPROVED\n")
            # print(f'\nEpoch {epoch + 1}: {train_return}% and Max Drawdown = {train_dd}%')
            print(f'\nEpoch {epoch + 1}: Return Improved to {val_return}% and Max Drawdown = {val_dd}%')
            self.model.save(self.checkpoint_path)

    def calculate_return(self, data):
        x, y_percents = data
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

