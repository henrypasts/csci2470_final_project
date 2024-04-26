import pandas as pd
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from transformer import Transformer, CustomSchedule
from helper_functions import print_confusion_matrix_stats, get_category, calculate_accuracy, \
    calculate_last_accuracy, simulate_trading
from preprocess import preprocess


def main():
    df = pd.read_csv("bitcoin_1-min_data_2019.csv")
    x_train, y_train, y_train_percents, x_val, y_val, y_val_percents, x_test, y_test, y_test_percents, \
        category_at_zero, category_size, train_dates, val_dates, test_dates = preprocess(df, window_size=10)

    # checkpoint_filepath = 'tmp'
    #
    # # Assuming x_val, y_val are your validation datasets
    # total_return_callback = EnhancedTopNReturnCallback(validation_data=(x_val, y_val_percents),
    #                                                    cat_at_zero=category_at_zero,
    #                                                    checkpoint_path=checkpoint_filepath)

    # Transformer parameters:
    num_layers = 2
    d_model = 128
    num_heads = 4
    dff = 128
    input_vocab_size = category_size
    target_vocab_size = category_size
    dropout_rate = 0.6

    # Instantiate the Transformer:
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
    learning_rate = CustomSchedule(64)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # loss fn:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False
    )

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[tf.metrics.SparseCategoricalAccuracy()])

    # training input and target data preparation
    train_inputs = (x_train, y_train)

    # validation input and target data preparation (pass in the same input and target)
    val_inputs = (x_val, x_val)

    # test input and target data preparation (pass in the same input and target)
    test_inputs = (x_test, x_test)

    checkpoint_filepath = 'tmp/checkpoint'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_sparse_categorical_accuracy',
        mode='min',
        save_best_only=True)

    # Train the model
    history = model.fit(x=train_inputs, y=y_train, batch_size=32, epochs=2, validation_data=(val_inputs, y_val),
                        callbacks=[model_checkpoint_callback])

    # # plot the training and validation loss:
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig("figs/training_validation_loss.png")

    # Load the best model:
    model.load_weights(checkpoint_filepath)

    # Make predictions
    # y_train_pred = model.predict(train_inputs)
    y_val_pred = model.predict(val_inputs)
    y_test_pred = model.predict(test_inputs)

    # Extract the last predictions for training and validation sets
    # y_train_last_pred = np.array([y_pred[-1] for y_pred in y_train_pred])
    y_val_last_pred = np.array([y_pred[-1] for y_pred in y_val_pred])
    y_test_last_pred = np.array([y_pred[-1] for y_pred in y_test_pred])

    # Extract the last true labels for training and validation sets
    # y_train_last_true = np.array([y_true[-1] for y_true in y_train])
    y_val_last_true = np.array([y_true[-1] for y_true in y_val])
    y_test_last_true = np.array([y_true[-1] for y_true in y_test])

    # Get binary classification for predictions and true labels:
    # y_train_last_pred_cat = get_category(y_train_last_pred, category_at_zero)
    y_val_last_pred_cat = get_category(y_val_last_pred, category_at_zero)
    y_test_last_pred_cat = get_category(y_test_last_pred, category_at_zero)

    # y_train_last_true_cat = np.where(y_train_last_true < category_at_zero, 0, 1)
    y_val_last_true_cat = np.where(y_val_last_true < category_at_zero, 0, 1)
    y_test_last_true_cat = np.where(y_test_last_true < category_at_zero, 0, 1)

    # simulate trading:
    # simulate_trading(y_train_last_pred_cat, y_train_percents, name="Training")
    simulate_trading(val_dates, y_val_last_pred_cat, y_val_percents, name="Validation")
    simulate_trading(test_dates, y_test_last_pred_cat, y_test_percents, name="Test")

    # Print confusion matrix stats:
    # print("Train Confusion Matrix:")
    # print_confusion_matrix_stats(y_train_last_true_cat, y_train_last_pred_cat)
    print("\n")
    print("Validation Confusion Matrix:")
    print_confusion_matrix_stats(y_val_last_true_cat, y_val_last_pred_cat)
    print("\n")
    print("Test Confusion Matrix:")
    print_confusion_matrix_stats(y_test_last_true_cat, y_test_last_pred_cat)
    print("\n")

    # Calculate accuracy for training and validation sets
    # train_accuracy = calculate_accuracy(y_train_last_true_cat, y_train_last_pred_cat)
    val_accuracy = calculate_accuracy(y_val_last_true_cat, y_val_last_pred_cat)
    test_accuracy = calculate_accuracy(y_test_last_true_cat, y_test_last_pred_cat)

    # print(f"Binary Training Accuracy: {train_accuracy}")
    print(f"Binary Validation Accuracy: {val_accuracy}")
    print(f"Binary Test Accuracy: {test_accuracy}")

    # Calculate accuracy for the last predictions
    # train_last_accuracy = calculate_last_accuracy(np.array(y_train_last_true), np.array(y_train_last_pred))
    val_last_accuracy = calculate_last_accuracy(np.array(y_val_last_true), np.array(y_val_last_pred))
    test_last_accuracy = calculate_last_accuracy(np.array(y_test_last_true), np.array(y_test_last_pred))

    # print(f"Categorical Last prediction accuracy on training set: {train_last_accuracy}")
    print(f"Categorical Last prediction accuracy on validation set: {val_last_accuracy}")
    print(f"Categorical Last prediction accuracy on test set: {test_last_accuracy}")

    # plot predictions over time (Validation):
    y_val_last_pred_indices = np.argmax(y_val_last_pred, axis=1)
    # Plotting
    plt.figure(figsize=(14, 12))
    # Plot test last predictions and true labels
    plt.plot(pd.to_datetime(val_dates), y_val_last_pred_indices, label='Validation Predictions', marker='o', linestyle='none', markersize=5)
    plt.plot(pd.to_datetime(val_dates), y_val_last_true, label='Validation True', marker='x', linestyle='none', markersize=5)
    plt.title('Validation Set: Last Prediction vs. True')
    plt.xlabel('Date')
    plt.ylabel('Category')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figs/validation_last_pred_true_over_time.png")
    plt.show()

    # Plotting
    plt.figure(figsize=(14, 12))
    # Plot test last predictions and true labels
    plt.plot(pd.to_datetime(val_dates[-100:]), y_val_last_pred_indices[-100:], label='Validation Predictions', marker='o',
             linestyle='none', markersize=5)
    plt.plot(pd.to_datetime(val_dates[-100:]), y_val_last_true[-100:], label='Validation True', marker='x', linestyle='none',
             markersize=5)
    plt.title('Validation Set: Last Prediction vs. True Last 100 Values')
    plt.xlabel('Date')
    plt.ylabel('Category')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figs/validation_last_pred_true_over_time_last_100.png")
    plt.show()

    # plot predictions over time (Test):
    y_test_last_pred_indices = np.argmax(y_test_last_pred, axis=1)
    # Plotting
    plt.figure(figsize=(14, 12))
    # Plot test last predictions and true labels
    plt.plot(pd.to_datetime(test_dates), y_test_last_pred_indices, label='Test Predictions', marker='o', linestyle='none', markersize=5)
    plt.plot(pd.to_datetime(test_dates), y_test_last_true, label='Test True', marker='x', linestyle='none', markersize=5)
    plt.title('Test Set: Last Prediction vs. True')
    plt.xlabel('Date')
    plt.ylabel('Category')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figs/test_last_pred_true_over_time.png")
    plt.show()

    # Plotting
    plt.figure(figsize=(14, 12))
    # Plot test last predictions and true labels
    plt.plot(pd.to_datetime(test_dates[-100:]), y_test_last_pred_indices[-100:], label='Test Predictions', marker='o',
             linestyle='none', markersize=5)
    plt.plot(pd.to_datetime(test_dates[-100:]), y_test_last_true[-100:], label='Test True', marker='x', linestyle='none',
             markersize=5)
    plt.title('Test Set: Last Prediction vs. True Last 100 Values')
    plt.xlabel('Date')
    plt.ylabel('Category')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figs/test_last_pred_true_over_time_last_100.png")
    plt.show()

    # # plot scatter of y_train_last_pred_indices vs y_train_last_true:
    # plt.figure(figsize=(7, 6))
    # plt.scatter(y_train_last_true, y_train_last_pred_indices)
    # plt.title('Scatter of Training Last Prediction vs. True')
    # plt.xlabel('True')
    # plt.ylabel('Prediction')
    # plt.savefig("figs/train_scatter.png")
    # plt.show()

    # plot scatter of y_val_last_pred_indices vs y_val_last_true:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_val_last_true, y_val_last_pred_indices)
    plt.title('Scatter of Validation Last Prediction vs. True')
    plt.xlabel('True')
    plt.ylabel('Prediction')
    plt.savefig("figs/val_scatter.png")
    plt.show()

    # plot scatter of y_test_last_pred_indices vs y_test_last_true:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test_last_true, y_test_last_pred_indices)
    plt.title('Scatter of Test Last Prediction vs. True')
    plt.xlabel('True')
    plt.ylabel('Prediction')
    plt.savefig("figs/test_scatter.png")
    plt.show()

    # # plot the training residuals:
    # plt.figure(figsize=(7, 6))
    # residuals = y_train_last_true - y_train_last_pred_indices
    # plt.scatter(y_train_last_true, residuals)
    # plt.title('Scatter of Training Residuals')
    # plt.xlabel('True')
    # plt.ylabel('Residuals')
    # plt.savefig("figs/train_residuals.png")
    # plt.show()

    # plot the validation residuals:
    plt.figure(figsize=(7, 6))
    residuals = y_val_last_true - y_val_last_pred_indices
    plt.scatter(y_val_last_true, residuals)
    plt.title('Scatter of Validation Residuals')
    plt.xlabel('True')
    plt.ylabel('Residuals')
    plt.savefig("figs/val_residuals.png")
    plt.show()

    # plot the test residuals:
    plt.figure(figsize=(7, 6))
    residuals = y_test_last_true - y_test_last_pred_indices
    plt.scatter(y_test_last_true, residuals)
    plt.title('Scatter of Test Residuals')
    plt.xlabel('True')
    plt.ylabel('Residuals')
    plt.savefig("figs/test_residuals.png")
    plt.show()


if __name__ == "__main__":
    main()

