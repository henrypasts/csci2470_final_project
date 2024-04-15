from sklearn.metrics import confusion_matrix
import numpy as np


def categorize_value(x, bucket_ranges):
    # Handle edge cases explicitly
    if x <= bucket_ranges[0]:
        return 0  # First bucket
    elif x > bucket_ranges[-2]:  # Account for inclusive upper bound in the last bucket
        return len(bucket_ranges) - 2  # Last bucket index
    else:
        # Digitize, accounting for the buckets
        bucket_index = np.digitize(x, bucket_ranges, right=True)
        # Adjust to ensure distinct categorization at 0
        if bucket_index > 0 and bucket_ranges[bucket_index-1] == 0:
            return bucket_index - 1
        else:
            return bucket_index - 1


def print_confusion_matrix_stats(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)  # Get the confusion matrix
    TN, FP, FN, TP = cm.ravel()  # Unpacking the confusion matrix

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    TNR = TN / (TN + FP)

    print(f"% of Positive Labels Correctly Classified (TPR): {round(TPR * 100, 2)}%")
    print(f"% of Negative Labels Misclassified as Positive (FPR): {round(FPR * 100, 2)}%")
    print(f"% of Positive Labels Misclassified as Negative (FNR): {round(FNR * 100, 2)}%")
    print(f"% of Negative Labels Correctly Classified (TNR): {round(TNR * 100, 2)}%")
