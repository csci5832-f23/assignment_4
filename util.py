from torch import logical_and, sum as t_sum


def accuracy(predicted_labels, true_labels):
    """
    Accuracy is correct predictions / all predicitons
    """
    return t_sum(predicted_labels == true_labels)/len(predicted_labels)


def precision(predicted_labels, true_labels):
    """
    Precision is True Positives / All Positives Predictions
    """
    denominator = t_sum(predicted_labels)
    if denominator:
        return t_sum(logical_and(predicted_labels, true_labels))/denominator
    else:
        return 0.


def recall(predicted_labels, true_labels):
    """
    Recall is True Positives / All Positive Labels
    """
    denominator = t_sum(true_labels)
    if denominator:
        return t_sum(logical_and(predicted_labels, true_labels))/denominator
    else:
        return 0.


def f1_score(predicted_labels, true_labels):
    """
    F1 score is the harmonic mean of precision and recall
    """
    P = precision(predicted_labels, true_labels)
    R = recall(predicted_labels, true_labels)
    if P and R:
        return 2*P*R/(P+R)
    else:
        return 0.
