from numpy import logical_and, sum as t_sum
import spacy


def accuracy(predicted_labels, true_labels):
    """
    Accuracy is correct predictions / all predicitons
    """
    return t_sum(predicted_labels == true_labels)/len(predicted_labels)


def precision(predicted_labels, true_labels, which_label='1'):
    """
    Precision is True Positives / All Positives Predictions
    """
    denominator = t_sum(predicted_labels == which_label)
    predicted_which = predicted_labels == which_label
    true_which = true_labels == which_label
    if denominator:
        return t_sum(logical_and(predicted_which, true_which))/denominator
    else:
        return 0.


def recall(predicted_labels, true_labels, which_label='1'):
    """
    Recall is True Positives / All Positive Labels
    """
    denominator = t_sum(true_labels == which_label)
    predicted_which = predicted_labels == which_label
    true_which = true_labels == which_label
    if denominator:
        return t_sum(logical_and(predicted_which, true_which))/denominator
    else:
        return 0.


def f1_score(predicted_labels, true_labels, which_label='1'):
    """
    F1 score is the harmonic mean of precision and recall
    """
    P = precision(predicted_labels, true_labels, which_label)
    R = recall(predicted_labels, true_labels, which_label)
    if P and R:
        return 2*P*R/(P+R)
    else:
        return 0.
