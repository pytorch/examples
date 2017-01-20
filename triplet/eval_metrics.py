"""Utility methods for computing evaluating metrics. All methods assumes greater
scores for better matches, and assumes label == 1 means match.
"""
import operator


def ErrorRateAt95Recall(labels, scores):
    recall_point = 0.95
    # Sort label-score tuples by the score in descending order.
    sorted_scores = zip(labels, scores)
    sorted_scores.sort(key=operator.itemgetter(1), reverse=True)

    # Compute error rate
    n_match = sum(1 for x in sorted_scores if x[0] == 1)
    n_thresh = recall_point * n_match
    tp = 0
    count = 0
    for label, score in sorted_scores:
        count += 1
        if label == 1:
            tp += 1
        if tp >= n_thresh:
            break

    return float(count - tp) / count
