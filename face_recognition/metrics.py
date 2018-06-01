from sklearn.model_selection import KFold
import numpy as np
import torch


def select_threshold(distances, matches, thresholds):
    best_threshold_true_predicts = 0
    best_threshold = 0
    for threshold in thresholds:
        true_predicts = torch.sum((
            distances < threshold
        ) == matches)

        if true_predicts > best_threshold_true_predicts:
            best_threshold_true_predicts = true_predicts
            best_threshold = threshold

    return best_threshold


def compute_roc(distances, matches, thresholds, fold_size=10):
    assert(len(distances) == len(matches))

    kf = KFold(n_splits=fold_size, shuffle=False)

    tpr = torch.zeros(fold_size, len(thresholds))
    fpr = torch.zeros(fold_size, len(thresholds))
    accuracy = torch.zeros(fold_size)
    best_thresholds = []

    for fold_index, (training_indices, val_indices) \
            in enumerate(kf.split(range(len(distances)))):

        training_distances = distances[training_indices]
        training_matches = matches[training_indices]

        # 1. find the best threshold for this fold using training set
        best_threshold_true_predicts = 0
        for threshold_index, threshold in enumerate(thresholds):
            true_predicts = torch.sum((
                training_distances < threshold
            ) == training_matches)

            if true_predicts > best_threshold_true_predicts:
                best_threshold = threshold
                best_threshold_true_predicts = true_predicts

        # 2. calculate tpr, fpr on validation set
        val_distances = distances[val_indices]
        val_matches = matches[val_indices]
        for threshold_index, threshold in enumerate(thresholds):
            predicts = val_distances < threshold

            tp = torch.sum(predicts & val_matches).item()
            fp = torch.sum(predicts & ~val_matches).item()
            tn = torch.sum(~predicts & ~val_matches).item()
            fn = torch.sum(~predicts & val_matches).item()

            tpr[fold_index][threshold_index] = float(tp) / (tp + fn)
            fpr[fold_index][threshold_index] = float(fp) / (fp + tn)

        best_thresholds.append(best_threshold)
        accuracy[fold_index] = best_threshold_true_predicts.item() / float(
            len(training_indices))

    # average fold
    tpr = torch.mean(tpr, dim=0).numpy()
    fpr = torch.mean(fpr, dim=0).numpy()
    accuracy = torch.mean(accuracy, dim=0).item()

    return tpr, fpr, accuracy, best_thresholds
