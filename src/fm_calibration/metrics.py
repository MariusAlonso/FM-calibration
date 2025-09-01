import numpy as np
import relplot as rp
import torch
from torch.nn import CrossEntropyLoss


def safeNLL(p, eps=1e-7):
    p += eps
    return p / p.sum(axis=-1)[:, None]


def mc_brier_score(y_pred, y_true, reduce=True):
    """
    Compute the Brier score for multiclass classification.

    Parameters:
    - y_true: array-like of shape (n_samples,) or (n_samples, n_classes),
      The true labels. Each label is an integer representing the class.
    - y_pred: array-like of shape (n_samples, n_classes),
      The predicted probabilities for each class.

    Returns:
    - brier_score: The average Brier score over all samples.
    """
    # Convert y_true to one-hot encoding
    n_classes = y_pred.shape[1]
    if len(y_true.shape) == 1:
        y_true_onehot = np.eye(n_classes)[y_true]
    else:
        y_true_onehot = y_true

    # Compute the Brier score for each sample and average over all samples
    brier_scores = np.mean((y_pred - y_true_onehot) ** 2, axis=1)
    return np.mean(brier_scores) if reduce else brier_scores


def safeCE(probas, labels, reduce=True):
    probas = safeNLL(probas)
    if reduce:
        return CrossEntropyLoss()(
            torch.log(torch.tensor(probas)), torch.tensor(labels).long()
        ).item()
    else:
        return CrossEntropyLoss(reduction="none")(
            torch.log(torch.tensor(probas)), torch.tensor(labels).long()
        ).numpy()


def ece_bins(confidences, trues, bin_boundaries):

    num_bins = len(bin_boundaries) - 1
    bin_counts = np.zeros(num_bins)
    bin_accs = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)

    if isinstance(confidences, torch.Tensor):
        confidences = confidences.cpu().numpy()
    if isinstance(trues, torch.Tensor):
        trues = trues.cpu().numpy().astype(np.float32)

    if len(confidences) != len(trues):
        print(len(confidences), len(trues))
        raise ValueError("Confidences and trues must have the same length")

    for i in range(num_bins):
        bin_mask = (confidences >= bin_boundaries[i]) & (
            confidences < bin_boundaries[i + 1]
        )
        bin_counts[i] = np.sum(bin_mask)
        if bin_counts[i] > 0:
            bin_accs[i] = trues[bin_mask].mean()
            bin_confidences[i] = confidences[bin_mask].mean()

    return bin_counts, bin_accs, bin_confidences


def ece_series(s):
    return np.abs((s["proba"] - s["true"]) * s["count"]).sum() / s["count"].sum()


def ece_conf(confidences, trues, num_bins=10, adaptative=False):
    if adaptative:
        bin_boundaries = np.quantile(confidences, np.linspace(0, 1, num_bins + 1))
    else:
        bin_boundaries = np.linspace(0, 1, num_bins + 1)

    bin_counts, bin_accs, bin_confidences = ece_bins(confidences, trues, bin_boundaries)
    ece_score = np.sum(bin_counts * np.abs(bin_accs - bin_confidences)) / len(trues)

    return ece_score


def ece(confidences, trues, num_bins=10):
    trues = confidences.argmax(axis=1) == trues
    confidences = np.max(confidences, axis=1)
    return ece_conf(confidences, trues, num_bins=num_bins)


def ace(confidences, trues, num_bins=10):
    labels = np.unique(trues)
    ece_scores = [
        ece_conf(
            confidences[:, i],
            1.0 * (trues == label),
            num_bins=num_bins,
            adaptative=True,
        )
        for i, label in enumerate(labels)
    ]
    return np.mean(ece_scores)


def smece(confidences, trues):
    return rp.smECE(confidences.max(axis=1), confidences.argmax(axis=1) == trues)
