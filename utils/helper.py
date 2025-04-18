import os
import json
import torch
import numpy as np
from sklearn import metrics
from operator import itemgetter
from typing import List, Optional, Tuple


def tune_threshold_from_score(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fa: float,
    target_fr: Optional[float] = None,
) -> Tuple[List[float], float, float, float]:

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        idx = np.nanargmin(np.absolute((tfa - fpr)))  # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100

    return tunedThreshold, eer, fpr, fnr


def compute_error_rates(
    scores: np.ndarray, labels: np.ndarray
) -> Tuple[List[float], List[float], List[float]]:
    """
    Creates a list of false-negative rates, a list of false-positive rates
    and a list of decision thresholds that give those error-rates.

    Sort the scores from smallest to largest, and also get the corresponding
    indexes of the sorted scores.  We will treat the sorted scores as the
    thresholds at which the the error-rates are evaluated.
    """
    sorted_indexes, thresholds = zip(
        *sorted(
            [(index, threshold) for index, threshold in enumerate(scores)],
            key=itemgetter(1),
        )
    )
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fprs[i - 1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds


def compute_min_dcf(
    fnrs: List[float],
    fprs: List[float],
    thresholds: List[float],
    p_target: float,
    c_miss: float,
    c_fa: float,
) -> Tuple[float, float]:
    """
    Computes the minimum of the detection cost function.  The comments refer to
    equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
    """
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)
) -> List[float]:

    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def load_pretrained_path(path: str):
    print("Model %s loaded from previous state!" % path)
    frontend_config = None
    frontend_config_path = os.path.join(path, "frontend_config.json")
    if os.path.exists(frontend_config_path):
        frontend_config = json.load(open(frontend_config_path))

    backend_config = None
    backend_config_path = os.path.join(path, "backend_config.json")
    if os.path.exists(backend_config_path):
        backend_config = json.load(open(backend_config_path))

    return frontend_config, backend_config


def save_config(model: torch.nn.Module, path: str) -> None:
    # Save frontend config (if needed)
    if model.frontend_version != "mel_spectrogram":
        frontend_config = model.feat_extractor.config
        frontend_config_path = os.path.join(path, "frontend_config.json")
        json.dump(frontend_config, open(frontend_config_path, "w"))

    # Save backend config
    backend_config = model.speaker_encoder.config
    backend_config_path = os.path.join(path, "backend_config.json")
    json.dump(backend_config, open(backend_config_path, "w"))
