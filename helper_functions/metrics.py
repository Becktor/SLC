import torch
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
import sklearn.metrics as sk

recall_level_default = 0.95


def entropy(x) -> torch.Tensor:
    return torch.sum(-x * torch.log(x.clip(min=1e-6)), dim=-1)


def g_softmax(x, norm):
    g = 1
    cdf_v = norm.cdf(x)
    top = torch.exp(x + g*cdf_v)
    bot = torch.sum(torch.exp(x+g*cdf_v), 1)
    return top / bot


def expected_entropy(mc_preds) -> torch.Tensor:
    """
    Take a tensor mc_preds [n_mc x batch_size x n_classes] and return the
    mean entropy of the predictive distribution across the MC samples.
    """
    return entropy(mc_preds).mean(0)  # batch_size


def predictive_entropy(mc_preds) -> torch.Tensor:
    """
    Take a tensor mc_preds [n_mc x batch_size x n_classes] and return the
    entropy of the mean predictive distribution across the MC samples.
    """
    return entropy(mc_preds.mean(0))


#MI
def BALD(mc_preds: torch.Tensor) -> torch.Tensor:
    """
    Calculate the BALD (Bayesian Active Learning by Disagreement) of a model;
    the difference between the mean of the entropy and the entropy of the mean
    of the predicted distribution on the n_mc x batch_size x n_classes tensor
    mc_preds. In the paper, this is referred to simply as the MI.
    """
    BALD = predictive_entropy(mc_preds) - expected_entropy(mc_preds)
    return BALD


def variance_score(p):
    return torch.pow(p-p.mean(0), 2).mean()


def expected_kl(p):
    return (p.mean(0) * torch.log(p.mean(0) / p)).sum(1).mean()


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def get_measures(_pos, _neg, recall_level=recall_level_default, plot_fpr_at_recall=False):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)

    if plot_fpr_at_recall:
        display = PrecisionRecallDisplay.from_predictions(labels, examples, name="VOS")
        _ = display.ax_.set_title("2-class Precision-Recall curve")
        plt.savefig(f'plots/pr_curve.png')
        plt.clf()
        fprs = []
        for recall_level in np.arange(0.05, 1, 0.05):
            fprs.append(fpr_and_fdr_at_recall(labels, examples, recall_level=recall_level))
        return auroc, aupr, fprs

    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)
    return auroc, aupr, fpr


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def show_performance_fpr(pos, neg, method_name='Ours', recall_level=recall_level_default):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    '''

    auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level, plot_fpr_at_recall=True)
    plt.plot(np.arange(0.05, 1, 0.05), fpr, label=method_name)
    plt.savefig(f'plots/fprs.png')
    print('\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr[-1]))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
    # print('FDR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fdr))
    return fpr[-1]


def fpr_tpr(in_scores, out_scores, tpr_level=0.95):
    '''
    calculate the false positive error rate when tpr is 95%
    '''
    in_scores = in_scores
    out_scores = out_scores

    thresh = np.quantile(out_scores, tpr_level)
    fpr = np.mean(in_scores <= thresh)

    return fpr


def fpr_at_tpr(pos, neg, tpr_level=0.95):
    '''
    calculate the false positive error rate when tpr is 95%
    '''

    thresh = np.quantile(pos, tpr_level)
    fpr = np.mean(neg <= thresh)

    return fpr