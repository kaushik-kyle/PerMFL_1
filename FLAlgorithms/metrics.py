"""Pooled classification metrics.

Accuracy is uninformative on imbalanced data: CICIDS2017 is roughly 74%
BENIGN after subsampling, so a constant predictor scores 0.74. Macro F1 is
reported alongside it.

Macro F1 is averaged over the FULL label set, including classes absent from a
given test slice, which contribute 0. Averaging only over classes that happen
to appear inflates personalised methods, since each device holds a subset of
classes and would otherwise be graded only on those. This is the
`labels=list(range(num_classes)), zero_division=0` behaviour of sklearn's
precision_recall_fscore_support, implemented directly to avoid adding a
dependency the upstream project does not have.
"""
import numpy as np
import torch


def num_classes_of(model):
    """Output width of the final Linear layer."""
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            last = m
    if last is None:
        raise ValueError("model has no Linear layer to read class count from")
    return last.out_features


def macro_f1(cm):
    """Macro F1 from a confusion matrix indexed [true, predicted]."""
    cm = np.asarray(cm, dtype=np.float64)
    f1 = np.zeros(len(cm))
    for c in range(len(cm)):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(f1.mean()), f1


def per_class_support(cm):
    return np.asarray(cm).sum(axis=1).astype(np.int64)


def full_report(cm):
    """Per-class precision, recall, F1 and false-positive rate from a
    confusion matrix indexed [true, predicted].

    Recall is the detection rate an IDS operator tunes to; FPR is the
    alert-fatigue constraint they tune against. Both come free from the
    confusion matrix macro_f1 already builds. Neither uses the TN cell for
    recall, which is why recall and F1 stay honest under class imbalance
    while accuracy does not: on our CICIDS split a detector that fires on
    nothing scores 0.955 to 0.992 per-class accuracy and 0.00 recall.
    """
    cm = np.asarray(cm, dtype=np.float64)
    C = len(cm); tot = cm.sum()
    prec = np.zeros(C); rec = np.zeros(C); f1 = np.zeros(C); fpr = np.zeros(C)
    for c in range(C):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = tot - tp - fp - fn
        prec[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = 2 * prec[c] * rec[c] / (prec[c] + rec[c]) if (prec[c] + rec[c]) > 0 else 0.0
        fpr[c] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return dict(macro_precision=float(prec.mean()), macro_recall=float(rec.mean()),
                macro_f1=float(f1.mean()), macro_fpr=float(fpr.mean()),
                per_class_recall=rec, per_class_f1=f1, per_class_fpr=fpr)
