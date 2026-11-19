"""Accuracy (IoU) and uncertainty (ECE, AUROC, confident mistakes) metrics."""

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from weather_shift.data import load_label
from weather_shift.model import predict


def evaluate(pairs, processor, model, num_classes=19):
    """Return the IoU of each class over all image/label pairs."""

    inter = np.zeros(num_classes)
    union = np.zeros(num_classes)
    for img_path, label_path in tqdm(pairs):
        image = Image.open(img_path).convert("RGB")
        label = load_label(label_path)
        pred, _ = predict(image, processor, model)
        valid = label != 255
        for c in range(num_classes):
            true_c = (label == c) & valid
            pred_c = (pred == c) & valid
            inter[c] += (true_c & pred_c).sum()
            union[c] += (true_c | pred_c).sum()
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)




def confusion(pairs, processor, model, num_classes=19):
    """ Count pixels for every class (correct class, predicted class) pairs"""

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for img_path, label_path in tqdm(pairs):
        image = Image.open(img_path).convert("RGB")
        label = load_label(label_path)
        pred, _ = predict(image, processor, model)
        valid = label != 255
        idx = num_classes * label[valid].astype(np.int64) + pred[valid]
        cm += np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)
    return cm




def image_miou(label, pred, num_classes=19):
    """mIoU of a single image (classes absent from both are skipped)."""
    valid = label != 255
    ious = []
    for c in range(num_classes):
        true_c = (label == c) & valid
        pred_c = (pred == c) & valid
        union = (true_c | pred_c).sum()
        if union > 0:
            ious.append((true_c & pred_c).sum() / union)
    return float(np.mean(ious))


def reliability(pairs, processor, model, n_bins=10, predict_fn=predict):
    """Per confidence box: pixel count, average confidence, accuracy."""
    bins = np.linspace(0, 1, n_bins + 1)
    count = np.zeros(n_bins)
    conf_sum = np.zeros(n_bins)
    correct_sum = np.zeros(n_bins)
    for img_path, label_path in tqdm(pairs):
        image = Image.open(img_path).convert("RGB")
        label = load_label(label_path)
        pred, conf = predict_fn(image, processor, model)
        valid = label != 255
        c = conf[valid]
        correct = pred[valid] == label[valid]
        for k, (lo, hi) in enumerate(zip(bins[:-1], bins[1:], strict=True)):
            in_bin = (c > lo) & (c <= hi)
            count[k] += in_bin.sum()
            conf_sum[k] += c[in_bin].sum()
            correct_sum[k] += correct[in_bin].sum()
    keep = count > 0
    return count[keep], conf_sum[keep] / count[keep], correct_sum[keep] / count[keep]


def error_detection(pairs, processor, model, pixels_per_image=20000, seed=0, predict_fn=predict):
    """AUROC of using (1 - confidence) to find wrong pixels."""
    rng = np.random.default_rng(seed)
    uncertainty, wrong = [], []
    for img_path, label_path in tqdm(pairs):
        image = Image.open(img_path).convert("RGB")
        label = load_label(label_path)
        pred, conf = predict_fn(image, processor, model)
        valid = label != 255
        c = conf[valid]
        w = pred[valid] != label[valid]
        idx = rng.choice(len(c), size=min(pixels_per_image, len(c)), replace=False)
        uncertainty.append(1 - c[idx])
        wrong.append(w[idx])
    return roc_auc_score(np.concatenate(wrong), np.concatenate(uncertainty))


def evaluate_uncertainty(pairs, processor, model, predict_fn=predict, n_bins=10,
                         pixels_per_image=20000, seed=0, num_classes=19, high=0.9):
    """One pass over the data: ECE, AUROC and confident-mistake numbers."""
    rng = np.random.default_rng(seed)
    n_correct = 0
    n_confident_correct = 0
    bins = np.linspace(0, 1, n_bins + 1)
    count = np.zeros(n_bins)
    conf_sum = np.zeros(n_bins)
    correct_sum = np.zeros(n_bins)
    wrong_per_class = np.zeros(num_classes)
    confident_wrong_per_class = np.zeros(num_classes)
    uncertainty, wrong_sample = [], []
    for img_path, label_path in tqdm(pairs):
        image = Image.open(img_path).convert("RGB")
        label = load_label(label_path)
        pred, conf = predict_fn(image, processor, model)
        valid = label != 255
        c = conf[valid]
        t = label[valid]
        correct = pred[valid] == t
        wrong = ~correct
        n_correct += correct.sum()
        n_confident_correct += (correct & (c > high)).sum()
        # 1. reliability boxes (for ECE)
        for k, (lo, hi) in enumerate(zip(bins[:-1], bins[1:], strict=True)):
            in_bin = (c > lo) & (c <= hi)
            count[k] += in_bin.sum()
            conf_sum[k] += c[in_bin].sum()
            correct_sum[k] += correct[in_bin].sum()
        # 2. mistakes and confident mistakes, per true class
        wrong_per_class += np.bincount(t[wrong], minlength=num_classes)
        confident_wrong_per_class += np.bincount(t[wrong & (c > high)], minlength=num_classes)
        # 3. random sample of pixels (for AUROC)
        idx = rng.choice(len(c), size=min(pixels_per_image, len(c)), replace=False)
        uncertainty.append(1 - c[idx])
        wrong_sample.append(wrong[idx])
    keep = count > 0
    avg_conf = conf_sum[keep] / count[keep]
    acc = correct_sum[keep] / count[keep]
    ece = np.sum(count[keep] / count.sum() * np.abs(acc - avg_conf))
    auroc = roc_auc_score(np.concatenate(wrong_sample), np.concatenate(uncertainty))
    summary = {
        "ECE": ece,
        "AUROC": auroc,
        "confident_mistakes": confident_wrong_per_class.sum() / wrong_per_class.sum(),
        "confident_correct": n_confident_correct / n_correct,
    }
    return summary, wrong_per_class, confident_wrong_per_class