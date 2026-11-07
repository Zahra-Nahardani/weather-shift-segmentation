import numpy as np
from PIL import Image
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