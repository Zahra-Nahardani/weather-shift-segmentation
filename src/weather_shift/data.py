from pathlib import Path

import numpy as np
from PIL import Image


def get_pairs(data_root, dataset, condition=None):

    """Return a sorted list of (image_path, label_path) pairs."""
    
    data_root = Path(data_root)
    pairs = []
    if dataset == "cityscapes":
        images = sorted((data_root / "cityscapes" / "leftImg8bit" / "val").rglob("*.png"))
        for img in images:
            label = str(img).replace("leftImg8bit", "gtFine", 1)
            label = label.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
            pairs.append((img, Path(label)))
    elif dataset == "acdc":
        folder = data_root / "acdc" / "rgb_anon" / condition / "val"
        for img in sorted(folder.rglob("*.png")):
            label = str(img).replace("rgb_anon", "gt", 1)
            label = label.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
            pairs.append((img, Path(label)))
    return pairs

    
ID_TO_TRAINID = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
                 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
                 31: 16, 32: 17, 33: 18}
LUT = np.full(256, 255, dtype=np.uint8)
for label_id, train_id in ID_TO_TRAINID.items():
    LUT[label_id] = train_id

def load_label(label_path):
    """Load a label as trainIds (0-18, 255 = ignore)."""
    label = np.array(Image.open(label_path))
    if str(label_path).endswith("_labelIds.png"):
        label = LUT[label]
    return label

     




