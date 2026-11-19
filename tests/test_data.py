import numpy as np
from PIL import Image

from weather_shift.data import LUT, get_pairs, load_label


def test_lut_maps_cityscapes_ids():
    assert LUT[7] == 0 # road
    assert LUT[26] == 13  # car
    assert LUT[0] == 255  # unlabeled -> ignore

def test_load_label_converts_cityscapes(tmp_path):
    path = tmp_path / "x_gtFine_labelIds.png"
    Image.fromarray(np.array([[7, 26, 0]], dtype=np.uint8)).save(path)
    assert load_label(path).tolist() == [[0, 13, 255]]

def test_load_label_keeps_acdc(tmp_path):
    path = tmp_path / "x_gt_labelTrainIds.png"
    Image.fromarray(np.array([[0, 13, 255]], dtype=np.uint8)).save(path)
    assert load_label(path).tolist() == [[0, 13, 255]]

def test_get_pairs_cityscapes(tmp_path):
    img_dir = tmp_path / "cityscapes" / "leftImg8bit" / "val" / "aachen"
    lbl_dir = tmp_path / "cityscapes" / "gtFine" / "val" / "aachen"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    (img_dir / "aachen_000000_000019_leftImg8bit.png").touch()
    (lbl_dir / "aachen_000000_000019_gtFine_labelIds.png").touch()
    pairs = get_pairs(tmp_path, "cityscapes")
    assert len(pairs) == 1
    assert pairs[0][1].exists()