import numpy as np
import pytest

pytest.importorskip("torch")

from weather_shift.metrics import image_miou  # noqa: E402


def test_image_miou_perfect_prediction():
    label = np.array([[0, 1], [1, 2]])
    assert image_miou(label, label.copy()) == 1.0


def test_image_miou_ignores_255():
    label = np.array([[0, 255], [1, 1]])
    pred = np.array([[0, 2], [1, 1]])  # the wrong pixel sits on "ignore"
    assert image_miou(label, pred) == 1.0


def test_image_miou_half_overlap():
    label = np.array([[0, 0, 1, 1]])
    pred = np.array([[0, 1, 1, 1]])
    # class 0: overlap 1 / union 2 = 0.5, class 1: overlap 2 / union 3
    assert np.isclose(image_miou(label, pred), (0.5 + 2 / 3) / 2)