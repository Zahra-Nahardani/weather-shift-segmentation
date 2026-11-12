import torch
import torch.nn.functional as F
from PIL import Image


def predict(image, processor, model):
    """Return the predicted class map and the confidence map, both at image size."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    logits = F.interpolate(
        logits, size=(image.height, image.width), mode="bilinear", align_corners=False
    )
    probs = logits.softmax(dim=1)
    conf, pred = probs.max(dim=1)
    return pred[0].numpy(), conf[0].numpy()


def predict_probs(image, processor, model):
    """Class probabilities at image size, shape (19, H, W)."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    logits = F.interpolate(
        logits, size=(image.height, image.width), mode="bilinear", align_corners=False
    )
    return logits.softmax(dim=1)[0]


def predict_flip(image, processor, model):
    """Average the predictions for the photo and its mirror image."""
    probs = predict_probs(image, processor, model)
    mirrored = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    probs_m = predict_probs(mirrored, processor, model).flip(dims=[2])
    conf, pred = ((probs + probs_m) / 2).max(dim=0)
    return pred.numpy(), conf.numpy()