import torch
import torch.nn.functional as F

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

   