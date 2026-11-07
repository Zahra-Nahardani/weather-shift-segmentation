# Weekly log

Short notes on what I tried, what I observed and what confused me. 

## Week 1

- Ran SegFormer-B0 (trained on Cityscapes, German cities) on a New York street photo.
- Mean confidence 0.90. Low confidence appears mainly along object edges.
- The right pavement is mostly mislabelled as road, yet confidence there is ~1.0:
  the model is confidently wrong, so its confidence did not flag this error.
- Open question: is this typical? Needs images with ground truth (Stage 2).

## Week 2

### Setup
- Model: SegFormer-B0 trained on Cityscapes, no changes.
- Data: Cityscapes val (500 images, clear) and ACDC val (fog 100, night 106, rain 100, snow 100).
- Every photo is squashed to 512x512 before the model sees it. The same setup is used for all conditions, so the comparison is fair.

### Result 1: mIoU per condition
| Condition | mIoU |
|---|---|
| Clear | 0.582 |
| Fog | 0.611 |
| Night | 0.218 |
| Rain | 0.385 |
| Snow | 0.396 |

Full per-class table: `reports/week2_iou_per_class.csv`.

- Clear weather (0.58) is lower than the paper (about 0.76), because the paper runs the model on the full-size photo in 1024x1024 pieces, while we squash the whole photo to 512x512. Thin objects like poles and signs suffer most.
- Night is by far the worst condition (0.22). The sky class drops from 0.91 to 0.005, and car drops from 0.87 to 0.26.
- Rain and snow lose about a third of the score. Sidewalk collapses (0.74 to 0.27-0.30).
- Fog scores slightly higher than clear. This should not be read as "fog is easy": ACDC is filmed in different cities with a different camera, many fog images are light fog, and rare classes (truck, bus, train) have only a few examples in 100 images, so their scores jump around and can't be trusted much.

### Result 2: what the model says instead (night confusion matrix)
IoU shows how much the model gets wrong. A confusion matrix shows what it says instead. Figure: `reports/figures/week2_confusion_night.png`.
- Night sky: 74% predicted as building and 25% as vegetation. The model almost never says "sky" at night.
- Sidewalk: 68% predicted as road. This is the same mistake as on my Week 1 photo, now a pattern across 106 images.
- Car: 59% correct and 13% predicted as building, because dark cars blend into the dark background.

### Result 3: failure gallery (3 worst night images, mIoU 0.07-0.10)
Figures: `reports/figures/week2_night_worst_*.png`.
- The biggest mistake is confident: the sky is predicted as building with confidence close to 1.0.
- The unsure (dark) areas in the confidence map are mostly edges between objects. So the model flags some mistakes, but not the biggest one.
- The road is correct and confident.
- ACDC labels mark areas that even the annotators couldn't see as "ignore" (black), so they aren't scored.

### Main takeaway
The model's own confidence does not warn us about its worst mistakes at night. This is the starting point for choosing the Week 3 question.

