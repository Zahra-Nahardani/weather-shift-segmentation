# Weekly log

Short notes on what I tried, what I observed and what confused me. 

## Week 1

- Ran SegFormer-B0 (trained on Cityscapes, German cities) on a New York street photo.
- Mean confidence 0.90. Low confidence appears mainly along object edges.
- The right pavement is mostly mislabelled as road, yet confidence there is ~1.0:
  the model is confidently wrong, so its confidence did not flag this error.
- Open question: is this typical? Needs images with ground truth (Stage 2).

## Week 2

Notebook: `notebooks/02_clear_vs_adverse.ipynb`

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

## Week 3: does the model's confidence warn about its mistakes?

Notebook: `notebooks/03_confidence_check.ipynb`

### Tools
- **Reliability check / ECE:** group pixels by confidence and compare the average confidence with the real accuracy. ECE is the average gap (0 = perfectly honest).
- **AUROC (error detection):** the chance that a random wrong pixel has lower confidence than a random correct pixel (1.0 = perfect, 0.5 = coin flip). 20,000 random pixels per image, seed 0.

### Result 1: one night image
- When the model is ~98% confident, it is right only about 71% of the time. At 75% confidence it is right only 22% of the time.
- Accuracy is still highest in the top boxes and lowest in the bottom boxes, so confidence still sorts pixels from worse to better, but its numbers are far too high.
- One image is not enough to draw conclusions, so the next step looks at all images.

### Result 2: reliability diagram, clear vs night (all images)
Figure: `reports/figures/week3_reliability_clear_night.png`
- Clear (ECE 0.01): the curve almost sits on the dashed line. In normal weather the confidence is honest: when the model says 80%, it is right about 80% of the time.
- Night (ECE 0.28): the curve drops far below the line. The same model, with the same confidence numbers, becomes overconfident at night.
- **The confidence is trustworthy in the conditions the model was trained on, and stops being trustworthy under weather shift.**

### Result 3: confidence quality per condition
Table: `reports/week3_confidence_quality.csv`

| Condition | ECE | AUROC |
|---|---|---|
| Clear | 0.014 | 0.926 |
| Fog | 0.025 | 0.903 |
| Rain | 0.052 | 0.883 |
| Snow | 0.080 | 0.826 |
| Night | 0.277 | 0.742 |

- Clear and fog: the confidence works well. The numbers are honest, and low confidence points at most mistakes.
- Rain and snow: it gets worse.
- Night: it breaks. The warning still helps a little (better than a 0.5 coin flip), but it misses many mistakes. This fits the Week 2 gallery: the night sky is wrong and confident.
- **The worse the weather shift, the worse the model is, and the worse its confidence warns about it.**

### Result 4: a first try at a better warning (flip test, night only)
| Method | ECE | AUROC |
|---|---|---|
| Plain confidence | 0.277 | 0.742 |
| Flip (photo + mirror, averaged) | 0.256 | 0.739 |

- The flip test barely changed anything. ECE improved a little and AUROC did not improve.
- The model makes the same mistake on the mirrored photo, so asking twice the same way does not help.

### Summary so far
1. The model gets much worse under weather shift (mIoU: clear 0.58, night 0.22).
2. Its confidence stops being honest (ECE: clear 0.014, night 0.277) and stops pointing at its mistakes (AUROC: clear 0.93, night 0.74).
3. Giving the model a second look with a flipped photo does not help, so the worst mistakes are **confident and consistent**.

### Focused question (next step)
When the weather gets bad, can the model tell us when it’s wrong? Which mistakes does the model make without lowering its confidence? Do plain confidence, the flip test or MC dropout catch those mistakes?

**Hypothesis:** methods that ask the same model again will help a little in rain and snow, but will miss most confident night mistakes such as sky -> building.