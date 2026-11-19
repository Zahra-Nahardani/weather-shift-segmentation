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

## Week 4: main experiment, can the model tell us when it's wrong?

Notebook: `notebooks/04_main_experiment.ipynb`

### Question
When the weather gets bad, can the model tell us when it's wrong?
1. Which mistakes come without a lowered confidence?
2. Do plain confidence, the flip test or MC dropout catch them?

### Methods compared
- **Plain:** the model's normal softmax confidence (one run).
- **Flip (test-time augmentation):** run on the photo and its mirror image, average the two answers.
- **MC dropout:** keep the dropout layers switched on, run 8 times, average the answers. Only the Dropout and DropPath layers are switched on; BatchNorm stays in normal mode. SegFormer-B0 has little dropout (one Dropout layer with p = 0.1 and 7 DropPath layers with p = 0.014-0.1), so the runs differ only slightly.

### Measures
- **ECE:** gap between confidence and accuracy (lower = more honest).
- **AUROC:** does low confidence point to the wrong pixels? (higher = better, 0.5 = coin flip). 20,000 random pixels per image, seed 0.
- **Confident mistakes:** share of wrong pixels with confidence > 0.9 (lower = better).
- **Confident correct:** share of correct pixels with confidence > 0.9 (to check whether a method only lowers confidence everywhere).
- "Confident" means confidence > 0.9 (parameter `high` in `evaluate_uncertainty`).
- Sanity check: plain on night gives ECE 0.277 and AUROC 0.742, the same as Week 3.

### Result 1: all methods, all conditions
File: `reports/week4_results.csv`

| Condition | Method | ECE | AUROC | Confident mistakes | Confident correct |
|---|---|---|---|---|---|
| Clear | plain | 0.014 | 0.926 | 0.175 | 0.874 |
| Clear | flip | 0.009 | 0.929 | 0.153 | 0.867 |
| Clear | MC | 0.004 | 0.928 | 0.110 | 0.834 |
| Fog | plain | 0.025 | 0.903 | 0.273 | 0.884 |
| Fog | flip | 0.020 | 0.907 | 0.242 | 0.875 |
| Fog | MC | 0.008 | 0.905 | 0.174 | 0.830 |
| Rain | plain | 0.052 | 0.883 | 0.250 | 0.846 |
| Rain | flip | 0.041 | 0.889 | 0.206 | 0.830 |
| Rain | MC | 0.021 | 0.898 | 0.120 | 0.776 |
| Snow | plain | 0.080 | 0.826 | 0.344 | 0.833 |
| Snow | flip | 0.067 | 0.833 | 0.302 | 0.813 |
| Snow | MC | 0.045 | 0.832 | 0.219 | 0.747 |
| Night | plain | 0.277 | 0.742 | 0.473 | 0.785 |
| Night | flip | 0.256 | 0.739 | 0.425 | 0.747 |
| Night | MC | 0.216 | 0.733 | 0.273 | 0.618 |

- **MC dropout makes the confidence more honest.** It has the lowest ECE in every condition (night 0.277 to 0.216). Flip helps a little.
- **No method gets better at finding the mistakes.** AUROC stays within about 0.015 across methods. At night, flip and MC are slightly worse than plain.
- **Fewer confident mistakes, mostly because everything gets less confident.** At night, MC cuts confident mistakes from 47% to 27%, but confident correct pixels also fall from 79% to 62%. MC lowers confidence across the board instead of picking out the mistakes.
- **A hard core stays hidden.** Even with MC dropout, about 1 in 4 night mistakes is still made with more than 90% confidence.

### Result 2: which mistakes stay confident? (night)
File: `reports/week4_confident_mistakes_per_class.csv` (share of each class's mistakes made with confidence > 0.9)

| Class | Plain | Flip | MC |
|---|---|---|---|
| Sky | 0.58 | 0.54 | 0.37 |
| Sidewalk | 0.58 | 0.54 | 0.32 |
| Fence | 0.43 | 0.37 | 0.23 |
| Vegetation | 0.34 | 0.28 | 0.17 |
| Wall | 0.29 | 0.22 | 0.11 |
| Pole | 0.27 | 0.21 | 0.18 |
| Traffic sign | 0.26 | 0.24 | 0.15 |
| Building | 0.24 | 0.16 | 0.11 |

- The top two are the same two confusions found in Week 2: **sky to building** and **sidewalk to road**.
- These are **large, smooth areas** where the model swaps one plausible class for its neighbour. The model doesn't hesitate at an edge; it confidently misreads the whole region. Asking again (flip, dropout) gives the same wrong answer.
- **Thin objects** (pole, traffic sign) have fewer confident mistakes: their errors sit mostly at edges, where the confidence already drops.
- **Sidewalk to road is a safety problem:** pedestrians walk on sidewalks, and a confident "road" gives no warning.

### Answers
1. **Which mistakes come without a lowered confidence?** Whole-region swaps between similar-looking classes, mainly sky to building and sidewalk to road at night.
2. **Do the methods catch them?** Only partly. MC dropout lowers their confidence somewhat (and makes the confidence more honest overall), but about a third of these mistakes stay above 90% confidence, and no method gets better at pointing to where the mistakes are.

### Open points for Week 5
- Check that the conclusions don't depend on the 0.9 threshold (repeat with 0.8 and 0.95).
- Turn the results into 3-4 clear figures.

## Week 5: robustness check and figures

Notebook: `notebooks/05_analysis_figures.ipynb`

### Robustness check: does the 0.9 threshold matter? (night)
File: `reports/week5_threshold_check.csv`

| Threshold | Confident mistakes (plain -> MC) | Confident correct (plain -> MC) |
|---|---|---|
| 0.80 | 0.60 -> 0.44 | 0.86 -> 0.76 |
| 0.90 | 0.47 -> 0.28 | 0.79 -> 0.63 |
| 0.95 | 0.37 -> 0.19 | 0.72 -> 0.51 |

- At every threshold, MC dropout lowers confident mistakes **and** confident correct pixels, and a hard core of confident mistakes remains (about 1 in 5 even at 0.95).
- **The conclusions hold for confidence thresholds from 0.8 to 0.95.**
- MC dropout is random, so its results vary by about +-0.01 between runs. From now on the seed is fixed with `torch.manual_seed(0)`.

### Figures
1. `fig1_miou_per_condition.png`: segmentation quality drops under adverse weather (clear 0.58, night 0.22). "Clear" comes from Cityscapes, the others from ACDC.
2. `fig2_reliability.png`: confidence is honest in clear weather (ECE 0.01) but overconfident at night (0.28). MC dropout moves the night curve towards honest (0.21) but stays far from it.
3. `fig3_methods_ece_auroc.png`: MC dropout lowers ECE in every condition, but AUROC is almost the same for all methods. Only the weather changes it.
4. `fig4_hard_core_example.png`: a night photo where the sky (called building) and the sidewalk (called road) are wrong with more than 90% confidence, even with MC dropout.

### Findings
A segmentation model trained on clear-weather city scenes (SegFormer-B0, Cityscapes) loses much of its accuracy under adverse weather, dropping from 0.58 mIoU in clear weather to 0.22 at night. Its confidence also stops being trustworthy: it is well calibrated in clear weather (ECE 0.01) but strongly overconfident at night (ECE 0.28), and low confidence becomes a weaker pointer to mistakes (AUROC 0.93 to 0.74). Two common uncertainty methods, test-time flipping and MC dropout, make the confidence more honest (MC dropout lowers night ECE to 0.22), but neither gets better at finding where the model is wrong: AUROC stays almost unchanged in every condition. MC dropout mainly lowers confidence everywhere, on correct and wrong pixels alike, instead of singling out the mistakes. A hard core of errors stays hidden: at night, about a quarter of all mistakes are still made with more than 90% confidence, mostly whole regions swapped for a similar-looking class, such as sky called building and sidewalk called road. Because these mistakes are consistent, asking the same model again does not reveal them, which suggests that an independent source of information is needed to catch them.