# Weekly log

Short notes on what I tried, what I observed and what confused me. 

## Week 1

- Ran SegFormer-B0 (trained on Cityscapes, German cities) on a New York street photo.
- Mean confidence 0.90. Low confidence appears mainly along object edges.
- The right pavement is mostly mislabelled as road, yet confidence there is ~1.0:
  the model is confidently wrong, so its confidence did not flag this error.
- Open question: is this typical? Needs images with ground truth (Stage 2).

## Week 2