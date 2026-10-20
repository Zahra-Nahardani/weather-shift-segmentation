# weather-shift-segmentation

**Does a segmentation model know when it is wrong?** Measuring how well the uncertainty of a
street-scene segmentation model flags its own errors when conditions change from clear weather
to fog, night, rain and snow.

Status: work in progress (started October 2026).

## Questions

1. How well does a pretrained segmentation model perform on clear-weather street scenes?
2. How much accuracy is lost in fog, night, rain and snow, and on which classes?
3. Is the model less confident where it is wrong, and does that still hold under adverse weather?

## Data

- [Cityscapes](https://www.cityscapes-dataset.com/): clear-weather street scenes, 19 classes.
- [ACDC](https://acdc.vision.ee.ethz.ch/): fog, night, rain and snow, same 19 classes.

Both datasets are for research use only and require registration. They are **not** included in
this repository; see [`data/README.md`](data/README.md) for where to place them.

## Project structure

```
app/            demo app (later)
configs/        experiment settings (YAML)
data/           datasets, not tracked by git
notebooks/      exploration, numbered in order
reports/        figures, results and the weekly log
src/weather_shift/  reusable code (data, model, uncertainty, metrics, plots)
tests/          unit tests
```

## Setup

```bash
uv sync
uv run pytest
```

In Google Colab: `pip install -e .` from the cloned repository.

## Results

Coming soon.

## License

Code: MIT. Datasets keep their own licences.
