# Dataset layout

The dataset is not included in this repository.

Place the data in the following structure before running the code:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    └── images/
```

Training and validation labels are expected as JSON annotation files matching the corresponding image filenames.
