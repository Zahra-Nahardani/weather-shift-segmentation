# Data

Datasets are not tracked by git. Download them after registering, and place them like this:

```
data/
  cityscapes/
    leftImg8bit/{train,val,test}/...
    gtFine/{train,val,test}/...
  acdc/
    rgb_anon/{fog,night,rain,snow}/...
    gt/{fog,night,rain,snow}/...
```

Only the `val` splits are used. Folder names follow the official packages: for Cityscapes,
`leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`; for ACDC, the anonymized RGB
images and the semantic segmentation labels. If you store the data somewhere else, change the
data folder path at the top of each notebook.