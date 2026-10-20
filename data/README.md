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

Folder names follow the official packages. Adjust `configs/default.yaml` if yours differ.
