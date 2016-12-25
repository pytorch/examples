# Faster R-CNN code example

```python
python main.py PATH_TO_DATASET
```

## Things to add/change/consider
* where to handle the image scaling. Need to scale the annotations, and also RPN filters the minimum size wrt the original image size, and not the scaled image
* properly supporting flipping
* best way to handle different parameters in RPN/FRCNN for train/eval modes
* uniformize Variables, they should be provided by the user and not processed by me
* should image scaling be handled in FasterRCNN class?
* general code cleanup, lots of torch/numpy mixture
