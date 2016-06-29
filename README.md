# Apply simple pruning on Caffemodel

1) Get caffemodel from [model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)

2) open caffemode.py and specify the name of target layer and its pruning ratio
```python
# AlexNet example
ratio = {"fc6":0.91, "fc7":0.91, "fc8":0.75}
```

3) Example commandline usage.
```
caffemodel.py bvlc_alexnet.caffemodel output_pruned.caffemodel
```
