# Apply simple pruning on Caffemodel

1) Get caffemodel from [model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)

2) open proto.py(or pycaffe.py) and specify the name of target layer and its pruning ratio
```python
# AlexNet example
ratio = {"fc6":0.91, "fc7":0.91, "fc8":0.75}
```

3) Example commandline usage.
```bash
# Using protobuf api
proto.py bvlc_alexnet.caffemodel output_pruned.caffemodel

# Using pycaffe api
pycaffe.py deploy.prototxt bvlc_alexnet.caffemodel output_pruned.caffemodel
```

**To run pycaffe.py, pycaffe should be built in advance.**
