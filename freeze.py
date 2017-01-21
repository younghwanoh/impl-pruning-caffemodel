#!/usr/bin/env python

import caffe
import sys
import numpy as np
sys.dont_write_bytecode = True

prototxt = "models/train_val.prototxt"
trained_model_file = "models/bvlc_alexnet.caffemodel" # original model
# trained_model_file = "models/bvlc_alexnet_itq_exp4.caffemodel" # itq-trained model
freezed_model_file = "models/output.caffemodel" # quantiezd model

prototxt = "./examples/cifar10/cifar10_itq_exp4.prototxt"
trained_model_file = "./examples/cifar10/cifar10_exp4_iter_6600.caffemodel.h5" # original model
freezed_model_file = "./examples/cifar10/output.caffemodel" # quantiezd model
 
caffe.set_mode_gpu()
net = caffe.Net(prototxt, trained_model_file, caffe.TEST)
# layers = filter(lambda x:'conv' in x or 'fc' in x or 'ip' in x, net.params.keys())
layers = filter(lambda x:'conv' in x, net.params.keys())

# Quantize with full-precision format
def quantize2exp_full(weight_arr, bitwidth=4, **kwargs):
    shape = weight_arr.shape
    flatten_arr = weight_arr.reshape(weight_arr.size)

    min_exp = -pow(2, bitwidth-1)
    max_exp =  pow(2, bitwidth-1)-1
    sign = np.sign(flatten_arr)
    exp_arr = np.log2(np.abs(flatten_arr))
    for idx, elem in enumerate(exp_arr):
        exp_arr[idx] = max(min(round(elem), max_exp), min_exp)
    recnstr_arr = sign * np.exp2(exp_arr)
    recnstr_arr = recnstr_arr.reshape(shape)
    return recnstr_arr

# Quantize with realistic format
def quantize2exp(weight_arr, bitwidth=4, **kwargs):
    shape = weight_arr.shape
    flatten_arr = weight_arr.reshape(weight_arr.size)

    min_exp = -pow(2, bitwidth-1)
    max_exp =  pow(2, bitwidth-1)-1
    sign = np.sign(flatten_arr)
    exp_arr = np.log2(np.abs(flatten_arr))
    for idx, elem in enumerate(exp_arr):
        exp_arr[idx] = max(min(round(elem), max_exp), min_exp)
    # TODO: sign bit should be concatenated to exp_arr
    exp_arr = exp_arr.reshape(shape)
    return exp_arr


# Quantizaiton all conv, fc layers
for idx, layer in enumerate(layers):
    print "layer name: ", layer

    weight_data = net.params[layer][0].data
    weight_data = quantize2exp_full(weight_data, bitwidth=4)
    np.copyto(net.params[layer][0].data, weight_data)

    print "Quantization done: %s" % layer
    print ""

net.save(freezed_model_file)
