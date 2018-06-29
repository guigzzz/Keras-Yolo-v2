import numpy as np

def load_bn_weights(layer, data, index):
    bn_weight_shapes = [w.shape for w in layer.get_weights()]
    bias_shape, gamma_shape, mean_shape, std_shape = bn_weight_shapes

    num_bias_weights = np.prod(bias_shape)
    bias_weights = data[index : index + num_bias_weights].reshape(bias_shape)
    index += num_bias_weights

    num_gamma_weights = np.prod(gamma_shape)
    gamma_weights = data[index : index + num_gamma_weights].reshape(gamma_shape)
    index += num_gamma_weights

    num_mean_weights = np.prod(mean_shape)
    mean_weights = data[index : index + num_mean_weights].reshape(mean_shape)
    index += num_mean_weights

    num_std_weights = np.prod(std_shape)
    std_weights = data[index : index + num_std_weights].reshape(std_shape)
    index += num_std_weights
    
    bn_layer_weights = [gamma_weights, bias_weights, mean_weights, std_weights]
    layer.set_weights(bn_layer_weights)
    
    return index


def load_conv_weights(layer, data, index):
    conv_layer_weights = []

    if layer.use_bias:
        kernel_shape, bias_shape = [w.shape for w in layer.get_weights()]
    
        num_bias_weights = np.prod(bias_shape)
        bias_weights = data[index : index + num_bias_weights].reshape(bias_shape)
        index += num_bias_weights

        conv_layer_weights = [bias_weights]

    else:
        kernel_shape = layer.get_weights()[0].shape

    num_kernel_weights = np.prod(kernel_shape)
    kernel_weights = data[index : index + num_kernel_weights].reshape(kernel_shape)
    kernel_weights = np.reshape(
        kernel_weights,
        (kernel_shape[3], kernel_shape[2], 
        kernel_shape[0], kernel_shape[1]),
        order='C'
    )
    kernel_weights = np.transpose(kernel_weights, [2, 3, 1, 0])
    index += num_kernel_weights

    conv_layer_weights = [kernel_weights] + conv_layer_weights
    layer.set_weights(conv_layer_weights)
    
    return index

def getLayerInfo(layer):
    splits = layer.name.split('_')
    layer_index = int(splits[-1])
    layer_type = '_'.join(splits[:-1])

    return layer_type, layer_index

def load_weights(model, yolo_weight_file):

    with open(yolo_weight_file, 'rb') as f:
        weights_header = np.ndarray(
            shape=(4, ), dtype='int32', buffer=f.read(16))

    """
    *** the following works but feels like a massive hack ***
    The tiny yolo weight file only needs an offset of 4, the weights start after

    The normal yolo weight file seems to need an offset of 5. If an offset of 4 is used then there
    is a single left over weight at the end.

    The weight header for tiny yolo is 0, 1, 0, [some number]
    while the weight header for yolo is 0, 2, 0, [some number]
    hence we can get the correct offset for both by substracting one from the second number
    and adding that as an offset.
    """
    index = 4 + weights_header[1] - 1

    print('Loading weights from {}'.format(yolo_weight_file))
    data = np.fromfile(yolo_weight_file, np.float32)


    print('got {} values'.format(len(data)))

    layer_dict = {}
    for l in model.layers:
        ltype, ind = getLayerInfo(l)
        
        if ind in layer_dict:
            layer_dict[ind][ltype] = l
        
        else:
            layer_dict[ind] = {ltype: l}


    for i in sorted(layer_dict.keys()):
        if 'batch_normalization' in layer_dict[i]:
            # retrieve bn layer weights
            index = load_bn_weights(layer_dict[i]['batch_normalization'], data, index)
        
        # retrieve conv layer weights
        index = load_conv_weights(layer_dict[i]['conv2d'], data, index)

    # make sure that we have consumed all weights in data file
    err = 'Not all weights consumed, used: {}, expected: {}, diff: {}'\
            .format(index, len(data), len(data) - index)
    assert index == len(data), err