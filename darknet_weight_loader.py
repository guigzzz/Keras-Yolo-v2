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

def load_conv_weights(layer, data, index, use_bias):
    conv_weight_shapes = [w.shape for w in layer.get_weights()]
    kernel_shape, bias_shape = conv_weight_shapes
    
    if use_bias:
        num_bias_weights = np.prod(bias_shape)
        bias_weights = data[index : index + num_bias_weights].reshape(bias_shape)
        index += num_bias_weights
    else:
        # yolov2 has biases set to zero
        # the batchnorm betas (offsets) play the same role
        num_bias_weights = np.prod(bias_shape)
        bias_weights = np.zeros(bias_shape) 

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

    conv_layer_weights = [kernel_weights, bias_weights]
    layer.set_weights(conv_layer_weights)
    
    return index

def load_weights(model, yolo_weight_file):
    print('Loading weights from {}'.format(yolo_weight_file))
    data = np.fromfile(yolo_weight_file, np.float32)
    
    # first four values of data aren't used
    index = 4 
    
    # 6 x (conv - bn - leakyrelu - maxpool) + 2 * (conv - bn - leakyrelu)
    layer_indices = list(range(0, 6*4, 4)) + list(range(6*4, 6*4+2*3, 3))
    
    for i in layer_indices:
        # retrieve bn layer weights
        index = load_bn_weights(model.layers[i+1], data, index)
        
        # retrieve conv layer weights
        index = load_conv_weights(model.layers[i], data, index, use_bias=False)
        
    # Final layer has no batchnorm, hence need to restore bias weights as well
    index = load_conv_weights(model.layers[-1], data, index, use_bias=True)

    # make sure that we have consumed all weights in data file
    err = 'Not all weights consumed, used: {}, expected: {}, diff: {}'\
            .format(index, len(data), len(data) - index)
    assert index == len(data), err