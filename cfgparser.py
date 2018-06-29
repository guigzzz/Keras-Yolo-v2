import configparser
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, \
                    add, Concatenate, MaxPooling2D, Input
from keras import regularizers, initializers, Model

def parseYoloCFG(path):
    with open(path, 'r') as f:
        lines = f.read().split('\n')
        lines = [l.strip() for l in lines]
        
    all_sections = []
    cursection = {}
    curtag = None

    for l in lines:
        if l.startswith('['):
            all_sections.append((curtag, cursection))
            curtag = l[1:-1]
            cursection = {}

        elif not l.startswith('#') and l != '':
            key, value = [s.strip() for s in l.split('=')]
            if ',' in value:
                if '.' in value:
                    value = [float(v) for v in value.split(',')]
                else:
                    value = [int(v) for v in value.split(',')]
            elif '.' in value:
                value = float(value)
            else:
                try:
                    value = int(value)
                except:
                    pass

            cursection[key] = value
    
    all_sections.append((curtag, cursection))

    return all_sections[1:]


def interpretConv(layers, vd):
    assert 'filters' in vd and 'size' in vd and 'pad' in vd and 'activation' in vd
    
    filters = vd['filters']
    size = vd['size']
    stride = vd['stride']
    pad = 'valid' if vd['pad'] == 0 else 'same'
    
    batch_normalize = bool(vd.get('batch_normalize', 0))
        
    ops = [Conv2D(filters, size, strides=stride, padding=pad, 
                kernel_regularizer=regularizers.l2(0.0005),
                kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                use_bias=not batch_normalize
                )]
    
    if batch_normalize:
        ops.append(BatchNormalization())
        
    actname = vd['activation']
    if actname == 'leaky':
        ops.append(LeakyReLU(alpha=0.1))
    elif actname == 'linear':
        pass
    else:
        raise ValueError('interpretConv - encountered unknown activation function: {}'.format(actname))
    
    tensor = layers[-1]
    for op in ops:
        tensor = op(tensor)
        
    return tensor, ops

def interpretShortcut(layers, vd):
    prev = vd['from']
    tensor = add([
        layers[-1],
        layers[prev]
    ])
    
    actname = vd['activation']
    if actname == 'linear':
        pass
    else:
        raise ValueError('interpretShortcut - encountered unknown activation function: {}'.format(actname))
        
    return tensor

def interpretUpsample(layers, vd):
    from yolo_v3 import Upsample

    stride = vd['stride']
    return Upsample(stride)(layers[-1])

def interpretRoute(layers, vd):
    v = vd['layers']
    if isinstance(v, list):
        v = [r if r < 0 else r+1 for r in v]
        l = [layers[r] for r in v]
        return Concatenate()(l)
    
    else:
        return layers[v-1]
    
def interpretMaxpool(layers, vd):
    size = vd['size']
    stride = vd['stride']
    return MaxPooling2D(size, stride)(layers[-1])


from darknet_weight_loader import load_bn_weights, load_conv_weights
def load_weights(ops, data, index):
    if len(ops) < 3: # no bn, no activation
        return load_conv_weights(ops[0], data, index)
    elif len(ops) == 3: # normal
        index = load_bn_weights(ops[1], data, index)
        return load_conv_weights(ops[0], data, index)
    else:
        raise ValueError('got unexpected list of operations: {}'.format(ops))
        
import numpy as np
def fetch_weights_from_file(yolo_weight_file):
    with open(yolo_weight_file, 'rb') as f:
        weights_header = np.ndarray(
            shape=(4, ), dtype='int32', buffer=f.read(16))

    index = 4 + weights_header[1] - 1
    data = np.fromfile(yolo_weight_file, np.float32)[index:]
    
    return data


def buildModel(input_tensor, sections, weight_file=None):
    
    if weight_file is not None:
        data = fetch_weights_from_file(weight_file)
        index = 0
    
    layers = [input_tensor]
    outputs = []
    
    for i, (name, vd) in enumerate(sections):
        if name == 'convolutional':
            tensor, ops = interpretConv(layers, vd)
            if weight_file is not None:
                index = load_weights(ops, data, index)

        elif name == 'shortcut':
            tensor = interpretShortcut(layers, vd)
        
        elif name == 'yolo':
            outputs.append(layers[-1])
            layers.append(())
        
        elif name == 'route':
            tensor = interpretRoute(layers, vd)
        
        elif name == 'upsample':
            tensor = interpretUpsample(layers, vd)
            
        elif name == 'maxpool':
            tensor = interpretMaxpool(layers, vd)
            
        elif name == 'region':
            outputs.append(layers[-1])
            layers.append(())
        
        else:
            raise ValueError('Unrecognized layer name {}, with values: {}'.format(name, vd))
            
        print('\rProcessed {}/{}'.format(i, len(sections)), end='')
        layers.append(tensor)
        
    print()
    if weight_file is not None:
        print('Processed {}/{} values from weight file {}'.format(index, len(data), weight_file))
            
    return Model(input_tensor, outputs)


def load_from_darknet_cfg(filepath, weight_file=None):
    sections = parseYoloCFG(filepath)

    net_info = sections[0][1]
    t = Input(shape=(net_info['height'], net_info['width'], net_info['channels']))
    
    return buildModel(t, sections[1:], weight_file=weight_file)
