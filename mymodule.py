import numpy
import struct
import math

def makeImage(path):
    fI = open('{}'.format(path), 'rb')
    nxI = fI.read(4)
    nxI = struct.unpack("i", nxI)[0]
    nyI = fI.read(4)
    nyI = struct.unpack("i", nyI)[0]
    img_target = []
    for _ in range(nxI*nyI):
        data = fI.read(8)
        data = struct.unpack("d", data)[0]
        img_target.append(data)
    fI.close()
    result = numpy.array(img_target).reshape(nyI, nxI)
    return result

def makeDataset(target, low, width_block, height_block, width_low, height_low, numberOfTargets):
    height = len(target)
    width = int(target.size/height)
    width_target, height_target = width_block, height_block - height_low
    size_block = width_block * height_block
    size_low = width_low * height_low
    size_target = width_target * height_target
    halfOfWidthBlock = int(math.ceil(width_block / 2))
    arr_data = []
    for y in range(height - height_block + 1):
        for x in range(0, width - width_block + 1, numberOfTargets):
            target_block = target[y:y+height_target, x:x+width_target].reshape(size_target, 1)
            arr_data.extend(target_block)
            low_block = low[y+height_target, x:x+width_low].reshape(size_low, 1)
            arr_data.extend(low_block)
    train_data = numpy.array(arr_data)
    train_x = train_data.reshape(int(train_data.size/size_block), size_block)
    arr_data = []
    for y in range(height_block - 1, height, 1):
        for x in range(halfOfWidthBlock - 1, width - (halfOfWidthBlock - 1), numberOfTargets):
            label = target[y, x:x+numberOfTargets]
            arr_data.append(label)
    label = numpy.array(arr_data)
    train_label = numpy.reshape(label, (len(train_x), numberOfTargets))
    return train_x, train_label, size_block
