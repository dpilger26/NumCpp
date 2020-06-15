import numpy as np
import os
import sys
sys.path.append(os.path.abspath(r'../lib'))
import NumCpp  # noqa E402


####################################################################################
def test_iterator():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.begin()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.begin()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.begin()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator += 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.begin()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator = iterator + 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.end()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.end()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.end()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator -= 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.end()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator = iterator - 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.begin()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == data.flatten()[idx]


####################################################################################
def test_const_iterator():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.beginConst()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.beginConst()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.beginConst()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator += 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.beginConst()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator = iterator + 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.endConst()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.endConst()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.endConst()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator -= 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.endConst()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator = iterator - 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.beginConst()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == data.flatten()[idx]


####################################################################################
def test_reverse_iterator():
    pass


####################################################################################
def test_const_reverse_iterator():
    pass


####################################################################################
def test_column_iterator():
    pass


####################################################################################
def test_const_column_iterator():
    pass


####################################################################################
def test_reverse_column_iterator():
    pass


####################################################################################
def test_const_reverse_column_iterator():
    pass


# if __name__ == '__main__':
#     test_iterator()
#     test_const_iterator()
