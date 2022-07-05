import numpy as np
import os
import sys

import NumCppPy as NumCpp  # noqa E402


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
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    iterator1 = cArray.begin()
    iterator2 = cArray.end()
    assert iterator2 - iterator1 == data.size
    assert iterator1 - iterator2 == -data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator1 = cArray.begin()
    iterator2 = cArray.end()
    assert iterator2 - iterator1 == data.size
    assert iterator1 - iterator2 == -data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator1 = cArray.begin()
    iterator2 = cArray.begin()
    assert iterator1 == iterator2
    assert not iterator1 != iterator2
    assert not iterator1 < iterator2
    assert iterator1 <= iterator2
    assert not iterator1 > iterator2
    assert iterator1 >= iterator2
    iterator1.operatorPlusPlusPre()
    assert not iterator1 == iterator2
    assert iterator1 != iterator2
    assert not iterator1 < iterator2
    assert not iterator1 <= iterator2
    assert iterator1 > iterator2
    assert iterator1 >= iterator2

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator1 = cArray.begin()
    iterator2 = cArray.begin()
    assert iterator1 == iterator2
    assert not iterator1 != iterator2
    assert not iterator1 < iterator2
    assert iterator1 <= iterator2
    assert not iterator1 > iterator2
    assert iterator1 >= iterator2
    iterator1.operatorPlusPlusPre()
    assert not iterator1 == iterator2
    assert iterator1 != iterator2
    assert not iterator1 < iterator2
    assert not iterator1 <= iterator2
    assert iterator1 > iterator2
    assert iterator1 >= iterator2

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.begin()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == data.flatten()[idx]

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
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
    iterator1 = cArray.beginConst()
    iterator2 = cArray.endConst()
    assert iterator2 - iterator1 == data.size
    assert iterator1 - iterator2 == -data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator1 = cArray.beginConst()
    iterator2 = cArray.endConst()
    assert iterator2 - iterator1 == data.size
    assert iterator1 - iterator2 == -data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator1 = cArray.beginConst()
    iterator2 = cArray.beginConst()
    assert iterator1 == iterator2
    assert not iterator1 != iterator2
    assert not iterator1 < iterator2
    assert iterator1 <= iterator2
    assert not iterator1 > iterator2
    assert iterator1 >= iterator2
    iterator1.operatorPlusPlusPre()
    assert not iterator1 == iterator2
    assert iterator1 != iterator2
    assert not iterator1 < iterator2
    assert not iterator1 <= iterator2
    assert iterator1 > iterator2
    assert iterator1 >= iterator2

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator1 = cArray.beginConst()
    iterator2 = cArray.beginConst()
    assert iterator1 == iterator2
    assert not iterator1 != iterator2
    assert not iterator1 < iterator2
    assert iterator1 <= iterator2
    assert not iterator1 > iterator2
    assert iterator1 >= iterator2
    iterator1.operatorPlusPlusPre()
    assert not iterator1 == iterator2
    assert iterator1 != iterator2
    assert not iterator1 < iterator2
    assert not iterator1 <= iterator2
    assert iterator1 > iterator2
    assert iterator1 >= iterator2

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.beginConst()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == data.flatten()[idx]

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.beginConst()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == data.flatten()[idx]


####################################################################################
def test_reverse_iterator():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rbegin()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rbegin()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rbegin()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rbegin()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rbegin()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator += 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rbegin()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator += 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rbegin()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator = iterator + 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rbegin()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator = iterator + 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rend()
    iterator.operatorMinusMinusPre()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rend()
    iterator.operatorMinusMinusPre()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rend()
    iterator.operatorMinusMinusPre()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rend()
    iterator.operatorMinusMinusPre()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rend()
    iterator.operatorMinusMinusPre()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator -= 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rend()
    iterator.operatorMinusMinusPre()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator -= 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rend()
    iterator.operatorMinusMinusPre()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator = iterator - 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rend()
    iterator.operatorMinusMinusPre()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator = iterator - 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator1 = cArray.rbegin()
    iterator2 = cArray.rend()
    assert iterator2 - iterator1 == data.size
    assert iterator1 - iterator2 == -data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator1 = cArray.rbegin()
    iterator2 = cArray.rend()
    assert iterator2 - iterator1 == data.size
    assert iterator1 - iterator2 == -data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator1 = cArray.rbegin()
    iterator2 = cArray.rbegin()
    assert iterator1 == iterator2
    assert not iterator1 != iterator2
    assert not iterator1 < iterator2
    assert iterator1 <= iterator2
    assert not iterator1 > iterator2
    assert iterator1 >= iterator2
    iterator1.operatorPlusPlusPre()
    assert not iterator1 == iterator2
    assert iterator1 != iterator2
    assert not iterator1 < iterator2
    assert not iterator1 <= iterator2
    assert iterator1 > iterator2
    assert iterator1 >= iterator2

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator1 = cArray.rbegin()
    iterator2 = cArray.rbegin()
    assert iterator1 == iterator2
    assert not iterator1 != iterator2
    assert not iterator1 < iterator2
    assert iterator1 <= iterator2
    assert not iterator1 > iterator2
    assert iterator1 >= iterator2
    iterator1.operatorPlusPlusPre()
    assert not iterator1 == iterator2
    assert iterator1 != iterator2
    assert not iterator1 < iterator2
    assert not iterator1 <= iterator2
    assert iterator1 > iterator2
    assert iterator1 >= iterator2

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rbegin()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == np.flip(data.flatten())[idx]

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rbegin()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == np.flip(data.flatten())[idx]


####################################################################################
def test_const_reverse_iterator():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rbeginConst()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rbeginConst()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rbeginConst()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rbeginConst()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rbeginConst()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator += 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rbeginConst()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator += 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rbeginConst()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator = iterator + 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rbeginConst()
    for value in np.flip(data.flatten()):
        assert value == iterator.operatorDereference()
        iterator = iterator + 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rendConst()
    iterator.operatorMinusMinusPre()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rendConst()
    iterator.operatorMinusMinusPre()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rendConst()
    iterator.operatorMinusMinusPre()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rendConst()
    iterator.operatorMinusMinusPre()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rendConst()
    iterator.operatorMinusMinusPre()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator -= 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rendConst()
    iterator.operatorMinusMinusPre()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator -= 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rendConst()
    iterator.operatorMinusMinusPre()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator = iterator - 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rendConst()
    iterator.operatorMinusMinusPre()
    for value in data.flatten():
        assert value == iterator.operatorDereference()
        iterator = iterator - 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator1 = cArray.rbeginConst()
    iterator2 = cArray.rendConst()
    assert iterator2 - iterator1 == data.size
    assert iterator1 - iterator2 == -data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator1 = cArray.rbeginConst()
    iterator2 = cArray.rendConst()
    assert iterator2 - iterator1 == data.size
    assert iterator1 - iterator2 == -data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator1 = cArray.rbeginConst()
    iterator2 = cArray.rbeginConst()
    assert iterator1 == iterator2
    assert not iterator1 != iterator2
    assert not iterator1 < iterator2
    assert iterator1 <= iterator2
    assert not iterator1 > iterator2
    assert iterator1 >= iterator2
    iterator1.operatorPlusPlusPre()
    assert not iterator1 == iterator2
    assert iterator1 != iterator2
    assert not iterator1 < iterator2
    assert not iterator1 <= iterator2
    assert iterator1 > iterator2
    assert iterator1 >= iterator2

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator1 = cArray.rbeginConst()
    iterator2 = cArray.rbeginConst()
    assert iterator1 == iterator2
    assert not iterator1 != iterator2
    assert not iterator1 < iterator2
    assert iterator1 <= iterator2
    assert not iterator1 > iterator2
    assert iterator1 >= iterator2
    iterator1.operatorPlusPlusPre()
    assert not iterator1 == iterator2
    assert iterator1 != iterator2
    assert not iterator1 < iterator2
    assert not iterator1 <= iterator2
    assert iterator1 > iterator2
    assert iterator1 >= iterator2

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rbeginConst()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == np.flip(data.flatten())[idx]

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rbeginConst()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == np.flip(data.flatten())[idx]


####################################################################################
def test_column_iterator():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colbegin()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colbegin()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colbegin()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colbegin()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colbegin()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator += 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colbegin()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator += 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colbegin()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator = iterator + 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colbegin()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator = iterator + 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colend()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colend()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colend()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colend()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colend()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator -= 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colend()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator -= 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colend()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator = iterator - 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colend()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator = iterator - 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator1 = cArray.colbegin()
    iterator2 = cArray.colend()
    assert iterator2 - iterator1 == data.size
    assert iterator1 - iterator2 == -data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator1 = cArray.colbegin()
    iterator2 = cArray.colend()
    assert iterator2 - iterator1 == data.size
    assert iterator1 - iterator2 == -data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator1 = cArray.colbegin()
    iterator2 = cArray.colbegin()
    assert iterator1 == iterator2
    assert not iterator1 != iterator2
    assert not iterator1 < iterator2
    assert iterator1 <= iterator2
    assert not iterator1 > iterator2
    assert iterator1 >= iterator2
    iterator1.operatorPlusPlusPre()
    assert not iterator1 == iterator2
    assert iterator1 != iterator2
    assert not iterator1 < iterator2
    assert not iterator1 <= iterator2
    assert iterator1 > iterator2
    assert iterator1 >= iterator2

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator1 = cArray.colbegin()
    iterator2 = cArray.colbegin()
    assert iterator1 == iterator2
    assert not iterator1 != iterator2
    assert not iterator1 < iterator2
    assert iterator1 <= iterator2
    assert not iterator1 > iterator2
    assert iterator1 >= iterator2
    iterator1.operatorPlusPlusPre()
    assert not iterator1 == iterator2
    assert iterator1 != iterator2
    assert not iterator1 < iterator2
    assert not iterator1 <= iterator2
    assert iterator1 > iterator2
    assert iterator1 >= iterator2

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colbegin()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == data.transpose().flatten()[idx]

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colbegin()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == data.transpose().flatten()[idx]


####################################################################################
def test_const_column_iterator():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colbeginConst()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colbeginConst()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colbeginConst()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colbeginConst()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colbeginConst()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator += 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colbeginConst()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator += 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colbeginConst()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator = iterator + 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colbeginConst()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator = iterator + 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colendConst()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colendConst()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colendConst()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colendConst()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colendConst()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator -= 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colendConst()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator -= 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colendConst()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator = iterator - 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colendConst()
    iterator.operatorMinusMinusPre()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator = iterator - 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator1 = cArray.colbeginConst()
    iterator2 = cArray.colendConst()
    assert iterator2 - iterator1 == data.size
    assert iterator1 - iterator2 == -data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator1 = cArray.colbeginConst()
    iterator2 = cArray.colendConst()
    assert iterator2 - iterator1 == data.size
    assert iterator1 - iterator2 == -data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator1 = cArray.colbeginConst()
    iterator2 = cArray.colbeginConst()
    assert iterator1 == iterator2
    assert not iterator1 != iterator2
    assert not iterator1 < iterator2
    assert iterator1 <= iterator2
    assert not iterator1 > iterator2
    assert iterator1 >= iterator2
    iterator1.operatorPlusPlusPre()
    assert not iterator1 == iterator2
    assert iterator1 != iterator2
    assert not iterator1 < iterator2
    assert not iterator1 <= iterator2
    assert iterator1 > iterator2
    assert iterator1 >= iterator2

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator1 = cArray.colbeginConst()
    iterator2 = cArray.colbeginConst()
    assert iterator1 == iterator2
    assert not iterator1 != iterator2
    assert not iterator1 < iterator2
    assert iterator1 <= iterator2
    assert not iterator1 > iterator2
    assert iterator1 >= iterator2
    iterator1.operatorPlusPlusPre()
    assert not iterator1 == iterator2
    assert iterator1 != iterator2
    assert not iterator1 < iterator2
    assert not iterator1 <= iterator2
    assert iterator1 > iterator2
    assert iterator1 >= iterator2

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.colbeginConst()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == data.transpose().flatten()[idx]

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.colbeginConst()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == data.transpose().flatten()[idx]


####################################################################################
def test_reverse_column_iterator():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolbegin()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolbegin()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolbegin()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolbegin()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolbegin()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator += 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolbegin()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator += 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolbegin()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator = iterator + 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolbegin()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator = iterator + 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolend()
    iterator.operatorMinusMinusPre()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolend()
    iterator.operatorMinusMinusPre()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolend()
    iterator.operatorMinusMinusPre()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolend()
    iterator.operatorMinusMinusPre()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolend()
    iterator.operatorMinusMinusPre()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator -= 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolend()
    iterator.operatorMinusMinusPre()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator -= 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolend()
    iterator.operatorMinusMinusPre()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator = iterator - 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolend()
    iterator.operatorMinusMinusPre()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator = iterator - 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator1 = cArray.rcolbegin()
    iterator2 = cArray.rcolend()
    assert iterator2 - iterator1 == data.size
    assert iterator1 - iterator2 == -data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator1 = cArray.rcolbegin()
    iterator2 = cArray.rcolend()
    assert iterator2 - iterator1 == data.size
    assert iterator1 - iterator2 == -data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator1 = cArray.rcolbegin()
    iterator2 = cArray.rcolbegin()
    assert iterator1 == iterator2
    assert not iterator1 != iterator2
    assert not iterator1 < iterator2
    assert iterator1 <= iterator2
    assert not iterator1 > iterator2
    assert iterator1 >= iterator2
    iterator1.operatorPlusPlusPre()
    assert not iterator1 == iterator2
    assert iterator1 != iterator2
    assert not iterator1 < iterator2
    assert not iterator1 <= iterator2
    assert iterator1 > iterator2
    assert iterator1 >= iterator2

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator1 = cArray.rcolbegin()
    iterator2 = cArray.rcolbegin()
    assert iterator1 == iterator2
    assert not iterator1 != iterator2
    assert not iterator1 < iterator2
    assert iterator1 <= iterator2
    assert not iterator1 > iterator2
    assert iterator1 >= iterator2
    iterator1.operatorPlusPlusPre()
    assert not iterator1 == iterator2
    assert iterator1 != iterator2
    assert not iterator1 < iterator2
    assert not iterator1 <= iterator2
    assert iterator1 > iterator2
    assert iterator1 >= iterator2

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolbegin()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == np.flip(data.transpose().flatten())[idx]

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolbegin()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == np.flip(data.transpose().flatten())[idx]


####################################################################################
def test_const_reverse_column_iterator():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolbeginConst()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolbeginConst()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolbeginConst()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolbeginConst()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator.operatorPlusPlusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolbeginConst()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator += 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolbeginConst()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator += 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolbeginConst()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator = iterator + 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolbeginConst()
    for value in np.flip(data.transpose().flatten()):
        assert value == iterator.operatorDereference()
        iterator = iterator + 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolendConst()
    iterator.operatorMinusMinusPre()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolendConst()
    iterator.operatorMinusMinusPre()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPre()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolendConst()
    iterator.operatorMinusMinusPre()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolendConst()
    iterator.operatorMinusMinusPre()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator.operatorMinusMinusPost()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolendConst()
    iterator.operatorMinusMinusPre()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator -= 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolendConst()
    iterator.operatorMinusMinusPre()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator -= 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolendConst()
    iterator.operatorMinusMinusPre()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator = iterator - 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolendConst()
    iterator.operatorMinusMinusPre()
    for value in data.transpose().flatten():
        assert value == iterator.operatorDereference()
        iterator = iterator - 1

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator1 = cArray.rcolbeginConst()
    iterator2 = cArray.rcolendConst()
    assert iterator2 - iterator1 == data.size
    assert iterator1 - iterator2 == -data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator1 = cArray.rcolbeginConst()
    iterator2 = cArray.rcolendConst()
    assert iterator2 - iterator1 == data.size
    assert iterator1 - iterator2 == -data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator1 = cArray.rcolbeginConst()
    iterator2 = cArray.rcolbeginConst()
    assert iterator1 == iterator2
    assert not iterator1 != iterator2
    assert not iterator1 < iterator2
    assert iterator1 <= iterator2
    assert not iterator1 > iterator2
    assert iterator1 >= iterator2
    iterator1.operatorPlusPlusPre()
    assert not iterator1 == iterator2
    assert iterator1 != iterator2
    assert not iterator1 < iterator2
    assert not iterator1 <= iterator2
    assert iterator1 > iterator2
    assert iterator1 >= iterator2

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator1 = cArray.rcolbeginConst()
    iterator2 = cArray.rcolbeginConst()
    assert iterator1 == iterator2
    assert not iterator1 != iterator2
    assert not iterator1 < iterator2
    assert iterator1 <= iterator2
    assert not iterator1 > iterator2
    assert iterator1 >= iterator2
    iterator1.operatorPlusPlusPre()
    assert not iterator1 == iterator2
    assert iterator1 != iterator2
    assert not iterator1 < iterator2
    assert not iterator1 <= iterator2
    assert iterator1 > iterator2
    assert iterator1 >= iterator2

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    iterator = cArray.rcolbeginConst()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == np.flip(data.transpose().flatten())[idx]

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    iterator = cArray.rcolbeginConst()
    idx = np.random.randint(0, shape.size())
    assert iterator[idx] == np.flip(data.transpose().flatten())[idx]
