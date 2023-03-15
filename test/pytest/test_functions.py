import os
import tempfile
import numpy as np
import scipy.ndimage as ndimage
from functools import reduce
import warnings

import NumCppPy as NumCpp  # noqa E402


####################################################################################
def factors(n):
    """docstring"""
    return set(reduce(list.__add__,
                      ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


####################################################################################
def test_seed():
    np.random.seed(26)


####################################################################################
def test_abs():
    randValue = np.random.randint(-100, -1, [1, ]).astype(float).item()
    assert NumCpp.absScalar(randValue) == np.abs(randValue)

    components = np.random.randint(-100, -1, [2, ]).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.absScalar(value), 9) == np.round(np.abs(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.absArray(cArray), np.abs(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols]) + \
        1j * np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.absArray(
        cArray), 9), np.round(np.abs(data), 9))


####################################################################################
def test_add():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.add(cArray1, cArray2), data1 + data2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    value = np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.add(cArray, value), data + value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    value = np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.add(value, cArray), data + value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.add(cArray1, cArray2), data1 + data2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(-100, 100) + 1j * np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.add(cArray, value), data + value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(-100, 100) + 1j * np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.add(value, cArray), data + value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArray(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.add(cArray1, cArray2), data1 + data2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.add(cArray1, cArray2), data1 + data2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    value = np.random.randint(-100, 100) + 1j * np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.add(cArray, value), data + value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    value = np.random.randint(-100, 100) + 1j * np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.add(value, cArray), data + value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.add(cArray, value), data + value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.add(value, cArray), data + value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.add(cArray1, cArray2), data1 + data2)


####################################################################################
def test_alen():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.alen(cArray) == shape.rows


####################################################################################
def test_all():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.all(cArray, NumCpp.Axis.NONE).astype(
        bool).item() == np.all(data).item()

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.all(cArray, NumCpp.Axis.NONE).astype(
        bool).item() == np.all(data).item()

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.all(
        cArray, NumCpp.Axis.ROW).flatten().astype(bool), np.all(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.all(
        cArray, NumCpp.Axis.ROW).flatten().astype(bool), np.all(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.all(
        cArray, NumCpp.Axis.COL).flatten().astype(bool), np.all(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.all(
        cArray, NumCpp.Axis.COL).flatten().astype(bool), np.all(data, axis=1))


####################################################################################
def test_allclose():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    cArray3 = NumCpp.NdArray(shape)
    tolerance = 1e-5
    data1 = np.random.randn(shape.rows, shape.cols)
    data2 = data1 + tolerance / 10
    data3 = data1 + 1
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    assert NumCpp.allclose(cArray1, cArray2, tolerance) and not NumCpp.allclose(
        cArray1, cArray3, tolerance)


####################################################################################
def test_amax():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.amax(cArray, NumCpp.Axis.NONE).item() == np.max(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.amax(cArray, NumCpp.Axis.NONE).item() == np.max(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.amax(
        cArray, NumCpp.Axis.ROW).flatten(), np.max(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.amax(
        cArray, NumCpp.Axis.ROW).flatten(), np.max(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.amax(
        cArray, NumCpp.Axis.COL).flatten(), np.max(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.amax(
        cArray, NumCpp.Axis.COL).flatten(), np.max(data, axis=1))


####################################################################################
def test_amin():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.amin(cArray, NumCpp.Axis.NONE).item() == np.min(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.amin(cArray, NumCpp.Axis.NONE).item() == np.min(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.amin(
        cArray, NumCpp.Axis.ROW).flatten(), np.min(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.amin(
        cArray, NumCpp.Axis.ROW).flatten(), np.min(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.amin(
        cArray, NumCpp.Axis.COL).flatten(), np.min(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.amin(
        cArray, NumCpp.Axis.COL).flatten(), np.min(data, axis=1))


####################################################################################
def test_angle():
    components = np.random.randint(-100, -1, [2, ]).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.angleScalar(value), 9) == np.round(np.angle(value), 9)  # noqa

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols]) + \
        1j * np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.angleArray(
        cArray), 9), np.round(np.angle(data), 9))


####################################################################################
def test_any():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.any(cArray, NumCpp.Axis.NONE).astype(
        bool).item() == np.any(data).item()

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.any(cArray, NumCpp.Axis.NONE).astype(
        bool).item() == np.any(data).item()

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.any(
        cArray, NumCpp.Axis.ROW).flatten().astype(bool), np.any(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.any(
        cArray, NumCpp.Axis.ROW).flatten().astype(bool), np.any(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.any(
        cArray, NumCpp.Axis.COL).flatten().astype(bool), np.any(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.any(
        cArray, NumCpp.Axis.COL).flatten().astype(bool), np.any(data, axis=1))


####################################################################################
def test_append():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.append(cArray1, cArray2, NumCpp.Axis.NONE).getNumpyArray().flatten(),
                          np.append(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    numRows = np.random.randint(1, 100, [1, ]).item()
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[0].item() + numRows, shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    data1 = np.random.randint(0, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(0, 100, [shape2.rows, shape2.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.append(cArray1, cArray2, NumCpp.Axis.ROW).getNumpyArray(),
                          np.append(data1, data2, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    NumCppols = np.random.randint(1, 100, [1, ]).item()
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[0].item(),
                          shapeInput[1].item() + NumCppols)
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    data1 = np.random.randint(0, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(0, 100, [shape2.rows, shape2.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.append(cArray1, cArray2, NumCpp.Axis.COL).getNumpyArray(),
                          np.append(data1, data2, axis=1))


####################################################################################
def test_arange():
    start = np.random.randn(1).item()
    stop = np.random.randn(1).item() * 100
    step = np.abs(np.random.randn(1).item())
    if stop < start:
        step *= -1
    data = np.arange(start, stop, step)
    assert np.array_equal(np.round(NumCpp.arange(
        start, stop, step).flatten(), 9), np.round(data, 9))


####################################################################################
def test_arccos():
    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.arccosScalar(value),
                    9) == np.round(np.arccos(value), 9)

    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.arccosScalar(value),
                    9) == np.round(np.arccos(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.arccosArray(
        cArray), 9), np.round(np.arccos(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.arccosArray(
        cArray), 9), np.round(np.arccos(data), 9))


####################################################################################
def test_arccosh():
    value = np.abs(np.random.rand(1).item()) + 1
    assert np.round(NumCpp.arccoshScalar(value),
                    9) == np.round(np.arccosh(value), 9)

    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.arccoshScalar(value),
                    9) == np.round(np.arccosh(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.arccoshArray(
        cArray), 9), np.round(np.arccosh(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.arccoshArray(
        cArray), 9), np.round(np.arccosh(data), 9))


####################################################################################
def test_arcsin():
    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.arcsinScalar(value),
                    9) == np.round(np.arcsin(value), 9)

    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.arcsinScalar(value),
                    9) == np.round(np.arcsin(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.arcsinArray(
        cArray), 9), np.round(np.arcsin(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    np.array_equal(np.round(NumCpp.arcsinArray(cArray), 9),
                   np.round(np.arcsin(data), 9))


####################################################################################
def test_arcsinh():
    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.arcsinhScalar(value),
                    9) == np.round(np.arcsinh(value), 9)

    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.arcsinhScalar(value),
                    9) == np.round(np.arcsinh(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.arcsinhArray(
        cArray), 9), np.round(np.arcsinh(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    np.array_equal(np.round(NumCpp.arcsinhArray(cArray), 9),
                   np.round(np.arcsinh(data), 9))


####################################################################################
def test_arctan():
    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.arctanScalar(value),
                    9) == np.round(np.arctan(value), 9)

    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.arctanScalar(value),
                    9) == np.round(np.arctan(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.arctanArray(
        cArray), 9), np.round(np.arctan(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    np.array_equal(np.round(NumCpp.arctanArray(cArray), 9),
                   np.round(np.arctan(data), 9))


####################################################################################
def test_arctan2():
    xy = np.random.rand(2) * 2 - 1
    assert np.round(NumCpp.arctan2Scalar(xy[1], xy[0]), 9) == np.round(
        np.arctan2(xy[1], xy[0]), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArrayX = NumCpp.NdArray(shape)
    cArrayY = NumCpp.NdArray(shape)
    xy = np.random.rand(*shapeInput, 2) * 2 - 1
    xData = xy[:, :, 0].reshape(shapeInput)
    yData = xy[:, :, 1].reshape(shapeInput)
    cArrayX.setArray(xData)
    cArrayY.setArray(yData)
    assert np.array_equal(np.round(NumCpp.arctan2Array(
        cArrayY, cArrayX), 9), np.round(np.arctan2(yData, xData), 9))


####################################################################################
def test_arctanh():
    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.arctanhScalar(value),
                    9) == np.round(np.arctanh(value), 9)

    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.arctanhScalar(value),
                    9) == np.round(np.arctanh(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.arctanhArray(
        cArray), 9), np.round(np.arctanh(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    np.array_equal(np.round(NumCpp.arctanhArray(cArray), 9),
                   np.round(np.arctanh(data), 9))


####################################################################################
def test_argmax():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmax(
        cArray, NumCpp.Axis.NONE).item(), np.argmax(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmax(
        cArray, NumCpp.Axis.NONE).item(), np.argmax(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmax(
        cArray, NumCpp.Axis.ROW).flatten(), np.argmax(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmax(
        cArray, NumCpp.Axis.ROW).flatten(), np.argmax(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmax(
        cArray, NumCpp.Axis.COL).flatten(), np.argmax(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmax(
        cArray, NumCpp.Axis.COL).flatten(), np.argmax(data, axis=1))


####################################################################################
def test_argmin():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmin(
        cArray, NumCpp.Axis.NONE).item(), np.argmin(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmin(
        cArray, NumCpp.Axis.NONE).item(), np.argmin(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmin(
        cArray, NumCpp.Axis.ROW).flatten(), np.argmin(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmin(
        cArray, NumCpp.Axis.ROW).flatten(), np.argmin(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmin(
        cArray, NumCpp.Axis.COL).flatten(), np.argmin(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmin(
        cArray, NumCpp.Axis.COL).flatten(), np.argmin(data, axis=1))


####################################################################################
def test_argsort():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    dataFlat = data.flatten()
    assert np.array_equal(dataFlat[NumCpp.argsort(cArray, NumCpp.Axis.NONE).flatten().astype(np.uint32)],
                          dataFlat[np.argsort(data, axis=None)])

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    dataFlat = data.flatten()
    assert np.array_equal(dataFlat[NumCpp.argsort(cArray, NumCpp.Axis.NONE).flatten().astype(np.uint32)],
                          dataFlat[np.argsort(data, axis=None)])

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    pIdx = np.argsort(data, axis=0)
    cIdx = NumCpp.argsort(cArray, NumCpp.Axis.ROW).astype(np.uint16)
    allPass = True
    for idx, row in enumerate(data.T):
        if not np.array_equal(row[cIdx[:, idx]], row[pIdx[:, idx]]):
            allPass = False
            break
    assert allPass

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    pIdx = np.argsort(data, axis=0)
    cIdx = NumCpp.argsort(cArray, NumCpp.Axis.ROW).astype(np.uint16)
    allPass = True
    for idx, row in enumerate(data.T):
        if not np.array_equal(row[cIdx[:, idx]], row[pIdx[:, idx]]):
            allPass = False
            break
    assert allPass

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    pIdx = np.argsort(data, axis=1)
    cIdx = NumCpp.argsort(cArray, NumCpp.Axis.COL).astype(np.uint16)
    allPass = True
    for idx, row in enumerate(data):
        if not np.array_equal(row[cIdx[idx, :]], row[pIdx[idx, :]]):  # noqa
            allPass = False
            break
    assert allPass

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    pIdx = np.argsort(data, axis=1)
    cIdx = NumCpp.argsort(cArray, NumCpp.Axis.COL).astype(np.uint16)
    allPass = True
    for idx, row in enumerate(data):
        if not np.array_equal(row[cIdx[idx, :]], row[pIdx[idx, :]]):
            allPass = False
            break
    assert allPass


####################################################################################
def test_argwhere():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    randValue = np.random.randint(0, 100, [1, ]).item()
    data2 = data > randValue
    cArray.setArray(data2)
    assert np.array_equal(NumCpp.argwhere(cArray).flatten(
    ), np.argwhere(data.flatten() > randValue).flatten())

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    randValue = np.random.randint(0, 100, [1, ]).item()
    data2 = data > randValue
    cArray.setArray(data2)
    assert np.array_equal(NumCpp.argwhere(cArray).flatten(
    ), np.argwhere(data.flatten() > randValue).flatten())


####################################################################################
def test_around():
    value = np.abs(np.random.rand(1).item()) * \
        np.random.randint(1, 10, [1, ]).item()
    numDecimalsRound = np.random.randint(0, 10, [1, ]).astype(np.uint8).item()
    assert NumCpp.aroundScalar(value, numDecimalsRound) == np.round(
        value, numDecimalsRound)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * \
        np.random.randint(1, 10, [1, ]).item()
    cArray.setArray(data)
    numDecimalsRound = np.random.randint(0, 10, [1, ]).astype(np.uint8).item()
    assert np.array_equal(NumCpp.aroundArray(
        cArray, numDecimalsRound), np.round(data, numDecimalsRound))


####################################################################################
def test_array_equal():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    cArray3 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, shapeInput)
    data2 = np.random.randint(1, 100, shapeInput)
    cArray1.setArray(data1)
    cArray2.setArray(data1)
    cArray3.setArray(data2)
    assert NumCpp.array_equal(
        cArray1, cArray2) and not NumCpp.array_equal(cArray1, cArray3)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    cArray3 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data1)
    cArray3.setArray(data2)
    assert NumCpp.array_equal(
        cArray1, cArray2) and not NumCpp.array_equal(cArray1, cArray3)


####################################################################################
def test_array_equiv():
    shapeInput1 = np.random.randint(1, 100, [2, ])
    shapeInput3 = np.random.randint(1, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput1[0].item(), shapeInput1[1].item())
    shape2 = NumCpp.Shape(shapeInput1[1].item(), shapeInput1[0].item())
    shape3 = NumCpp.Shape(shapeInput3[0].item(), shapeInput3[1].item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    cArray3 = NumCpp.NdArray(shape3)
    data1 = np.random.randint(1, 100, shapeInput1)
    data3 = np.random.randint(1, 100, shapeInput3)
    cArray1.setArray(data1)
    cArray2.setArray(data1.reshape(
        [shapeInput1[1].item(), shapeInput1[0].item()]))
    cArray3.setArray(data3)
    assert NumCpp.array_equiv(
        cArray1, cArray2) and not NumCpp.array_equiv(cArray1, cArray3)

    shapeInput1 = np.random.randint(1, 100, [2, ])
    shapeInput3 = np.random.randint(1, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput1[0].item(), shapeInput1[1].item())
    shape2 = NumCpp.Shape(shapeInput1[1].item(), shapeInput1[0].item())
    shape3 = NumCpp.Shape(shapeInput3[0].item(), shapeInput3[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape1)
    cArray2 = NumCpp.NdArrayComplexDouble(shape2)
    cArray3 = NumCpp.NdArrayComplexDouble(shape3)
    real1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    imag1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data1 = real1 + 1j * imag1
    real3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    imag3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    data3 = real3 + 1j * imag3
    cArray1.setArray(data1)
    cArray2.setArray(data1.reshape(
        [shapeInput1[1].item(), shapeInput1[0].item()]))
    cArray3.setArray(data3)
    assert NumCpp.array_equiv(
        cArray1, cArray2) and not NumCpp.array_equiv(cArray1, cArray3)


####################################################################################
def test_asarray():
    values = np.random.randint(0, 100, [2, ]).astype(float)
    assert np.array_equal(NumCpp.asarrayArray1D(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayArray1D(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    assert np.array_equal(NumCpp.asarrayArray1DCopy(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayArray1DCopy(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayArray2D(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayArray2D(*values), data)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayArray2DCopy(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayArray2DCopy(*values), data)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    assert np.array_equal(NumCpp.asarrayVector1D(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayVector1D(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    assert np.array_equal(
        NumCpp.asarrayVector1DCopy(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    assert np.array_equal(
        NumCpp.asarrayVector1DCopy(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayVector2D(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayVector2D(*values), data)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayVectorArray2D(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayVectorArray2D(*values), data)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayVectorArray2DCopy(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayVectorArray2DCopy(*values), data)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    assert np.array_equal(NumCpp.asarrayDeque1D(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayDeque1D(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayDeque2D(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayDeque2D(*values), data)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    assert np.array_equal(NumCpp.asarrayList(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayList(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    assert np.array_equal(NumCpp.asarrayIterators(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayIterators(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    assert np.array_equal(
        NumCpp.asarrayPointerIterators(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    assert np.array_equal(
        NumCpp.asarrayPointerIterators(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    assert np.array_equal(NumCpp.asarrayPointer(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayPointer(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayPointer2D(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayPointer2D(*values), data)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    assert np.array_equal(
        NumCpp.asarrayPointerShell(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    assert np.array_equal(
        NumCpp.asarrayPointerShell(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayPointerShell2D(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayPointerShell2D(*values), data)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    assert np.array_equal(
        NumCpp.asarrayPointerShellTakeOwnership(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    assert np.array_equal(
        NumCpp.asarrayPointerShellTakeOwnership(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(float)
    data = np.vstack([values, values])
    assert np.array_equal(
        NumCpp.asarrayPointerShell2DTakeOwnership(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(float)
    imag = np.random.randint(0, 100, [2, ]).astype(float)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(
        NumCpp.asarrayPointerShell2DTakeOwnership(*values), data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    cArrayCast = NumCpp.astypeDoubleToUint32(cArray).getNumpyArray()
    assert np.array_equal(cArrayCast, data.astype(np.uint32))
    assert cArrayCast.dtype == np.uint32

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    cArrayCast = NumCpp.astypeDoubleToComplex(cArray).getNumpyArray()
    assert np.array_equal(cArrayCast, data.astype(np.complex128))
    assert cArrayCast.dtype == np.complex128

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    cArrayCast = NumCpp.astypeComplexToComplex(cArray).getNumpyArray()
    assert np.array_equal(cArrayCast, data.astype(np.complex64))
    assert cArrayCast.dtype == np.complex64

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    cArrayCast = NumCpp.astypeComplexToDouble(cArray).getNumpyArray()
    warnings.filterwarnings('ignore', category=np.ComplexWarning)
    assert np.array_equal(cArrayCast, data.astype(float))
    warnings.filters.pop()  # noqa
    assert cArrayCast.dtype == float


####################################################################################
def test_average():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.round(NumCpp.average(cArray, NumCpp.Axis.NONE).item(),
                    9) == np.round(np.average(data), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.round(NumCpp.average(cArray, NumCpp.Axis.NONE).item(),
                    9) == np.round(np.average(data), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.average(cArray, NumCpp.Axis.ROW).flatten(), 9),
                          np.round(np.average(data, axis=0), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.average(cArray, NumCpp.Axis.ROW).flatten(), 9),
                          np.round(np.average(data, axis=0), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.average(cArray, NumCpp.Axis.COL).flatten(), 9),
                          np.round(np.average(data, axis=1), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.average(cArray, NumCpp.Axis.COL).flatten(), 9),
                          np.round(np.average(data, axis=1), 9))


####################################################################################
def test_averageWeighted():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    cWeights = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    weights = np.random.randint(1, 5, [shape.rows, shape.cols])
    cArray.setArray(data)
    cWeights.setArray(weights)
    assert np.round(NumCpp.averageWeighted(cArray, cWeights, NumCpp.Axis.NONE).item(), 9) == \
        np.round(np.average(data, weights=weights), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cWeights = NumCpp.NdArray(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    weights = np.random.randint(1, 5, [shape.rows, shape.cols])
    cArray.setArray(data)
    cWeights.setArray(weights)
    assert np.round(NumCpp.averageWeighted(cArray, cWeights, NumCpp.Axis.NONE).item(), 9) == \
        np.round(np.average(data, weights=weights), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    cWeights = NumCpp.NdArray(1, shape.cols)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    weights = np.random.randint(1, 5, [1, shape.rows])
    cArray.setArray(data)
    cWeights.setArray(weights)
    assert np.array_equal(np.round(NumCpp.averageWeighted(cArray, cWeights, NumCpp.Axis.ROW).flatten(), 9),
                          np.round(np.average(data, weights=weights.flatten(), axis=0), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cWeights = NumCpp.NdArray(1, shape.cols)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    weights = np.random.randint(1, 5, [1, shape.rows])
    cArray.setArray(data)
    cWeights.setArray(weights)
    assert np.array_equal(np.round(NumCpp.averageWeighted(cArray, cWeights, NumCpp.Axis.ROW).flatten(), 9),
                          np.round(np.average(data, weights=weights.flatten(), axis=0), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    cWeights = NumCpp.NdArray(1, shape.rows)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    weights = np.random.randint(1, 5, [1, shape.cols])
    cWeights.setArray(weights)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.averageWeighted(cArray, cWeights, NumCpp.Axis.COL).flatten(), 9),
                          np.round(np.average(data, weights=weights.flatten(), axis=1), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cWeights = NumCpp.NdArray(1, shape.rows)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    weights = np.random.randint(1, 5, [1, shape.cols])
    cWeights.setArray(weights)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.averageWeighted(cArray, cWeights, NumCpp.Axis.COL).flatten(), 9),
                          np.round(np.average(data, weights=weights.flatten(), axis=1), 9))


####################################################################################
def test_bartlett():
    m = np.random.randint(2, 100)
    assert np.array_equal(np.round(NumCpp.bartlett(m), 9).flatten(),
                          np.round(np.bartlett(m), 9))


####################################################################################
def test_binaryRepr():
    value = np.random.randint(0, np.iinfo(np.uint64).max, [
                              1, ], dtype=np.uint64).item()
    assert NumCpp.binaryRepr(np.uint64(value)) == np.binary_repr(
        value, np.iinfo(np.uint64).bits)


####################################################################################
def test_bincount():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.bincount(cArray, 0).flatten(),
                          np.bincount(data.flatten(), minlength=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    cArray.setArray(data)
    minLength = int(np.max(data) + 10)
    assert np.array_equal(NumCpp.bincount(cArray, minLength).flatten(),
                          np.bincount(data.flatten(), minlength=minLength))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    cWeights = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    weights = np.random.randint(
        0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    cArray.setArray(data)
    cWeights.setArray(weights)
    assert np.array_equal(NumCpp.bincountWeighted(cArray, cWeights, 0).flatten(),
                          np.bincount(data.flatten(), minlength=0, weights=weights.flatten()))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    cWeights = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    weights = np.random.randint(
        0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    cArray.setArray(data)
    cWeights.setArray(weights)
    minLength = int(np.max(data) + 10)
    assert np.array_equal(NumCpp.bincountWeighted(cArray, cWeights, minLength).flatten(),
                          np.bincount(data.flatten(), minlength=minLength, weights=weights.flatten()))


####################################################################################
def test_bit_count():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt64(shape)
    data = np.random.randint(0, np.iinfo(np.uint64).max, [
                             shape.rows, shape.cols], dtype=np.uint64)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.bit_count(cArray),
                          np.array(list(map(lambda x: bin(x).count('1'), data.flatten()))).reshape(data.shape))


####################################################################################
def test_bitwise_and():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt64(shape)
    cArray2 = NumCpp.NdArrayUInt64(shape)
    data1 = np.random.randint(
        0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    data2 = np.random.randint(
        0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.bitwise_and(
        cArray1, cArray2), np.bitwise_and(data1, data2))


####################################################################################
def test_bitwise_not():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt64(shape)
    data = np.random.randint(
        0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.bitwise_not(cArray), np.bitwise_not(data))


####################################################################################
def test_bitwise_or():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt64(shape)
    cArray2 = NumCpp.NdArrayUInt64(shape)
    data1 = np.random.randint(
        0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    data2 = np.random.randint(
        0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.bitwise_or(
        cArray1, cArray2), np.bitwise_or(data1, data2))


####################################################################################
def test_bitwise_xor():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt64(shape)
    cArray2 = NumCpp.NdArrayUInt64(shape)
    data1 = np.random.randint(
        0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    data2 = np.random.randint(
        0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.bitwise_xor(
        cArray1, cArray2), np.bitwise_xor(data1, data2))


####################################################################################
def test_blackman():
    m = np.random.randint(2, 100)
    assert np.array_equal(np.round(NumCpp.blackman(m), 9).flatten(),
                          np.round(np.blackman(m), 9))


####################################################################################
def test_byteswap():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt64(shape)
    data = np.random.randint(
        0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.byteswap(cArray).shape, shapeInput)


####################################################################################
def test_cbrt():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.cbrtArray(
        cArray), 9), np.round(np.cbrt(data), 9))


####################################################################################
def test_ceil():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols).astype(float) * 1000
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.ceilArray(
        cArray), 9), np.round(np.ceil(data), 9))


####################################################################################
def test_center_of_mass():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols).astype(float) * 1000
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.centerOfMass(cArray, NumCpp.Axis.NONE).flatten(), 9),
                          np.round(ndimage.center_of_mass(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols).astype(float) * 1000
    cArray.setArray(data)

    coms = list()
    for col in range(data.shape[1]):
        coms.append(np.round(ndimage.center_of_mass(data[:, col])[0], 9))

    assert np.array_equal(np.round(NumCpp.centerOfMass(
        cArray, NumCpp.Axis.ROW).flatten(), 9), np.round(coms, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols).astype(float) * 1000
    cArray.setArray(data)

    coms = list()
    for row in range(data.shape[0]):
        coms.append(np.round(ndimage.center_of_mass(data[row, :])[0], 9))

    assert np.array_equal(np.round(NumCpp.centerOfMass(
        cArray, NumCpp.Axis.COL).flatten(), 9), np.round(coms, 9))


####################################################################################
def test_clip():
    value = np.random.randint(0, 100, [1, ]).item()
    minValue = np.random.randint(0, 10, [1, ]).item()
    maxValue = np.random.randint(90, 100, [1, ]).item()
    assert NumCpp.clipScalar(value, minValue, maxValue) == np.clip(
        value, minValue, maxValue)

    value = np.random.randint(0, 100, [1, ]).item(
    ) + 1j * np.random.randint(0, 100, [1, ]).item()
    minValue = np.random.randint(0, 10, [1, ]).item(
    ) + 1j * np.random.randint(0, 10, [1, ]).item()
    maxValue = np.random.randint(90, 100, [1, ]).item(
    ) + 1j * np.random.randint(0, 100, [1, ]).item()
    assert NumCpp.clipScalar(value, minValue, maxValue) == np.clip(value, minValue, maxValue)  # noqa

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    minValue = np.random.randint(0, 10, [1, ]).item()
    maxValue = np.random.randint(90, 100, [1, ]).item()
    assert np.array_equal(NumCpp.clipArray(
        cArray, minValue, maxValue), np.clip(data, minValue, maxValue))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    minValue = np.random.randint(0, 10, [1, ]).item(
    ) + 1j * np.random.randint(0, 10, [1, ]).item()
    maxValue = np.random.randint(90, 100, [1, ]).item(
    ) + 1j * np.random.randint(0, 100, [1, ]).item()
    assert np.array_equal(NumCpp.clipArray(cArray, minValue, maxValue), np.clip(data, minValue, maxValue))  # noqa


####################################################################################
def test_column_stack():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item(
    ) + np.random.randint(1, 10, [1, ]).item())
    shape3 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item(
    ) + np.random.randint(1, 10, [1, ]).item())
    shape4 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item(
    ) + np.random.randint(1, 10, [1, ]).item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    cArray3 = NumCpp.NdArray(shape3)
    cArray4 = NumCpp.NdArray(shape4)
    data1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    data4 = np.random.randint(1, 100, [shape4.rows, shape4.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    assert np.array_equal(NumCpp.column_stack(cArray1, cArray2, cArray3, cArray4),
                          np.column_stack([data1, data2, data3, data4]))


####################################################################################
def test_complex():
    real = np.random.rand(1).astype(float).item()
    value = complex(real)
    assert np.round(NumCpp.complexScalar(real), 9) == np.round(value, 9)

    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.complexScalar(
        components[0], components[1]), 9) == np.round(value, 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    realArray = NumCpp.NdArray(shape)
    real = np.random.rand(shape.rows, shape.cols)
    realArray.setArray(real)
    assert np.array_equal(np.round(NumCpp.complexArray(
        realArray), 9), np.round(real + 1j * np.zeros_like(real), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    realArray = NumCpp.NdArray(shape)
    imagArray = NumCpp.NdArray(shape)
    real = np.random.rand(shape.rows, shape.cols)
    imag = np.random.rand(shape.rows, shape.cols)
    realArray.setArray(real)
    imagArray.setArray(imag)
    assert np.array_equal(np.round(NumCpp.complexArray(
        realArray, imagArray), 9), np.round(real + 1j * imag, 9))


####################################################################################
def test_concatenate():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item(
    ) + np.random.randint(1, 10, [1, ]).item())
    shape3 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item(
    ) + np.random.randint(1, 10, [1, ]).item())
    shape4 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item(
    ) + np.random.randint(1, 10, [1, ]).item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    cArray3 = NumCpp.NdArray(shape3)
    cArray4 = NumCpp.NdArray(shape4)
    data1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    data4 = np.random.randint(1, 100, [shape4.rows, shape4.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    assert np.array_equal(NumCpp.concatenate(cArray1, cArray2, cArray3, cArray4, NumCpp.Axis.NONE).flatten(),
                          np.concatenate([data1.flatten(), data2.flatten(), data3.flatten(), data4.flatten()]))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[0].item(
    ) + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
    shape3 = NumCpp.Shape(shapeInput[0].item(
    ) + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
    shape4 = NumCpp.Shape(shapeInput[0].item(
    ) + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    cArray3 = NumCpp.NdArray(shape3)
    cArray4 = NumCpp.NdArray(shape4)
    data1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    data4 = np.random.randint(1, 100, [shape4.rows, shape4.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    assert np.array_equal(NumCpp.concatenate(cArray1, cArray2, cArray3, cArray4, NumCpp.Axis.ROW),
                          np.concatenate([data1, data2, data3, data4], axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item(
    ) + np.random.randint(1, 10, [1, ]).item())
    shape3 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item(
    ) + np.random.randint(1, 10, [1, ]).item())
    shape4 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item(
    ) + np.random.randint(1, 10, [1, ]).item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    cArray3 = NumCpp.NdArray(shape3)
    cArray4 = NumCpp.NdArray(shape4)
    data1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    data4 = np.random.randint(1, 100, [shape4.rows, shape4.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    assert np.array_equal(NumCpp.concatenate(cArray1, cArray2, cArray3, cArray4, NumCpp.Axis.COL),
                          np.concatenate([data1, data2, data3, data4], axis=1))


####################################################################################
def test_conj():
    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.conjScalar(value), 9) == np.round(np.conj(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.conjArray(
        cArray), 9), np.round(np.conj(data), 9))


####################################################################################
def test_contains():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    value = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    assert NumCpp.contains(
        cArray, value, NumCpp.Axis.NONE).getNumpyArray().item() == (value in data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    value = np.random.randint(0, 100, [1, ]).item(
    ) + 1j * np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    assert NumCpp.contains(
        cArray, value, NumCpp.Axis.NONE).getNumpyArray().item() == (value in data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    value = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    truth = list()
    for row in data:
        truth.append(value in row)
    assert np.array_equal(NumCpp.contains(
        cArray, value, NumCpp.Axis.COL).getNumpyArray().flatten(), np.asarray(truth))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    value = np.random.randint(0, 100, [1, ]).item(
    ) + 1j * np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    truth = list()
    for row in data:
        truth.append(value in row)
    assert np.array_equal(NumCpp.contains(
        cArray, value, NumCpp.Axis.COL).getNumpyArray().flatten(), np.asarray(truth))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    value = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    truth = list()
    for row in data.T:
        truth.append(value in row)
    assert np.array_equal(NumCpp.contains(
        cArray, value, NumCpp.Axis.ROW).getNumpyArray().flatten(), np.asarray(truth))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    value = np.random.randint(0, 100, [1, ]).item(
    ) + 1j * np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    truth = list()
    for row in data.T:
        truth.append(value in row)
    assert np.array_equal(NumCpp.contains(
        cArray, value, NumCpp.Axis.ROW).getNumpyArray().flatten(), np.asarray(truth))


####################################################################################
def test_copy():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.copy(cArray), data)


####################################################################################
def test_copysign():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.copysign(
        cArray1, cArray2), np.copysign(data1, data2))


####################################################################################
def test_copyto():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray()
    data1 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    assert np.array_equal(NumCpp.copyto(cArray2, cArray1), data1)


####################################################################################
def test_corrcoef():
    shape = np.random.randint(10, 50, [2, ])
    a = np.random.randint(0, 100, shape)
    assert np.array_equal(np.round(NumCpp.corrcoef(a), 9),
                          np.round(np.corrcoef(a), 9))


####################################################################################
def test_cos():
    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.cosScalar(value), 9) == np.round(np.cos(value), 9)

    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.cosScalar(value), 9) == np.round(np.cos(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.cosArray(
        cArray), 9), np.round(np.cos(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.cosArray(
        cArray), 9), np.round(np.cos(data), 9))


####################################################################################
def test_cosh():
    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.coshScalar(value), 9) == np.round(np.cosh(value), 9)

    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.coshScalar(value), 9) == np.round(np.cosh(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.coshArray(
        cArray), 9), np.round(np.cosh(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.coshArray(
        cArray), 9), np.round(np.cosh(data), 9))


####################################################################################
def test_count_nonzero():
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 3, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert NumCpp.count_nonzero(
        cArray, NumCpp.Axis.NONE) == np.count_nonzero(data)

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 3, [shape.rows, shape.cols])
    imag = np.random.randint(1, 3, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.count_nonzero(
        cArray, NumCpp.Axis.NONE) == np.count_nonzero(data)

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 3, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.count_nonzero(
        cArray, NumCpp.Axis.ROW).flatten(), np.count_nonzero(data, axis=0))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 3, [shape.rows, shape.cols])
    imag = np.random.randint(1, 3, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.count_nonzero(
        cArray, NumCpp.Axis.ROW).flatten(), np.count_nonzero(data, axis=0))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 3, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.count_nonzero(
        cArray, NumCpp.Axis.COL).flatten(), np.count_nonzero(data, axis=1))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 3, [shape.rows, shape.cols])
    imag = np.random.randint(1, 3, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.count_nonzero(
        cArray, NumCpp.Axis.COL).flatten(), np.count_nonzero(data, axis=1))


####################################################################################
def test_cov():
    shape = np.random.randint(10, 50, [2, ])
    a = np.random.randint(0, 100, shape)
    assert np.array_equal(np.round(NumCpp.cov(a, False), 9),
                          np.round(np.cov(a, bias=False), 9))
    assert np.array_equal(np.round(NumCpp.cov(a, True), 9),
                          np.round(np.cov(a, bias=True), 9))


####################################################################################
def test_cov_inv():
    a = np.random.randint(0, 100, [5, 8])
    assert np.array_equal(np.round(NumCpp.cov_inv(a, False), 9),
                          np.round(np.linalg.inv(np.cov(a, bias=False)), 9))
    assert np.array_equal(np.round(NumCpp.cov_inv(a, True), 9),
                          np.round(np.linalg.inv(np.cov(a, bias=True)), 9))


####################################################################################
def test_cross():
    shape = NumCpp.Shape(1, 2)
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(float)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(float)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert NumCpp.cross(cArray1, cArray2, NumCpp.Axis.NONE).item(
    ) == np.cross(data1, data2).item()

    shape = NumCpp.Shape(1, 2)
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert NumCpp.cross(cArray1, cArray2, NumCpp.Axis.NONE).item(
    ) == np.cross(data1, data2).item()

    shape = NumCpp.Shape(2, np.random.randint(1, 100, [1, ]).item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(float)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(float)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.cross(cArray1, cArray2, NumCpp.Axis.ROW).getNumpyArray().flatten(),
                          np.cross(data1, data2, axis=0))

    shape = NumCpp.Shape(2, np.random.randint(1, 100, [1, ]).item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.cross(cArray1, cArray2, NumCpp.Axis.ROW).getNumpyArray().flatten(),
                          np.cross(data1, data2, axis=0))

    shape = NumCpp.Shape(np.random.randint(1, 100, [1, ]).item(), 2)
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(float)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(float)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.cross(cArray1, cArray2, NumCpp.Axis.COL).getNumpyArray().flatten(),
                          np.cross(data1, data2, axis=1))

    shape = NumCpp.Shape(np.random.randint(1, 100, [1, ]).item(), 2)
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.cross(cArray1, cArray2, NumCpp.Axis.COL).getNumpyArray().flatten(),
                          np.cross(data1, data2, axis=1))

    shape = NumCpp.Shape(1, 3)
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(float)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(float)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.cross(cArray1, cArray2, NumCpp.Axis.NONE).getNumpyArray().flatten(),
                          np.cross(data1, data2).flatten())

    shape = NumCpp.Shape(1, 3)
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.cross(cArray1, cArray2, NumCpp.Axis.NONE).getNumpyArray().flatten(),
                          np.cross(data1, data2).flatten())

    shape = NumCpp.Shape(3, np.random.randint(1, 100, [1, ]).item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(float)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(float)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.cross(cArray1, cArray2, NumCpp.Axis.ROW).getNumpyArray(),
                          np.cross(data1, data2, axis=0))

    shape = NumCpp.Shape(3, np.random.randint(1, 100, [1, ]).item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.cross(cArray1, cArray2, NumCpp.Axis.ROW).getNumpyArray(),
                          np.cross(data1, data2, axis=0))

    shape = NumCpp.Shape(np.random.randint(1, 100, [1, ]).item(), 3)
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(float)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(float)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.cross(cArray1, cArray2, NumCpp.Axis.COL).getNumpyArray(),
                          np.cross(data1, data2, axis=1))

    shape = NumCpp.Shape(np.random.randint(1, 100, [1, ]).item(), 3)
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.cross(cArray1, cArray2, NumCpp.Axis.COL).getNumpyArray(),
                          np.cross(data1, data2, axis=1))


####################################################################################
def test_cube():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.cube(cArray), 9),
                          np.round(data * data * data, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.cube(cArray), 9),
                          np.round(data * data * data, 9))


####################################################################################
def test_cumprod():
    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumprod(
        cArray, NumCpp.Axis.NONE).flatten(), data.cumprod())

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 4, [shape.rows, shape.cols])
    imag = np.random.randint(1, 4, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumprod(
        cArray, NumCpp.Axis.NONE).flatten(), data.cumprod())

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumprod(
        cArray, NumCpp.Axis.ROW), data.cumprod(axis=0))

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 4, [shape.rows, shape.cols])
    imag = np.random.randint(1, 4, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumprod(
        cArray, NumCpp.Axis.ROW), data.cumprod(axis=0))

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumprod(
        cArray, NumCpp.Axis.COL), data.cumprod(axis=1))

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 4, [shape.rows, shape.cols])
    imag = np.random.randint(1, 4, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumprod(
        cArray, NumCpp.Axis.COL), data.cumprod(axis=1))


####################################################################################
def test_cumsum():
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumsum(
        cArray, NumCpp.Axis.NONE).flatten(), data.cumsum())

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag = np.random.randint(1, 50, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumsum(
        cArray, NumCpp.Axis.NONE).flatten(), data.cumsum())

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumsum(
        cArray, NumCpp.Axis.ROW), data.cumsum(axis=0))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag = np.random.randint(1, 50, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumsum(
        cArray, NumCpp.Axis.ROW), data.cumsum(axis=0))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumsum(
        cArray, NumCpp.Axis.COL), data.cumsum(axis=1))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag = np.random.randint(1, 50, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumsum(
        cArray, NumCpp.Axis.COL), data.cumsum(axis=1))


####################################################################################
def test_deg2rad():
    value = np.abs(np.random.rand(1).item()) * 360
    assert np.round(NumCpp.deg2radScalar(value),
                    9) == np.round(np.deg2rad(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 360
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.deg2radArray(
        cArray), 9), np.round(np.deg2rad(data), 9))


####################################################################################
def test_degrees():
    value = np.abs(np.random.rand(1).item()) * 2 * np.pi
    assert np.round(NumCpp.degreesScalar(value),
                    9) == np.round(np.degrees(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 2 * np.pi
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.degreesArray(
        cArray), 9), np.round(np.degrees(data), 9))


####################################################################################
def test_deleteIndices():
    shapeInput = np.asarray([100, 100])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    index = np.random.randint(0, shape.size(), [1, ]).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.deleteIndicesScalar(cArray, index, NumCpp.Axis.NONE).flatten(),
                          np.delete(data, index, axis=None))

    shapeInput = np.asarray([100, 100])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    index = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.deleteIndicesScalar(
        cArray, index, NumCpp.Axis.ROW), np.delete(data, index, axis=0))

    shapeInput = np.asarray([100, 100])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    index = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.deleteIndicesScalar(
        cArray, index, NumCpp.Axis.COL), np.delete(data, index, axis=1))

    shapeInput = np.asarray([100, 100])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    indices = NumCpp.Slice(0, 100, 4)
    indicesPy = slice(0, 99, 4)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.deleteIndicesSlice(cArray, indices, NumCpp.Axis.NONE).flatten(),
                          np.delete(data, indicesPy, axis=None))

    shapeInput = np.asarray([100, 100])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    indices = NumCpp.Slice(0, 100, 4)
    indicesPy = slice(0, 99, 4)
    assert np.array_equal(NumCpp.deleteIndicesSlice(cArray, indices, NumCpp.Axis.ROW),
                          np.delete(data, indicesPy, axis=0))

    shapeInput = np.asarray([100, 100])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    indices = NumCpp.Slice(0, 100, 4)
    indicesPy = slice(0, 99, 4)
    assert np.array_equal(NumCpp.deleteIndicesSlice(cArray, indices, NumCpp.Axis.COL),
                          np.delete(data, indicesPy, axis=1))

    shapeInput = np.asarray([100, 100])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    indicesPy = np.arange(0, 100, 4).astype(np.int32)
    cIndices = NumCpp.NdArrayInt32(1, indicesPy.size)
    cIndices.setArray(indicesPy)
    assert np.array_equal(NumCpp.deleteIndicesIndices(cArray, cIndices, NumCpp.Axis.NONE).flatten(),
                          np.delete(data, indicesPy, axis=None))

    shapeInput = np.asarray([100, 100])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    indicesPy = np.arange(0, 100, 4).astype(np.int32)
    cIndices = NumCpp.NdArrayInt32(1, indicesPy.size)
    cIndices.setArray(indicesPy)
    assert np.array_equal(NumCpp.deleteIndicesIndices(cArray, cIndices, NumCpp.Axis.ROW),
                          np.delete(data, indicesPy, axis=0))

    shapeInput = np.asarray([100, 100])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    indicesPy = np.arange(0, 100, 4).astype(np.int32)
    cIndices = NumCpp.NdArrayInt32(1, indicesPy.size)
    cIndices.setArray(indicesPy)
    assert np.array_equal(NumCpp.deleteIndicesIndices(cArray, cIndices, NumCpp.Axis.COL),
                          np.delete(data, indicesPy, axis=1))


####################################################################################
def test_diag():
    shapeInput = np.random.randint(2, 25, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    k = np.random.randint(0, np.min(shapeInput), [1, ]).item()
    elements = np.random.randint(1, 100, shapeInput)
    cElements = NumCpp.NdArray(shape)
    cElements.setArray(elements)
    assert np.array_equal(NumCpp.diag(
        cElements, k).flatten(), np.diag(elements, k))

    shapeInput = np.random.randint(2, 25, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    k = np.random.randint(0, np.min(shapeInput), [1, ]).item()
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    elements = real + 1j * imag
    cElements = NumCpp.NdArrayComplexDouble(shape)
    cElements.setArray(elements)
    assert np.array_equal(NumCpp.diag(
        cElements, k).flatten(), np.diag(elements, k))


####################################################################################
def test_diagflat():
    numElements = np.random.randint(2, 25, [1, ]).item()
    shape = NumCpp.Shape(1, numElements)
    k = np.random.randint(0, 10, [1, ]).item()
    elements = np.random.randint(1, 100, [numElements, ])
    cElements = NumCpp.NdArray(shape)
    cElements.setArray(elements)
    assert np.array_equal(NumCpp.diagflat(cElements, k),
                          np.diagflat(elements, k))

    numElements = np.random.randint(2, 25, [1, ]).item()
    shape = NumCpp.Shape(1, numElements)
    k = np.random.randint(0, 10, [1, ]).item()
    real = np.random.randint(1, 100, [numElements, ])
    imag = np.random.randint(1, 100, [numElements, ])
    elements = real + 1j * imag
    cElements = NumCpp.NdArrayComplexDouble(shape)
    cElements.setArray(elements)
    assert np.array_equal(NumCpp.diagflat(cElements, k),
                          np.diagflat(elements, k))

    numElements = np.random.randint(1, 25, [1, ]).item()
    shape = NumCpp.Shape(1, numElements)
    k = np.random.randint(0, 10, [1, ]).item()
    elements = np.random.randint(1, 100, [numElements, ])
    cElements = NumCpp.NdArray(shape)
    cElements.setArray(elements)
    assert np.array_equal(NumCpp.diagflat(cElements, k),
                          np.diagflat(elements, k))

    numElements = np.random.randint(1, 25, [1, ]).item()
    shape = NumCpp.Shape(1, numElements)
    k = np.random.randint(0, 10, [1, ]).item()
    real = np.random.randint(1, 100, [numElements, ])
    imag = np.random.randint(1, 100, [numElements, ])
    elements = real + 1j * imag
    cElements = NumCpp.NdArrayComplexDouble(shape)
    cElements.setArray(elements)
    assert np.array_equal(NumCpp.diagflat(cElements, k),
                          np.diagflat(elements, k))


####################################################################################
def test_diagonal():
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols])
    cArray.setArray(data)
    offset = np.random.randint(0, min(shape.rows, shape.cols), [1, ]).item()
    assert np.array_equal(NumCpp.diagonal(cArray, offset, NumCpp.Axis.ROW).flatten(),
                          np.diagonal(data, offset, axis1=0, axis2=1))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag = np.random.randint(1, 50, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    offset = np.random.randint(0, min(shape.rows, shape.cols), [1, ]).item()
    assert np.array_equal(NumCpp.diagonal(cArray, offset, NumCpp.Axis.ROW).flatten(),
                          np.diagonal(data, offset, axis1=0, axis2=1))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols])
    cArray.setArray(data)
    offset = np.random.randint(0, min(shape.rows, shape.cols), [1, ]).item()
    assert np.array_equal(NumCpp.diagonal(cArray, offset, NumCpp.Axis.COL).flatten(),
                          np.diagonal(data, offset, axis1=1, axis2=0))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag = np.random.randint(1, 50, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    offset = np.random.randint(0, min(shape.rows, shape.cols), [1, ]).item()
    assert np.array_equal(NumCpp.diagonal(cArray, offset, NumCpp.Axis.COL).flatten(),
                          np.diagonal(data, offset, axis1=1, axis2=0))


####################################################################################
def test_diff():
    shapeInput = np.random.randint(10, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.diff(cArray, NumCpp.Axis.NONE).flatten(),
                          np.diff(data.flatten()))

    shapeInput = np.random.randint(10, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag = np.random.randint(1, 50, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.diff(cArray, NumCpp.Axis.NONE).flatten(),
                          np.diff(data.flatten()))

    shapeInput = np.random.randint(10, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.diff(
        cArray, NumCpp.Axis.ROW), np.diff(data, axis=0))

    shapeInput = np.random.randint(10, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag = np.random.randint(1, 50, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.diff(
        cArray, NumCpp.Axis.ROW), np.diff(data, axis=0))

    shapeInput = np.random.randint(10, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.diff(cArray, NumCpp.Axis.COL).astype(
        np.uint32), np.diff(data, axis=1))

    shapeInput = np.random.randint(10, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag = np.random.randint(1, 50, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.diff(
        cArray, NumCpp.Axis.COL), np.diff(data, axis=1))


####################################################################################
def test_digitize():
    x = np.random.rand(1000) * 12 - 1
    cX = NumCpp.NdArray(1, x.size)
    cX.setArray(x)
    bins = np.linspace(0, 10, 10)
    cBins = NumCpp.NdArray(1, bins.size)
    cBins.setArray(bins)
    assert np.array_equal(NumCpp.digitize(cX, cBins).flatten(), np.digitize(x, bins))


####################################################################################
def test_divide():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    data2[data2 == 0] = 1
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(np.round(NumCpp.divide(cArray1, cArray2), 9),
                          np.round(data1 / data2, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    value = 0
    while value == 0:
        value = np.random.randint(-100, 100)
    assert np.array_equal(np.round(NumCpp.divide(cArray, value), 9),
                          np.round(data / value, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    data[data == 0] = 1
    cArray.setArray(data)
    value = np.random.randint(-100, 100)
    assert np.array_equal(np.round(NumCpp.divide(value, cArray), 9),
                          np.round(value / data, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    data2[data2 == complex(0)] = complex(1)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(np.round(NumCpp.divide(cArray1, cArray2), 9),
                          np.round(data1 / data2, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    value = 0
    while value == complex(0):
        value = np.random.randint(-100, 100) + 1j * \
            np.random.randint(-100, 100)
    assert np.array_equal(np.round(NumCpp.divide(cArray, value), 9),
                          np.round(data / value, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    data[data == complex(0)] = complex(1)
    cArray.setArray(data)
    value = np.random.randint(-100, 100) + 1j * np.random.randint(-100, 100)
    assert np.array_equal(np.round(NumCpp.divide(value, cArray), 9),
                          np.round(value / data, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArray(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2[data2 == 0] = 1
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(np.round(NumCpp.divide(cArray1, cArray2), 9),
                          np.round(data1 / data2, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    data2[data2 == complex(0)] = complex(1)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(np.round(NumCpp.divide(cArray1, cArray2), 9),
                          np.round(data1 / data2, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    while value == complex(0):
        value = np.random.randint(-100, 100) + 1j * \
            np.random.randint(-100, 100)
    assert np.array_equal(np.round(NumCpp.divide(cArray, value), 9),
                          np.round(data / value, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    data[data == 0] = 1
    cArray.setArray(data)
    value = np.random.randint(-100, 100) + 1j * np.random.randint(-100, 100)
    assert np.array_equal(np.round(NumCpp.divide(value, cArray), 9),
                          np.round(value / data, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    value = 0
    while value == 0:
        value = np.random.randint(-100, 100)
    assert np.array_equal(np.round(NumCpp.divide(cArray, value), 9),
                          np.round(data / value, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    data[data == complex(0)] = complex(1)
    cArray.setArray(data)
    value = np.random.randint(-100, 100)
    assert np.array_equal(np.round(NumCpp.divide(value, cArray), 9),
                          np.round(value / data, 9))


####################################################################################
def test_dot():
    size = np.random.randint(1, 100, [1, ]).item()
    shape = NumCpp.Shape(1, size)
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 50, [shape.rows, shape.cols])
    data2 = np.random.randint(1, 50, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert NumCpp.dot(cArray1, cArray2).item() == np.dot(data1, data2.T).item()

    size = np.random.randint(1, 100, [1, ]).item()
    shape = NumCpp.Shape(1, size)
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    data1 = np.random.randint(1, 50, [shape.rows, shape.cols])
    real2 = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 50, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert NumCpp.dot(cArray1, cArray2).item() == np.dot(data1, data2.T).item()

    size = np.random.randint(1, 100, [1, ]).item()
    shape = NumCpp.Shape(1, size)
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 50, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 50, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert NumCpp.dot(cArray1, cArray2).item() == np.dot(data1, data2.T).item()

    size = np.random.randint(1, 100, [1, ]).item()
    shape = NumCpp.Shape(1, size)
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArray(shape)
    real1 = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 50, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    data2 = np.random.randint(1, 50, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert NumCpp.dot(cArray1, cArray2).item() == np.dot(data1, data2.T).item()

    shapeInput = np.random.randint(2, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[1].item(
    ), np.random.randint(1, 100, [1, ]).item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    data1 = np.random.randint(1, 50, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 50, [shape2.rows, shape2.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.dot(cArray1, cArray2), np.dot(data1, data2))

    shapeInput = np.random.randint(2, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[1].item(
    ), np.random.randint(1, 100, [1, ]).item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape1)
    cArray2 = NumCpp.NdArrayComplexDouble(shape2)
    real1 = np.random.randint(1, 50, [shape1.rows, shape1.cols])
    imag1 = np.random.randint(1, 50, [shape1.rows, shape1.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 50, [shape2.rows, shape2.cols])
    imag2 = np.random.randint(1, 50, [shape2.rows, shape2.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.dot(cArray1, cArray2), np.dot(data1, data2))


####################################################################################
def test_empty():
    shapeInput = np.random.randint(2, 100, [2, ])
    cArray = NumCpp.emptyRowCol(shapeInput[0].item(), shapeInput[1].item())
    assert cArray.shape[0] == shapeInput[0]
    assert cArray.shape[1] == shapeInput[1]
    assert cArray.size == shapeInput.prod()

    shapeInput = np.random.randint(2, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.emptyShape(shape)
    assert cArray.shape[0] == shape.rows
    assert cArray.shape[1] == shape.cols
    assert cArray.size == shapeInput.prod()

    shapeInput = np.random.randint(2, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.empty_like(cArray1)
    assert cArray2.shape().rows == shape.rows
    assert cArray2.shape().cols == shape.cols
    assert cArray2.size() == shapeInput.prod()


####################################################################################
def test_endianess():
    shapeInput = np.random.randint(2, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    assert NumCpp.endianess(cArray) == NumCpp.Endian.NATIVE


####################################################################################
def test_equal():
    shapeInput = np.random.randint(2, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 10, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 10, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.equal(cArray1, cArray2),
                          np.equal(data1, data2))

    shapeInput = np.random.randint(2, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.equal(cArray1, cArray2),
                          np.equal(data1, data2))


####################################################################################
def test_extract():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    mask = np.random.randint(0, 2, [shape.rows, shape.cols]).astype(bool)
    assert np.array_equal(NumCpp.extract(mask, data).flatten(),
                          np.extract(mask, data))


####################################################################################
def test_exp2():
    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.expScalar(value), 9) == np.round(np.exp(value), 9)

    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.expScalar(value), 9) == np.round(np.exp(value), 9)

    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.exp2Scalar(value), 9) == np.round(np.exp2(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.exp2Array(
        cArray), 9), np.round(np.exp2(data), 9))


####################################################################################
def test_exp():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.expArray(
        cArray), 9), np.round(np.exp(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.expArray(
        cArray), 9), np.round(np.exp(data), 9))

    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.expm1Scalar(value),
                    9) == np.round(np.expm1(value), 9)

    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.expm1Scalar(value),
                    9) == np.round(np.expm1(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.expm1Array(
        cArray), 9), np.round(np.expm1(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.rand(shape.rows, shape.cols)
    imag = np.random.rand(shape.rows, shape.cols)
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.expm1Array(
        cArray), 9), np.round(np.expm1(data), 9))


####################################################################################
def test_eye():
    shapeInput = np.random.randint(1, 100, [1, ]).item()
    randK = np.random.randint(0, shapeInput, [1, ]).item()
    assert np.array_equal(NumCpp.eye1D(shapeInput, randK),
                          np.eye(shapeInput, k=randK))

    shapeInput = np.random.randint(1, 100, [1, ]).item()
    randK = np.random.randint(0, shapeInput, [1, ]).item()
    assert np.array_equal(NumCpp.eye1DComplex(shapeInput, randK),
                          np.eye(shapeInput, k=randK) + 1j * np.zeros([shapeInput, shapeInput]))

    shapeInput = np.random.randint(10, 100, [2, ])
    randK = np.random.randint(0, np.min(shapeInput), [1, ]).item()
    assert np.array_equal(NumCpp.eye2D(shapeInput[0].item(), shapeInput[1].item(), randK),
                          np.eye(shapeInput[0].item(), shapeInput[1].item(), k=randK))

    shapeInput = np.random.randint(10, 100, [2, ])
    randK = np.random.randint(0, np.min(shapeInput), [1, ]).item()
    assert np.array_equal(NumCpp.eye2DComplex(shapeInput[0].item(), shapeInput[1].item(), randK),
                          np.eye(shapeInput[0].item(), shapeInput[1].item(), k=randK) +
                          1j * np.zeros(shapeInput))

    shapeInput = np.random.randint(10, 100, [2, ])
    cShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    randK = np.random.randint(0, np.min(shapeInput), [1, ]).item()
    assert np.array_equal(NumCpp.eyeShape(cShape, randK), np.eye(
        shapeInput[0].item(), shapeInput[1].item(), k=randK))

    shapeInput = np.random.randint(10, 100, [2, ])
    cShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    randK = np.random.randint(0, np.min(shapeInput), [1, ]).item()
    assert np.array_equal(NumCpp.eyeShapeComplex(cShape, randK),
                          np.eye(shapeInput[0].item(), shapeInput[1].item(), k=randK) +
                          1j * np.zeros(shapeInput))


####################################################################################
def test_fill_diagonal():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    NumCpp.fillDiagonal(cArray, 666)
    np.fill_diagonal(data, 666)
    assert np.array_equal(cArray.getNumpyArray(), data)


####################################################################################
def test_find():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    value = data.mean()
    cMask = NumCpp.operatorGreater(cArray, value)
    cMaskArray = NumCpp.NdArrayBool(cMask.shape[0], cMask.shape[1])
    cMaskArray.setArray(cMask)
    idxs = NumCpp.find(cMaskArray).astype(np.int64)
    idxsPy = np.nonzero((data > value).flatten())[0]
    assert np.array_equal(idxs.flatten(), idxsPy)


####################################################################################
def test_findN():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    value = data.mean()
    cMask = NumCpp.operatorGreater(cArray, value)
    cMaskArray = NumCpp.NdArrayBool(cMask.shape[0], cMask.shape[1])
    cMaskArray.setArray(cMask)
    idxs = NumCpp.findN(cMaskArray, 8).astype(np.int64)
    idxsPy = np.nonzero((data > value).flatten())[0]
    assert np.array_equal(idxs.flatten(), idxsPy[:8])


####################################################################################
def test_fix():
    value = np.random.randn(1).item() * 100
    assert NumCpp.fixScalar(value) == np.fix(value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    assert np.array_equal(NumCpp.fixArray(cArray), np.fix(data))


####################################################################################
def test_flatten():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.flatten(
        cArray).getNumpyArray(), np.resize(data, [1, data.size]))


####################################################################################
def test_flatnonzero():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.flatnonzero(
        cArray).getNumpyArray().flatten(), np.flatnonzero(data))


####################################################################################
def test_flip():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.flip(cArray, NumCpp.Axis.NONE).getNumpyArray(),
                          np.flip(data.reshape(1, data.size), axis=1).reshape(shapeInput))


####################################################################################
def test_fliplr():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.fliplr(
        cArray).getNumpyArray(), np.fliplr(data))


####################################################################################
def test_flipud():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.flipud(
        cArray).getNumpyArray(), np.flipud(data))


####################################################################################
def test_floor():
    value = np.random.randn(1).item() * 100
    assert NumCpp.floorScalar(value) == np.floor(value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    assert np.array_equal(NumCpp.floorArray(cArray), np.floor(data))


####################################################################################
def test_floor_divide():
    value1 = np.random.randn(1).item() * 100 + 1000
    value2 = np.random.randn(1).item() * 100 + 1000
    assert NumCpp.floor_divideScalar(
        value1, value2) == np.floor_divide(value1, value2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data2 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.floor_divideArray(
        cArray1, cArray2), np.floor_divide(data1, data2))


####################################################################################
def test_fmax():
    value1 = np.random.randn(1).item() * 100 + 1000
    value2 = np.random.randn(1).item() * 100 + 1000
    assert NumCpp.fmaxScalar(value1, value2) == np.fmax(value1, value2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data2 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.fmaxArray(
        cArray1, cArray2), np.fmax(data1, data2))


####################################################################################
def test_fmin():
    value1 = np.random.randn(1).item() * 100 + 1000
    value2 = np.random.randn(1).item() * 100 + 1000
    assert NumCpp.fminScalar(value1, value2) == np.fmin(value1, value2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data2 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.fminArray(
        cArray1, cArray2), np.fmin(data1, data2))


####################################################################################
def test_fmod():
    value1 = np.random.randint(1, 100, [1, ]).item() * 100 + 1000
    value2 = np.random.randint(1, 100, [1, ]).item() * 100 + 1000
    assert NumCpp.fmodScalarInt(value1, value2) == np.fmod(value1, value2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt32(shape)
    cArray2 = NumCpp.NdArrayUInt32(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols], dtype=np.uint32) * 100 + 1000
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols], dtype=np.uint32) * 100 + 1000
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.fmodArrayInt(
        cArray1, cArray2), np.fmod(data1, data2))

    value1 = np.random.randint(1, 100, [1, ]).item() * 100 + 1000.5
    value2 = np.random.randint(1, 100, [1, ]).item() * 100 + 1000.5
    assert NumCpp.fmodScalarFloat(value1, value2) == np.fmod(value1, value2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols], dtype=np.uint32).astype(float) * 100 + 1000.5
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols], dtype=np.uint32).astype(float) * 100 + 1000.5
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.fmodArrayFloat(
        cArray1, cArray2), np.fmod(data1, data2))


####################################################################################
def test_fromfile():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    tempDir = r'C:\Temp'
    if not os.path.exists(tempDir):
        os.mkdir(tempDir)
    tempFile = os.path.join(tempDir, 'NdArrayDump.bin')
    NumCpp.tofile(cArray, tempFile)
    assert os.path.isfile(tempFile)
    data2 = NumCpp.fromfile(tempFile).reshape(shapeInput)
    assert np.array_equal(data, data2)
    os.remove(tempFile)

    # delimiter = ' '
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    tempDir = tempfile.gettempdir()
    tempFile = os.path.join(tempDir, 'NdArrayDump')
    NumCpp.tofile(cArray, tempFile, ' ')
    assert os.path.exists(tempFile + '.txt')
    data2 = NumCpp.fromfile(tempFile + '.txt', ' ').reshape(shapeInput)
    assert np.array_equal(data, data2)
    os.remove(tempFile + '.txt')

    # delimiter = '\n'
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    tempDir = tempfile.gettempdir()
    tempFile = os.path.join(tempDir, 'NdArrayDump')
    NumCpp.tofile(cArray, tempFile, '\n')
    assert os.path.exists(tempFile + '.txt')
    data2 = NumCpp.fromfile(tempFile + '.txt', '\n').reshape(shapeInput)
    assert np.array_equal(data, data2)
    os.remove(tempFile + '.txt')

    # delimiter = '\t'
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    tempDir = tempfile.gettempdir()
    tempFile = os.path.join(tempDir, 'NdArrayDump')
    NumCpp.tofile(cArray, tempFile, '\t')
    assert os.path.exists(tempFile + '.txt')
    data2 = NumCpp.fromfile(tempFile + '.txt', '\t').reshape(shapeInput)
    assert np.array_equal(data, data2)
    os.remove(tempFile + '.txt')

    # delimiter = ','
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    tempDir = tempfile.gettempdir()
    tempFile = os.path.join(tempDir, 'NdArrayDump')
    NumCpp.tofile(cArray, tempFile, ',')
    assert os.path.exists(tempFile + '.txt')
    data2 = NumCpp.fromfile(tempFile + '.txt', ',').reshape(shapeInput)
    assert np.array_equal(data, data2)
    os.remove(tempFile + '.txt')

    # delimiter = '|'
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    tempDir = tempfile.gettempdir()
    tempFile = os.path.join(tempDir, 'NdArrayDump')
    NumCpp.tofile(cArray, tempFile, '|')
    assert os.path.exists(tempFile + '.txt')
    data2 = NumCpp.fromfile(tempFile + '.txt', '|').reshape(shapeInput)
    assert np.array_equal(data, data2)
    os.remove(tempFile + '.txt')


####################################################################################
def test_fromfunction():
    def functionSize(idx: int) -> float:
        return idx / 10

    size = np.random.randint(100)
    assert np.array_equal(np.round(NumCpp.fromfunction(size).flatten(), 8),
                          np.round(np.fromfunction(functionSize, [size, ]), 8))

    def functionShape(row: int, col: int) -> float:
        return (row * col) / 10

    shape = np.random.randint(100, size=[2, ])
    cShape = NumCpp.Shape(*shape)
    assert np.array_equal(np.round(NumCpp.fromfunction(cShape), 8),
                          np.round(np.fromfunction(functionShape, shape), 8))


####################################################################################
def test_fromiter():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.fromiter(cArray).flatten(), data.flatten())

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.fromiter(cArray).flatten(), data.flatten())


####################################################################################
def test_fromstring():
    seperator = ' '
    shape = np.random.randint(20, 100, [2, ])
    data = np.random.randint(1, 50, shape).astype(int)
    dataStr = seperator.join(str(x) for x in data.flatten())
    assert np.array_equal(NumCpp.fromstringInt(dataStr, seperator).reshape(shape), data)

    shape = np.random.randint(20, 100, [2, ])
    data = np.random.rand(*shape)
    dataStr = seperator.join(str(x) for x in data.flatten())
    assert np.array_equal(NumCpp.fromstringDouble(dataStr, seperator).reshape(shape), data)

    seperator = '\n'
    shape = np.random.randint(20, 100, [2, ])
    data = np.random.randint(1, 50, shape).astype(int)
    dataStr = seperator.join(str(x) for x in data.flatten())
    assert np.array_equal(NumCpp.fromstringInt(dataStr, seperator).reshape(shape), data)

    shape = np.random.randint(20, 100, [2, ])
    data = np.random.rand(*shape)
    dataStr = seperator.join(str(x) for x in data.flatten())
    assert np.array_equal(NumCpp.fromstringDouble(dataStr, seperator).reshape(shape), data)

    seperator = '\t'
    shape = np.random.randint(20, 100, [2, ])
    data = np.random.randint(1, 50, shape).astype(int)
    dataStr = seperator.join(str(x) for x in data.flatten())
    assert np.array_equal(NumCpp.fromstringInt(dataStr, seperator).reshape(shape), data)

    shape = np.random.randint(20, 100, [2, ])
    data = np.random.rand(*shape)
    dataStr = seperator.join(str(x) for x in data.flatten())
    assert np.array_equal(NumCpp.fromstringDouble(dataStr, seperator).reshape(shape), data)

    seperator = ','
    shape = np.random.randint(20, 100, [2, ])
    data = np.random.randint(1, 50, shape).astype(int)
    dataStr = seperator.join(str(x) for x in data.flatten())
    assert np.array_equal(NumCpp.fromstringInt(dataStr, seperator).reshape(shape), data)

    shape = np.random.randint(20, 100, [2, ])
    data = np.random.rand(*shape)
    dataStr = seperator.join(str(x) for x in data.flatten())
    assert np.array_equal(NumCpp.fromstringDouble(dataStr, seperator).reshape(shape), data)

    seperator = '|'
    shape = np.random.randint(20, 100, [2, ])
    data = np.random.randint(1, 50, shape).astype(int)
    dataStr = seperator.join(str(x) for x in data.flatten())
    assert np.array_equal(NumCpp.fromstringInt(dataStr, seperator).reshape(shape), data)

    shape = np.random.randint(20, 100, [2, ])
    data = np.random.rand(*shape)
    dataStr = seperator.join(str(x) for x in data.flatten())
    assert np.array_equal(NumCpp.fromstringDouble(dataStr, seperator).reshape(shape), data)


####################################################################################
def test_full():
    shapeInput = np.random.randint(1, 100, [1, ]).item()
    value = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.fullSquare(shapeInput, value)
    assert (cArray.shape[0] == shapeInput and cArray.shape[1] == shapeInput and
            cArray.size == shapeInput**2 and np.all(cArray == value))

    shapeInput = np.random.randint(1, 100, [1, ]).item()
    value = np.random.randint(1, 100, [1, ]).item(
    ) + 1j * np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.fullSquareComplex(shapeInput, value)
    assert (cArray.shape[0] == shapeInput and cArray.shape[1] == shapeInput and
            cArray.size == shapeInput**2 and np.all(cArray == value))

    shapeInput = np.random.randint(2, 100, [2, ])
    value = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.fullRowCol(
        shapeInput[0].item(), shapeInput[1].item(), value)
    assert (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod() and np.all(cArray == value))

    shapeInput = np.random.randint(2, 100, [2, ])
    value = np.random.randint(1, 100, [1, ]).item(
    ) + 1j * np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.fullRowColComplex(
        shapeInput[0].item(), shapeInput[1].item(), value)
    assert (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod() and np.all(cArray == value))

    shapeInput = np.random.randint(2, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    value = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.fullShape(shape, value)
    assert (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod() and np.all(cArray == value))

    shapeInput = np.random.randint(2, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    value = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.fullShape(shape, value)
    assert (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod() and np.all(cArray == value))


####################################################################################
def test_full_like():
    shapeInput = np.random.randint(2, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    value = np.random.randint(1, 100, [1, ]).item()
    cArray2 = NumCpp.full_like(cArray1, value)
    assert (cArray2.shape().rows == shape.rows and cArray2.shape().cols == shape.cols and
            cArray2.size() == shapeInput.prod() and np.all(cArray2.getNumpyArray() == value))

    shapeInput = np.random.randint(2, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    value = np.random.randint(1, 100, [1, ]).item(
    ) + 1j * np.random.randint(1, 100, [1, ]).item()
    cArray2 = NumCpp.full_likeComplex(cArray1, value)
    assert (cArray2.shape().rows == shape.rows and cArray2.shape().cols == shape.cols and
            cArray2.size() == shapeInput.prod() and np.all(cArray2.getNumpyArray() == value))


####################################################################################
def test_gcd():
    if not NumCpp.NUMCPP_NO_USE_BOOST or NumCpp.STL_GCD_LCM:
        value1 = np.random.randint(1, 1000, [1, ]).item()
        value2 = np.random.randint(1, 1000, [1, ]).item()
        assert NumCpp.gcdScalar(value1, value2) == np.gcd(value1, value2)

    if not NumCpp.NUMCPP_NO_USE_BOOST:
        size = np.random.randint(20, 100, [1, ]).item()
        cArray = NumCpp.NdArrayUInt32(1, size)
        data = np.random.randint(1, 1000, [size, ], dtype=np.uint32)
        cArray.setArray(data)
        assert NumCpp.gcdArray(cArray) == np.gcd.reduce(data)  # noqa


####################################################################################
def test_geomspace():
    start = np.random.randint(1, 100)
    stop = np.random.randint(start + 1, 3 * start)
    num = np.random.randint(1, 100)
    assert np.array_equal(np.round(NumCpp.geomspace(start, stop, num, True).flatten(), 9),
                          np.round(np.geomspace(start=start, stop=stop, num=num, endpoint=True), 9))
    assert np.array_equal(np.round(NumCpp.geomspace(start, stop, num, False).flatten(), 9),
                          np.round(np.geomspace(start=start, stop=stop, num=num, endpoint=False), 9))


####################################################################################
def test_gradient():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 1000, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.gradient(
        cArray, NumCpp.Axis.ROW), np.gradient(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 1000, [shape.rows, shape.cols])
    imag = np.random.randint(1, 1000, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.gradient(
        cArray, NumCpp.Axis.ROW), np.gradient(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 1000, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.gradient(
        cArray, NumCpp.Axis.COL), np.gradient(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 1000, [shape.rows, shape.cols])
    imag = np.random.randint(1, 1000, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.gradient(
        cArray, NumCpp.Axis.COL), np.gradient(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 1000, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.gradient(cArray, NumCpp.Axis.NONE).flatten(),
                          np.gradient(data.flatten(), axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 1000, [shape.rows, shape.cols])
    imag = np.random.randint(1, 1000, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.gradient(cArray, NumCpp.Axis.NONE).flatten(),
                          np.gradient(data.flatten(), axis=0))


####################################################################################
def test_greater():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.greater(cArray1, cArray2).getNumpyArray(),
                          np.greater(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.greater(cArray1, cArray2).getNumpyArray(),
                          np.greater(data1, data2))


####################################################################################
def test_greater_equal():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.greater_equal(cArray1, cArray2).getNumpyArray(),
                          np.greater_equal(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.greater_equal(cArray1, cArray2).getNumpyArray(),
                          np.greater_equal(data1, data2))


####################################################################################
def test_hamming():
    m = np.random.randint(2, 100)
    assert np.array_equal(np.round(NumCpp.hamming(m), 9).flatten(),
                          np.round(np.hamming(m), 9))


####################################################################################
def test_hanning():
    m = np.random.randint(2, 100)
    assert np.array_equal(np.round(NumCpp.hanning(m), 9).flatten(),
                          np.round(np.hanning(m), 9))


####################################################################################
def test_histogram():
    shape = NumCpp.Shape(1024, 1024)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(1024, 1024) * np.random.randint(1,
                                                           10, [1, ]).item() + np.random.randint(1, 10, [1, ]).item()
    cArray.setArray(data)
    numBins = np.random.randint(10, 30, [1, ]).item()
    histogram, bins = NumCpp.histogram(cArray, numBins)
    h, b = np.histogram(data, numBins)
    assert np.array_equal(
        histogram.getNumpyArray().flatten().astype(np.int32), h)
    assert np.array_equal(
        np.round(bins.getNumpyArray().flatten(), 9), np.round(b, 9))

    shape = NumCpp.Shape(1024, 1024)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(1024, 1024) * np.random.randint(1,
                                                           10, [1, ]).item() + np.random.randint(1, 10, [1, ]).item()
    cArray.setArray(data)
    binEdges = np.linspace(data.min(), data.max(), 15, endpoint=True)
    cBinEdges = NumCpp.NdArray(1, binEdges.size)
    cBinEdges.setArray(binEdges)
    histogram = NumCpp.histogram(cArray, cBinEdges)
    h, _ = np.histogram(data, binEdges)
    assert np.array_equal(histogram.flatten().astype(np.int32), h)


def test_hsplit():
    shapeInput = np.random.randint(80, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    numSplits = np.random.randint(2, 5)
    indices = np.sort(np.random.choice(range(shape.cols), numSplits, replace=False))
    cIndices = NumCpp.NdArrayInt32(indices.size)
    cIndices.setArray(indices.astype(np.int32))
    result = np.hsplit(data, indices)
    cResult = NumCpp.hsplit(cArray, cIndices)
    assert len(cResult) == len(result)
    for i in range(len(result)):
        assert np.array_equal(cResult[i], result[i])


####################################################################################
def test_hstack():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item(
    ) + np.random.randint(1, 10, [1, ]).item())
    shape3 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item(
    ) + np.random.randint(1, 10, [1, ]).item())
    shape4 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item(
    ) + np.random.randint(1, 10, [1, ]).item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    cArray3 = NumCpp.NdArray(shape3)
    cArray4 = NumCpp.NdArray(shape4)
    data1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    data4 = np.random.randint(1, 100, [shape4.rows, shape4.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    assert np.array_equal(NumCpp.hstack(cArray1, cArray2, cArray3, cArray4),
                          np.hstack([data1, data2, data3, data4]))


####################################################################################
def test_hypot():
    value1 = np.random.randn(1).item() * 100 + 1000
    value2 = np.random.randn(1).item() * 100 + 1000
    assert NumCpp.hypotScalar(value1, value2) == np.hypot(value1, value2)

    value1 = np.random.randn(1).item() * 100 + 1000
    value2 = np.random.randn(1).item() * 100 + 1000
    value3 = np.random.randn(1).item() * 100 + 1000
    assert (np.round(NumCpp.hypotScalarTriple(value1, value2, value3), 9) ==
            np.round(np.sqrt(value1**2 + value2**2 + value3**2), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data2 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(np.round(NumCpp.hypotArray(cArray1, cArray2), 9),
                          np.round(np.hypot(data1, data2), 9))


####################################################################################
def test_identity():
    squareSize = np.random.randint(10, 100, [1, ]).item()
    assert np.array_equal(NumCpp.identity(
        squareSize).getNumpyArray(), np.identity(squareSize))

    squareSize = np.random.randint(10, 100, [1, ]).item()
    assert np.array_equal(NumCpp.identityComplex(squareSize).getNumpyArray(),
                          np.identity(squareSize) + 1j * np.zeros([squareSize, squareSize]))


####################################################################################
def test_imag():
    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.imagScalar(value), 9) == np.round(np.imag(value), 9)  # noqa

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.imagArray(
        cArray), 9), np.round(np.imag(data), 9))


####################################################################################
def test_inner():
    arraySize = np.random.randint(10, 100)
    a = np.random.randint(0, 100, [arraySize, ])
    b = np.random.randint(0, 100, [arraySize, ])
    assert NumCpp.inner(a, b) == np.inner(a, b)


####################################################################################
def test_insert():
    # insertIndexScalar
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = np.random.randint(0, data.size, 1).item()
    value = 666
    assert np.array_equal(NumCpp.insert(cArray, index, value).flatten(), np.insert(data, index, value))

    # negative index
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = -5
    value = 666
    assert np.array_equal(NumCpp.insert(cArray, index, value).flatten(), np.insert(data, index, value))

    # edge case index 0
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = 0
    value = 666
    assert np.array_equal(NumCpp.insert(cArray, index, value).flatten(), np.insert(data, index, value))

    # edge case index last
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = data.size
    value = 666
    assert np.array_equal(NumCpp.insert(cArray, index, value).flatten(), np.insert(data, index, value))

    # insertIndexArray
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = np.random.randint(0, data.size + 1, 1).item()
    values = np.random.randint(100, 200, np.random.randint(10, 50))
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, index, cValues).flatten(), np.insert(data, index, values))

    # negative index
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = -5
    values = np.random.randint(100, 200, np.random.randint(10, 50))
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, index, cValues).flatten(), np.insert(data, index, values))

    # edge case index 0
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = 0
    values = np.random.randint(100, 200, np.random.randint(10, 50))
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, index, cValues).flatten(), np.insert(data, index, values))

    # edge case index last
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = data.size
    values = np.random.randint(100, 200, np.random.randint(10, 50))
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, index, cValues).flatten(), np.insert(data, index, values))

    # insertIndexScalarAxis None
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = np.random.randint(0, data.size + 1, 1).item()
    value = 666
    assert np.array_equal(NumCpp.insert(cArray, index, value, NumCpp.Axis.NONE).flatten(),
                          np.insert(data, index, value, axis=None))

    # negative index
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = -5
    value = 666
    assert np.array_equal(NumCpp.insert(cArray, index, value, NumCpp.Axis.NONE).flatten(),
                          np.insert(data, index, value, axis=None))

    # edge case index 0
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = 0
    value = 666
    assert np.array_equal(NumCpp.insert(cArray, index, value, NumCpp.Axis.NONE).flatten(),
                          np.insert(data, index, value, axis=None))

    # edge case index last
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = data.size
    value = 666
    assert np.array_equal(NumCpp.insert(cArray, index, value, NumCpp.Axis.NONE).flatten(),
                          np.insert(data, index, value, axis=None))

    # insertIndexScalarAxis ROW
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = np.random.randint(1, shape.rows, 1).item()
    value = 666
    assert np.array_equal(NumCpp.insert(cArray, index, value, NumCpp.Axis.ROW),
                          np.insert(data, index, value, axis=0))

    # negative index
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = -5
    value = 666
    assert np.array_equal(NumCpp.insert(cArray, index, value, NumCpp.Axis.ROW),
                          np.insert(data, index, value, axis=0))

    # edge case index 0
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = 0
    value = 666
    assert np.array_equal(NumCpp.insert(cArray, index, value, NumCpp.Axis.ROW),
                          np.insert(data, index, value, axis=0))

    # edge case index last
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = shape.rows
    value = 666
    assert np.array_equal(NumCpp.insert(cArray, index, value, NumCpp.Axis.ROW),
                          np.insert(data, index, value, axis=0))

    # insertIndexScalarAxis COL
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = np.random.randint(1, shape.cols, 1).item()
    value = 666
    assert np.array_equal(NumCpp.insert(cArray, index, value, NumCpp.Axis.COL),
                          np.insert(data, index, value, axis=1))

    # negative index
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = -5
    value = 666
    assert np.array_equal(NumCpp.insert(cArray, index, value, NumCpp.Axis.COL),
                          np.insert(data, index, value, axis=1))

    # edge case index 0
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = 0
    value = 666
    assert np.array_equal(NumCpp.insert(cArray, index, value, NumCpp.Axis.COL),
                          np.insert(data, index, value, axis=1))

    # edge case index last
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = shape.cols
    value = 666
    assert np.array_equal(NumCpp.insert(cArray, index, value, NumCpp.Axis.COL),
                          np.insert(data, index, value, axis=1))

    # insertIndexArrayAxis None
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = np.random.randint(1, data.size, 1).item()
    values = np.random.randint(100, 200, np.random.randint(10, 50))
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, index, cValues, NumCpp.Axis.NONE).flatten(),
                          np.insert(data, index, values, axis=None))

    # negative index
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = -5
    values = np.random.randint(100, 200, np.random.randint(10, 50))
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, index, cValues, NumCpp.Axis.NONE).flatten(),
                          np.insert(data, index, values, axis=None))

    # edge case index 0
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = 0
    values = np.random.randint(100, 200, np.random.randint(10, 50))
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, index, cValues, NumCpp.Axis.NONE).flatten(),
                          np.insert(data, index, values, axis=None))

    # edge case index last
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = data.size
    values = np.random.randint(100, 200, np.random.randint(10, 50))
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, index, cValues, NumCpp.Axis.NONE).flatten(),
                          np.insert(data, index, values, axis=None))

    # insertIndexArrayAxis Row
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = np.random.randint(1, shape.rows, 1).item()
    values = np.random.randint(100, 200, [shape.cols, ])
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, index, cValues, NumCpp.Axis.ROW),
                          np.insert(data, index, values, axis=0))

    # negative index
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = -5
    values = np.random.randint(100, 200, [shape.cols, ])
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, index, cValues, NumCpp.Axis.ROW),
                          np.insert(data, index, values, axis=0))

    # edge case index 0
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = 0
    values = np.random.randint(100, 200, [shape.cols, ])
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, index, cValues, NumCpp.Axis.ROW),
                          np.insert(data, index, values, axis=0))

    # edge case index last
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = shape.rows
    values = np.random.randint(100, 200, [shape.cols, ])
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, index, cValues, NumCpp.Axis.ROW),
                          np.insert(data, index, values, axis=0))

    # insertIndexArrayAxis Col
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = np.random.randint(1, shape.cols, 1).item()
    values = np.random.randint(100, 200, [shape.rows, ])
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, index, cValues, NumCpp.Axis.COL),
                          np.insert(data, index, values, axis=1))

    # negative index
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = -5
    values = np.random.randint(100, 200, [shape.rows, ])
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, index, cValues, NumCpp.Axis.COL),
                          np.insert(data, index, values, axis=1))

    # edge case index 0
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = 0
    values = np.random.randint(100, 200, [shape.rows, ])
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, index, cValues, NumCpp.Axis.COL),
                          np.insert(data, index, values, axis=1))

    # edge case index last
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    index = shape.cols
    values = np.random.randint(100, 200, [shape.rows, ])
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, index, cValues, NumCpp.Axis.COL),
                          np.insert(data, index, values, axis=1))

    # insertIndicesScalar axis=NONE
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.unique(np.random.randint(1, shape.size(), [np.random.randint(2, 10, 1).item(), ])).astype(np.int32)
    cIndices = NumCpp.NdArrayInt32(1, indices.size)
    cIndices.setArray(indices)
    value = np.random.randint(100, 200, 1).item()
    assert np.array_equal(NumCpp.insert(cArray, cIndices, value, NumCpp.Axis.NONE).flatten(),
                          np.insert(data, indices, value, axis=None))

    # edge cases indices 0, last, and negative
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.array([-3, -1, 0, 2, data.size], dtype=np.int32)
    cIndices = NumCpp.NdArrayInt32(1, indices.size)
    cIndices.setArray(indices)
    value = np.random.randint(100, 200, 1).item()
    assert np.array_equal(NumCpp.insert(cArray, cIndices, value, NumCpp.Axis.NONE).flatten(),
                          np.insert(data, indices, value, axis=None))

    # insertIndicesScalar axis=ROW
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.unique(np.random.randint(1, shape.rows, [np.random.randint(2, 10, 1).item(), ]).astype(np.int32))
    cIndices = NumCpp.NdArrayInt32(1, indices.size)
    cIndices.setArray(indices)
    value = np.random.randint(100, 200, 1).item()
    assert np.array_equal(NumCpp.insert(cArray, cIndices, value, NumCpp.Axis.ROW),
                          np.insert(data, indices, value, axis=0))

    # edge cases indices 0, last, and negative
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.array([-3, -1, 0, 2, data.shape[0]], dtype=np.int32)
    cIndices = NumCpp.NdArrayInt32(1, indices.size)
    cIndices.setArray(indices)
    value = np.random.randint(100, 200, 1).item()
    assert np.array_equal(NumCpp.insert(cArray, cIndices, value, NumCpp.Axis.ROW),
                          np.insert(data, indices, value, axis=0))

    # insertIndicesScalar axis=COL
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.unique(np.random.randint(1, shape.cols, [np.random.randint(2, 10, 1).item(), ], dtype=np.int32))
    cIndices = NumCpp.NdArrayInt32(1, indices.size)
    cIndices.setArray(indices)
    value = np.random.randint(100, 200, 1).item()
    assert np.array_equal(NumCpp.insert(cArray, cIndices, value, NumCpp.Axis.COL),
                          np.insert(data, indices, value, axis=1))

    # edge cases indices 0, last, and negative
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.array([-3, -1, 0, 2, data.shape[1]], dtype=np.int32)
    cIndices = NumCpp.NdArrayInt32(1, indices.size)
    cIndices.setArray(indices)
    value = np.random.randint(100, 200, 1).item()
    assert np.array_equal(NumCpp.insert(cArray, cIndices, value, NumCpp.Axis.COL),
                          np.insert(data, indices, value, axis=1))

    # insertSliceScalar axis=NONE
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    sliceStart = np.random.randint(1, shape.size() / 10, 1).item()
    sliceEnd = np.random.randint(sliceStart + 1, shape.size(), 1).item()
    sliceStep = np.random.randint(2, 5, 1).item()
    cSlice = NumCpp.Slice(sliceStart, sliceEnd, sliceStep)
    value = np.random.randint(100, 200, 1).item()
    assert np.array_equal(NumCpp.insert(cArray, cSlice, value, NumCpp.Axis.NONE).flatten(),
                          np.insert(data, slice(sliceStart, sliceEnd, sliceStep), value, axis=None))

    # edge cases indices 0, last, and negative
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    sliceStart = 3
    sliceEnd = -3
    sliceStep = 1
    cSlice = NumCpp.Slice(sliceStart, sliceEnd, sliceStep)
    value = np.random.randint(100, 200, 1).item()
    assert np.array_equal(NumCpp.insert(cArray, cSlice, value, NumCpp.Axis.NONE).flatten(),
                          np.insert(data, slice(sliceStart, sliceEnd, sliceStep), value, axis=None))

    # insertSliceScalar axis=ROW
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.unique(np.random.randint(1, shape.rows, [np.random.randint(2, 10, 1).item(), ]))
    sliceStart = np.random.randint(1, shape.rows // 10, 1).item()
    sliceEnd = np.random.randint(sliceStart + 1, shape.rows, 1).item()
    sliceStep = np.random.randint(2, 5, 1).item()
    value = np.random.randint(100, 200, 1).item()
    assert np.array_equal(NumCpp.insert(cArray, NumCpp.Slice(sliceStart, sliceEnd, sliceStep), value, NumCpp.Axis.ROW),
                          np.insert(data, slice(sliceStart, sliceEnd, sliceStep), value, axis=0))

    # edge cases indices 0, last, and negative
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.array([-3, -1, 0, 2])
    sliceStart = np.random.randint(1, shape.rows // 10, 1).item()
    sliceEnd = np.random.randint(sliceStart + 1, shape.rows, 1).item()
    sliceStep = np.random.randint(2, 5, 1).item()
    value = np.random.randint(100, 200, 1).item()
    assert np.array_equal(NumCpp.insert(cArray, NumCpp.Slice(sliceStart, sliceEnd, sliceStep), value, NumCpp.Axis.ROW),
                          np.insert(data, slice(sliceStart, sliceEnd, sliceStep), value, axis=0))

    # insertSliceScalar axis=COL
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.unique(np.random.randint(1, shape.cols, [np.random.randint(2, 10, 1).item(), ]))
    sliceStart = np.random.randint(1, shape.cols // 10, 1).item()
    sliceEnd = np.random.randint(sliceStart + 1, shape.cols, 1).item()
    sliceStep = np.random.randint(2, 5, 1).item()
    value = np.random.randint(100, 200, 1).item()
    assert np.array_equal(NumCpp.insert(cArray, NumCpp.Slice(sliceStart, sliceEnd, sliceStep), value, NumCpp.Axis.COL),
                          np.insert(data, slice(sliceStart, sliceEnd, sliceStep), value, axis=1))

    # edge cases indices 0, last, and negative
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.array([-3, -1, 0, 2])
    sliceStart = np.random.randint(1, shape.cols // 10, 1).item()
    sliceEnd = np.random.randint(sliceStart + 1, shape.cols, 1).item()
    sliceStep = np.random.randint(2, 5, 1).item()
    value = np.random.randint(100, 200, 1).item()
    assert np.array_equal(NumCpp.insert(cArray, NumCpp.Slice(sliceStart, sliceEnd, sliceStep), value, NumCpp.Axis.COL),
                          np.insert(data, slice(sliceStart, sliceEnd, sliceStep), value, axis=1))

    # insertIndicesArray axis=NONE
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.unique(np.random.randint(1, shape.size(), [np.random.randint(2, 10, 1).item(), ], dtype=np.int32))
    cIndices = NumCpp.NdArrayInt32(1, indices.size)
    cIndices.setArray(indices)
    values = np.random.randint(100, 200, [1, indices.size])
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cIndices, cValues, NumCpp.Axis.NONE).flatten(),
                          np.insert(data, indices, values, axis=None))

    # edge cases indices 0, last, and negative
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.array([-3, -1, 0, 2, data.size], dtype=np.int32)
    cIndices = NumCpp.NdArrayInt32(1, indices.size)
    cIndices.setArray(indices)
    values = np.random.randint(100, 200, [1, indices.size])
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cIndices, cValues, NumCpp.Axis.NONE).flatten(),
                          np.insert(data, indices, values, axis=None))

    # insertIndicesArray axis=ROW, 1d
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.unique(np.random.randint(1, shape.rows, [np.random.randint(2, 10, 1).item(), ], dtype=np.int32))
    cIndices = NumCpp.NdArrayInt32(1, indices.size)
    cIndices.setArray(indices)
    values = np.random.randint(100, 200, [1, shape.cols])
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cIndices, cValues, NumCpp.Axis.ROW),
                          np.insert(data, indices, values, axis=0))

    # edge cases indices 0, last, and negative
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.array([-3, -1, 0, 2, data.shape[0]], dtype=np.int32)
    cIndices = NumCpp.NdArrayInt32(1, indices.size)
    cIndices.setArray(indices)
    values = np.random.randint(100, 200, [1, shape.cols])
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cIndices, cValues, NumCpp.Axis.ROW),
                          np.insert(data, indices, values, axis=0))

    # insertIndicesArray axis=ROW, 2d
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.unique(np.random.randint(1, shape.rows, [np.random.randint(2, 10, 1).item(), ], dtype=np.int32))
    cIndices = NumCpp.NdArrayInt32(1, indices.size)
    cIndices.setArray(indices)
    values = np.random.randint(100, 200, [indices.size, shape.cols])
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cIndices, cValues, NumCpp.Axis.ROW),
                          np.insert(data, indices, values, axis=0))

    # edge cases indices 0, last, and negative
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.array([-3, -1, 0, 2, data.shape[0]], dtype=np.int32)
    cIndices = NumCpp.NdArrayInt32(1, indices.size)
    cIndices.setArray(indices)
    values = np.random.randint(100, 200, [indices.size, shape.cols])
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cIndices, cValues, NumCpp.Axis.ROW),
                          np.insert(data, indices, values, axis=0))

    # insertIndicesArray axis=COL 1d
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.unique(np.random.randint(1, shape.cols, [np.random.randint(2, 10, 1).item(), ], dtype=np.int32))
    cIndices = NumCpp.NdArrayInt32(1, indices.size)
    cIndices.setArray(indices)
    values = np.random.randint(100, 200, [shape.rows, 1])
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cIndices, cValues, NumCpp.Axis.COL),
                          np.insert(data, indices, values, axis=1))

    # edge cases indices 0, last, and negative
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.array([-3, -1, 0, 2, data.shape[1]], dtype=np.int32)
    cIndices = NumCpp.NdArrayInt32(1, indices.size)
    cIndices.setArray(indices)
    values = np.random.randint(100, 200, [shape.rows, 1])
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cIndices, cValues, NumCpp.Axis.COL),
                          np.insert(data, indices, values, axis=1))

    # insertIndicesArray axis=COL 2d
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.unique(np.random.randint(1, shape.cols, [np.random.randint(2, 10, 1).item(), ], dtype=np.int32))
    cIndices = NumCpp.NdArrayInt32(1, indices.size)
    cIndices.setArray(indices)
    values = np.random.randint(100, 200, [shape.rows, indices.size])
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cIndices, cValues, NumCpp.Axis.COL),
                          np.insert(data, indices, values, axis=1))

    # edge cases indices 0, last, and negative
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    indices = np.array([-3, -1, 0, 2, data.shape[1]], dtype=np.int32)
    cIndices = NumCpp.NdArrayInt32(1, indices.size)
    cIndices.setArray(indices)
    values = np.random.randint(100, 200, [shape.rows, indices.size])
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cIndices, cValues, NumCpp.Axis.COL),
                          np.insert(data, indices, values, axis=1))

    # insertSliceArray axis=NONE
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    sliceStart = np.random.randint(1, shape.size() // 10, 1).item()
    sliceEnd = np.random.randint(sliceStart + 1, shape.size(), 1).item()
    sliceStep = np.random.randint(2, 5, 1).item()
    cSlice = NumCpp.Slice(sliceStart, sliceEnd, sliceStep)
    values = np.random.randint(100, 200, [cSlice.numElements(cArray.size()), ])
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cSlice, cValues, NumCpp.Axis.NONE).flatten(),
                          np.insert(data, slice(sliceStart, sliceEnd, sliceStep), values, axis=None))

    # edge cases indices 0, last, and negative
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    sliceStart = 3
    sliceEnd = -3
    sliceStep = 1
    cSlice = NumCpp.Slice(sliceStart, sliceEnd, sliceStep)
    values = np.random.randint(100, 200, [cSlice.numElements(cArray.size()), ])
    cValues = NumCpp.NdArray(1, values.size)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cSlice, cValues, NumCpp.Axis.NONE).flatten(),
                          np.insert(data, slice(sliceStart, sliceEnd, sliceStep), values, axis=None))

    # insertSliceArray axis=ROW 1d
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    sliceStart = np.random.randint(1, shape.rows // 10, 1).item()
    sliceEnd = np.random.randint(sliceStart + 1, shape.rows, 1).item()
    sliceStep = np.random.randint(2, 5, 1).item()
    cSlice = NumCpp.Slice(sliceStart, sliceEnd, sliceStep)
    values = np.random.randint(100, 200, [1, shape.cols])
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cSlice, cValues, NumCpp.Axis.ROW),
                          np.insert(data, slice(sliceStart, sliceEnd, sliceStep), values, axis=0))

    # edge cases indices 0, last, and negative
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    sliceStart = 3
    sliceEnd = -3
    sliceStep = 1
    cSlice = NumCpp.Slice(sliceStart, sliceEnd, sliceStep)
    values = np.random.randint(100, 200, [1, shape.cols])
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cSlice, cValues, NumCpp.Axis.ROW),
                          np.insert(data, slice(sliceStart, sliceEnd, sliceStep), values, axis=0))

    # insertSliceArray axis=ROW 2d
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    sliceStart = np.random.randint(1, shape.rows // 10, 1).item()
    sliceEnd = np.random.randint(sliceStart + 1, shape.rows, 1).item()
    sliceStep = np.random.randint(2, 5, 1).item()
    cSlice = NumCpp.Slice(sliceStart, sliceEnd, sliceStep)
    values = np.random.randint(100, 200, [cSlice.numElements(shapeInput[0]), shape.cols])
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cSlice, cValues, NumCpp.Axis.ROW),
                          np.insert(data, slice(sliceStart, sliceEnd, sliceStep), values, axis=0))

    # edge cases indices 0, last, and negative
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    sliceStart = 3
    sliceEnd = -3
    sliceStep = 1
    cSlice = NumCpp.Slice(sliceStart, sliceEnd, sliceStep)
    values = np.random.randint(100, 200, [cSlice.numElements(shapeInput[0]), shape.cols])
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cSlice, cValues, NumCpp.Axis.ROW),
                          np.insert(data, slice(sliceStart, sliceEnd, sliceStep), values, axis=0))
    
    # insertSliceArray axis=COL 1d
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    sliceStart = np.random.randint(1, shape.cols // 10, 1).item()
    sliceEnd = np.random.randint(sliceStart + 1, shape.cols, 1).item()
    sliceStep = np.random.randint(2, 5, 1).item()
    cSlice = NumCpp.Slice(sliceStart, sliceEnd, sliceStep)
    values = np.random.randint(100, 200, [shape.rows, 1])
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cSlice, cValues, NumCpp.Axis.COL),
                          np.insert(data, slice(sliceStart, sliceEnd, sliceStep), values, axis=1))

    # edge cases indices 0, last, and negative
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    sliceStart = 3
    sliceEnd = -3
    sliceStep = 1
    cSlice = NumCpp.Slice(sliceStart, sliceEnd, sliceStep)
    values = np.random.randint(100, 200, [shape.rows, 1])
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cSlice, cValues, NumCpp.Axis.COL),
                          np.insert(data, slice(sliceStart, sliceEnd, sliceStep), values, axis=1))

    # insertSliceArray axis=COL 2d
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    sliceStart = np.random.randint(1, shape.cols // 10, 1).item()
    sliceEnd = np.random.randint(sliceStart + 1, shape.cols, 1).item()
    sliceStep = np.random.randint(2, 5, 1).item()
    cSlice = NumCpp.Slice(sliceStart, sliceEnd, sliceStep)
    values = np.random.randint(100, 200, [shape.rows, cSlice.numElements(shapeInput[1])])
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cSlice, cValues, NumCpp.Axis.COL),
                          np.insert(data, slice(sliceStart, sliceEnd, sliceStep), values, axis=1))

    # edge cases indices 0, last, and negative
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 100, shapeInput)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    sliceStart = 3
    sliceEnd = -3
    sliceStep = 1
    cSlice = NumCpp.Slice(sliceStart, sliceEnd, sliceStep)
    values = np.random.randint(100, 200, [shape.rows, cSlice.numElements(shapeInput[1])])
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.insert(cArray, cSlice, cValues, NumCpp.Axis.COL),
                          np.insert(data, slice(sliceStart, sliceEnd, sliceStep), values, axis=1))


####################################################################################
def test_interp():
    endPoint = np.random.randint(10, 20, [1, ]).item()
    numPoints = np.random.randint(50, 100, [1, ]).item()
    resample = np.random.randint(2, 5, [1, ]).item()
    xpData = np.linspace(0, endPoint, numPoints, endpoint=True)
    fpData = np.sin(xpData)
    xData = np.linspace(0, endPoint, numPoints * resample, endpoint=True)
    cXp = NumCpp.NdArray(1, numPoints)
    cFp = NumCpp.NdArray(1, numPoints)
    cX = NumCpp.NdArray(1, numPoints * resample)
    cXp.setArray(xpData)
    cFp.setArray(fpData)
    cX.setArray(xData)
    assert np.array_equal(np.round(NumCpp.interp(cX, cXp, cFp).flatten(), 9),
                          np.round(np.interp(xData, xpData, fpData), 9))

    endPoint = np.random.randint(10, 20, [1, ]).item()
    numPoints = np.random.randint(50, 100, [1, ]).item()
    resample = np.random.randint(2, 5, [1, ]).item()
    xpData = np.linspace(0, endPoint, numPoints, endpoint=True)
    fpData = np.sin(xpData)
    xData = np.linspace(0, endPoint, numPoints * resample, endpoint=True)
    # NumPy doesn't require ordered data so let's match that behavoir
    np.random.shuffle(xData)
    cXp = NumCpp.NdArray(1, numPoints)
    cFp = NumCpp.NdArray(1, numPoints)
    cX = NumCpp.NdArray(1, numPoints * resample)
    cXp.setArray(xpData)
    cFp.setArray(fpData)
    cX.setArray(xData)
    assert np.array_equal(np.round(NumCpp.interp(cX, cXp, cFp).flatten(), 9),
                          np.round(np.interp(xData, xpData, fpData), 9))


####################################################################################
def test_intersect1d():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt32(shape)
    cArray2 = NumCpp.NdArrayUInt32(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.intersect1d(
        cArray1, cArray2).getNumpyArray().flatten(), np.intersect1d(data1, data2))


####################################################################################
def test_invert():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.invert(
        cArray).getNumpyArray(), np.invert(data))


####################################################################################
def test_isclose():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.rand(shape.rows, shape.cols)
    data2 = data1 + np.random.randn(shape.rows, shape.cols) * 1e-5
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    rtol = 1e-5
    atol = 1e-8
    assert np.array_equal(NumCpp.isclose(cArray1, cArray2, rtol, atol).getNumpyArray(),
                          np.isclose(data1, data2, rtol=rtol, atol=atol))


####################################################################################
def test_isinf():
    value = np.random.randn(1).item() * 100 + 1000
    assert not NumCpp.isinfScalar(value)

    assert NumCpp.isinfScalar(np.inf)
    assert NumCpp.isinfScalar(-np.inf)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data[data > 1000] = np.inf
    cArray.setArray(data)
    assert np.array_equal(NumCpp.isinfArray(cArray), np.isinf(data))


####################################################################################
def test_isposinf():
    value = np.random.randn(1).item() * 100 + 1000
    assert not NumCpp.isposinfScalar(value)

    assert NumCpp.isposinfScalar(np.inf)
    assert not NumCpp.isposinfScalar(-np.inf)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data[data > np.percentile(data, 75)] = np.inf
    data[data < np.percentile(data, 25)] = -np.inf
    cArray.setArray(data)
    assert np.array_equal(NumCpp.isposinfArray(cArray), np.isposinf(data))


####################################################################################
def test_isneginf():
    value = np.random.randn(1).item() * 100 + 1000
    assert not NumCpp.isneginfScalar(value)

    assert not NumCpp.isneginfScalar(np.inf)
    assert NumCpp.isneginfScalar(-np.inf)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data[data > np.percentile(data, 75)] = np.inf
    data[data < np.percentile(data, 25)] = -np.inf
    cArray.setArray(data)
    assert np.array_equal(NumCpp.isneginfArray(cArray), np.isneginf(data))


####################################################################################
def test_isnan():
    value = np.random.randn(1).item() * 100 + 1000
    assert NumCpp.isnanScalar(value) == np.isnan(value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data[data > 1000] = np.nan
    cArray.setArray(data)
    assert np.array_equal(NumCpp.isnanArray(cArray), np.isnan(data))


####################################################################################
def test_kaiser():
    if NumCpp.STL_SPECIAL_FUNCTIONS or not NumCpp.NUMCPP_NO_USE_BOOST:
        m = np.random.randint(2, 100)
        beta = np.random.rand(1).item()
        assert np.array_equal(np.round(NumCpp.kaiser(m, beta), 9).flatten(),
                              np.round(np.kaiser(m, beta), 9))


####################################################################################
def test_lcm():
    if not NumCpp.NUMCPP_NO_USE_BOOST or NumCpp.STL_GCD_LCM:
        value1 = np.random.randint(1, 1000, [1, ]).item()
        value2 = np.random.randint(1, 1000, [1, ]).item()
        assert NumCpp.lcmScalar(value1, value2) == np.lcm(value1, value2)

    if not NumCpp.NUMCPP_NO_USE_BOOST:
        size = np.random.randint(2, 10, [1, ]).item()
        cArray = NumCpp.NdArrayUInt32(1, size)
        data = np.random.randint(1, 100, [size, ], dtype=np.uint32)
        cArray.setArray(data)
        assert NumCpp.lcmArray(cArray) == np.lcm.reduce(data)  # noqa


####################################################################################
def test_ldexp():
    value1 = np.random.randn(1).item() * 100
    value2 = np.random.randint(1, 20, [1, ]).item()
    assert np.round(NumCpp.ldexpScalar(value1, value2),
                    9) == np.round(np.ldexp(value1, value2), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArrayUInt8(shape)
    data1 = np.random.randn(shape.rows, shape.cols) * 100
    data2 = np.random.randint(1, 20, [shape.rows, shape.cols], dtype=np.uint8)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(np.round(NumCpp.ldexpArray(
        cArray1, cArray2), 9), np.round(np.ldexp(data1, data2), 9))


####################################################################################
def test_left_shift():
    shapeInput = np.random.randint(20, 100, [2, ])
    bitsToshift = np.random.randint(1, 32, [1, ]).item()
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(1, np.iinfo(np.uint32).max, [
                             shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.left_shift(cArray, bitsToshift).getNumpyArray(),
                          np.left_shift(data, bitsToshift))


####################################################################################
def test_less():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.less(cArray1, cArray2).getNumpyArray(),
                          np.less(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.less(cArray1, cArray2).getNumpyArray(),
                          np.less(data1, data2))


####################################################################################
def test_less_equal():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.less_equal(cArray1, cArray2).getNumpyArray(),
                          np.less_equal(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.less_equal(cArray1, cArray2).getNumpyArray(),
                          np.less_equal(data1, data2))


####################################################################################
def test_load():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    tempDir = tempfile.gettempdir()
    tempFile = os.path.join(tempDir, 'NdArrayDump.bin')
    NumCpp.dump(cArray, tempFile)
    assert os.path.isfile(tempFile)
    data2 = NumCpp.load(tempFile).reshape(shape)
    assert np.array_equal(data, data2)
    os.remove(tempFile)


####################################################################################
def test_linspace():
    start = np.random.randint(1, 10, [1, ]).item()
    end = np.random.randint(start + 10, 100, [1, ]).item()
    numPoints = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(np.round(NumCpp.linspace(start, end, numPoints, True).getNumpyArray().flatten(), 9),
                          np.round(np.linspace(start, end, numPoints, endpoint=True), 9))

    start = np.random.randint(1, 10, [1, ]).item()
    end = np.random.randint(start + 10, 100, [1, ]).item()
    numPoints = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(np.round(NumCpp.linspace(start, end, numPoints, False).getNumpyArray().flatten(), 9),
                          np.round(np.linspace(start, end, numPoints, endpoint=False), 9))


####################################################################################
def test_log():
    value = np.random.randn(1).item() * 100 + 1000
    assert np.round(NumCpp.logScalar(value), 9) == np.round(np.log(value), 9)

    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.logScalar(value), 9) == np.round(np.log(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.logArray(
        cArray), 9), np.round(np.log(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.logArray(
        cArray), 9), np.round(np.log(data), 9))


####################################################################################
def test_logb():
    value = np.random.randn(1).item() * 100 + 1000
    base = np.random.randn(1).item() * 2 + 10
    assert np.round(NumCpp.logbScalar(value, base), 9) == np.round(
        np.log(value) / np.log(base), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.logbArray(
        cArray, base), 9), np.round(np.log(data) / np.log(base), 9))


####################################################################################
def test_logspace():
    start = np.random.randint(0, 10)
    stop = np.random.randint(start + 1, 3 * start + 2)
    num = np.random.randint(1, 100)
    base = np.random.rand(1) * 10
    assert np.array_equal(np.round(NumCpp.logspace(start, stop, num, True, base).flatten(), 9),
                          np.round(np.logspace(start=start, stop=stop, num=num, endpoint=True, base=base), 9))
    assert np.array_equal(np.round(NumCpp.logspace(start, stop, num, False, base).flatten(), 9),
                          np.round(np.logspace(start=start, stop=stop, num=num, endpoint=False, base=base), 9))


####################################################################################
def test_log10():
    value = np.random.randn(1).item() * 100 + 1000
    assert np.round(NumCpp.log10Scalar(value),
                    9) == np.round(np.log10(value), 9)

    components = np.random.randn(2).astype(float) * 100 + 100
    value = complex(components[0], components[1])
    assert np.round(NumCpp.log10Scalar(value),
                    9) == np.round(np.log10(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.log10Array(
        cArray), 9), np.round(np.log10(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.log10Array(
        cArray), 9), np.round(np.log10(data), 9))


####################################################################################
def test_log1p():
    value = np.random.randn(1).item() * 100 + 1000
    assert np.round(NumCpp.log1pScalar(value),
                    9) == np.round(np.log1p(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.log1pArray(
        cArray), 9), np.round(np.log1p(data), 9))


####################################################################################
def test_log2():
    value = np.random.randn(1).item() * 100 + 1000
    assert np.round(NumCpp.log2Scalar(value), 9) == np.round(np.log2(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.log2Array(
        cArray), 9), np.round(np.log2(data), 9))


####################################################################################
def test_logaddexp():
    values = np.random.rand(2) * 10
    assert np.round(NumCpp.logaddexpScalar(*values), 9) == np.round(np.logaddexp(*values), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.rand(shape.rows, shape.cols) * 10
    data2 = np.random.rand(shape.rows, shape.cols) * 10
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(np.round(NumCpp.logaddexpArray(cArray1, cArray2), 9),
                          np.round(np.logaddexp(data1, data2), 9))


####################################################################################
def test_logaddexp2():
    values = np.random.rand(2) * 10
    assert np.round(NumCpp.logaddexp2Scalar(*values), 9) == np.round(np.logaddexp2(*values), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.rand(shape.rows, shape.cols) * 10
    data2 = np.random.rand(shape.rows, shape.cols) * 10
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(np.round(NumCpp.logaddexp2Array(cArray1, cArray2), 9),
                          np.round(np.logaddexp2(data1, data2), 9))


####################################################################################
def test_logical_and():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 20, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 20, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.logical_and(
        cArray1, cArray2).getNumpyArray(), np.logical_and(data1, data2))


####################################################################################
def test_logical_not():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 20, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.logical_not(
        cArray).getNumpyArray(), np.logical_not(data))


####################################################################################
def test_logical_or():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 20, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 20, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.logical_or(
        cArray1, cArray2).getNumpyArray(), np.logical_or(data1, data2))


####################################################################################
def test_logical_xor():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 20, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 20, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.logical_xor(
        cArray1, cArray2).getNumpyArray(), np.logical_xor(data1, data2))


####################################################################################
def test_matmul():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[1].item(), shapeInput[0].item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    data1 = np.random.randint(0, 20, [shape1.rows, shape1.cols])
    data2 = np.random.randint(0, 20, [shape2.rows, shape2.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.matmul(
        cArray1, cArray2), np.matmul(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[1].item(), shapeInput[0].item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArrayComplexDouble(shape2)
    data1 = np.random.randint(0, 20, [shape1.rows, shape1.cols])
    real2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    imag2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.matmul(
        cArray1, cArray2), np.matmul(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[1].item(), shapeInput[0].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape1)
    cArray2 = NumCpp.NdArrayComplexDouble(shape2)
    real1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    imag1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    imag2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.matmul(
        cArray1, cArray2), np.matmul(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[1].item(), shapeInput[0].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    real1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    imag1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data1 = real1 + 1j * imag1
    data2 = np.random.randint(0, 20, [shape2.rows, shape2.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.matmul(
        cArray1, cArray2), np.matmul(data1, data2))


####################################################################################
def test_max():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.max(cArray, NumCpp.Axis.NONE).item() == np.max(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.max(cArray, NumCpp.Axis.NONE).item() == np.max(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.max(cArray, NumCpp.Axis.ROW).flatten(),
                          np.max(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.max(cArray, NumCpp.Axis.ROW).flatten(),
                          np.max(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.max(cArray, NumCpp.Axis.COL).flatten(),
                          np.max(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.max(cArray, NumCpp.Axis.COL).flatten(),
                          np.max(data, axis=1))


####################################################################################
def test_maximum():
    # shapes the same
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.maximum(
        cArray1, cArray2), np.maximum(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.maximum(
        cArray1, cArray2), np.maximum(data1, data2))

    # shape and scalar
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 100)
    cArray1.setArray(data1)
    assert np.array_equal(NumCpp.maximum(
        cArray1, data2), np.maximum(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(0, 100)
    imag2 = np.random.randint(0, 100)
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    assert np.array_equal(NumCpp.maximum(
        cArray1, data2), np.maximum(data1, data2))

    #scalar and shape
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray2 = NumCpp.NdArray(shape)
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data1 = np.random.randint(0, 100)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.maximum(
        data1, cArray2), np.maximum(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    real1 = np.random.randint(0, 100)
    imag1 = np.random.randint(0, 100)
    data1 = real1 + 1j * imag1
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.maximum(
        data1, cArray2), np.maximum(data1, data2))

    # shape and  flat row
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(1, shape.cols)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 100, [1, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.maximum(
        cArray1, cArray2), np.maximum(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(1, shape.cols)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [1, shape.cols])
    imag2 = np.random.randint(1, 100, [1, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.maximum(
        cArray1, cArray2), np.maximum(data1, data2))

    # shape and flat col
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape.rows, 1)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 100, [shape.rows, 1])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.maximum(
        cArray1, cArray2), np.maximum(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, 1])
    imag2 = np.random.randint(1, 100, [shape.rows, 1])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.maximum(
        cArray1, cArray2), np.maximum(data1, data2))

    # flat row and flat col
    shape = np.random.randint(20, 100, [2, ])
    cArray1 = NumCpp.NdArray(1, shape[0])
    cArray2 = NumCpp.NdArray(shape[1], 1)
    data1 = np.random.randint(0, 100, [1, shape[0]])
    data2 = np.random.randint(0, 100, [shape[1], 1])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.maximum(
        cArray1, cArray2), np.maximum(data1, data2))

    shape = np.random.randint(20, 100, [2, ])
    cArray1 = NumCpp.NdArrayComplexDouble(1, shape[0])
    cArray2 = NumCpp.NdArrayComplexDouble(shape[1], 1)
    real1 = np.random.randint(1, 100, [1, shape[0]])
    imag1 = np.random.randint(1, 100, [1, shape[0]])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape[1], 1])
    imag2 = np.random.randint(1, 100, [shape[1], 1])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.maximum(
        cArray1, cArray2), np.maximum(data1, data2))

    # flat col and flat row
    shape = np.random.randint(20, 100, [2, ])
    cArray2 = NumCpp.NdArray(1, shape[0])
    cArray1 = NumCpp.NdArray(shape[1], 1)
    data2 = np.random.randint(0, 100, [1, shape[0]])
    data1 = np.random.randint(0, 100, [shape[1], 1])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.maximum(
        cArray1, cArray2), np.maximum(data1, data2))

    shape = np.random.randint(20, 100, [2, ])
    cArray2 = NumCpp.NdArrayComplexDouble(1, shape[0])
    cArray1 = NumCpp.NdArrayComplexDouble(shape[1], 1)
    real2 = np.random.randint(1, 100, [1, shape[0]])
    imag2 = np.random.randint(1, 100, [1, shape[0]])
    data2 = real2 + 1j * imag2
    real1 = np.random.randint(1, 100, [shape[1], 1])
    imag1 = np.random.randint(1, 100, [shape[1], 1])
    data1 = real1 + 1j * imag1
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.maximum(
        cArray1, cArray2), np.maximum(data1, data2))


####################################################################################
def test_mean():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.round(NumCpp.mean(cArray, NumCpp.Axis.NONE).getNumpyArray().item(), 9) == \
        np.round(np.mean(data, axis=None).item(), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.round(NumCpp.mean(cArray, NumCpp.Axis.NONE).getNumpyArray().item(), 9) == \
        np.round(np.mean(data, axis=None).item(), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.mean(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), 9),
                          np.round(np.mean(data, axis=0), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.mean(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), 9),
                          np.round(np.mean(data, axis=0), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.mean(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), 9),
                          np.round(np.mean(data, axis=1), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.mean(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), 9),
                          np.round(np.mean(data, axis=1), 9))


####################################################################################
def test_median():
    isEven = True
    while isEven:
        shapeInput = np.random.randint(20, 100, [2, ])
        isEven = shapeInput.prod().item() % 2 == 0
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())  # noqa
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.median(cArray, NumCpp.Axis.NONE).getNumpyArray(
    ).flatten().item() == np.median(data, axis=None).item()

    isEven = True
    while isEven:
        shapeInput = np.random.randint(20, 100, [2, ])
        isEven = shapeInput.prod().item() % 2 == 0
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.median(
        cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), np.median(data, axis=0))

    isEven = True
    while isEven:
        shapeInput = np.random.randint(20, 100, [2, ])
        isEven = shapeInput.prod().item() % 2 == 0
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.median(
        cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), np.median(data, axis=1))


####################################################################################
def test_meshgrid():
    start = np.random.randint(0, 20, [1, ]).item()
    end = np.random.randint(30, 100, [1, ]).item()
    step = np.random.randint(1, 5, [1, ]).item()
    dataI = np.arange(start, end, step)
    iSlice = NumCpp.Slice(start, end, step)
    start = np.random.randint(0, 20, [1, ]).item()
    end = np.random.randint(30, 100, [1, ]).item()
    step = np.random.randint(1, 5, [1, ]).item()
    dataJ = np.arange(start, end, step)
    jSlice = NumCpp.Slice(start, end, step)
    iMesh, jMesh = np.meshgrid(dataI, dataJ)
    iMeshC, jMeshC = NumCpp.meshgrid(iSlice, jSlice)
    assert np.array_equal(iMeshC.getNumpyArray(), iMesh)
    assert np.array_equal(jMeshC.getNumpyArray(), jMesh)


####################################################################################
def test_min():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.min(cArray, NumCpp.Axis.NONE).item() == np.min(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.min(cArray, NumCpp.Axis.NONE).item() == np.min(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.min(cArray, NumCpp.Axis.ROW).flatten(),
                          np.min(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.min(cArray, NumCpp.Axis.ROW).flatten(),
                          np.min(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.min(cArray, NumCpp.Axis.COL).flatten(),
                          np.min(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.min(cArray, NumCpp.Axis.COL).flatten(),
                          np.min(data, axis=1))


####################################################################################
def test_minimum():
    # shapes the same
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.minimum(
        cArray1, cArray2), np.minimum(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.minimum(
        cArray1, cArray2), np.minimum(data1, data2))

    # shape and scalar
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 100)
    cArray1.setArray(data1)
    assert np.array_equal(NumCpp.minimum(
        cArray1, data2), np.minimum(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(0, 100)
    imag2 = np.random.randint(0, 100)
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    assert np.array_equal(NumCpp.minimum(
        cArray1, data2), np.minimum(data1, data2))

    #scalar and shape
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray2 = NumCpp.NdArray(shape)
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data1 = np.random.randint(0, 100)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.minimum(
        data1, cArray2), np.minimum(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    real1 = np.random.randint(0, 100)
    imag1 = np.random.randint(0, 100)
    data1 = real1 + 1j * imag1
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.minimum(
        data1, cArray2), np.minimum(data1, data2))

    # shape and  flat row
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(1, shape.cols)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 100, [1, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.minimum(
        cArray1, cArray2), np.minimum(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(1, shape.cols)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [1, shape.cols])
    imag2 = np.random.randint(1, 100, [1, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.minimum(
        cArray1, cArray2), np.minimum(data1, data2))

    # shape and flat col
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape.rows, 1)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 100, [shape.rows, 1])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.minimum(
        cArray1, cArray2), np.minimum(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, 1])
    imag2 = np.random.randint(1, 100, [shape.rows, 1])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.minimum(
        cArray1, cArray2), np.minimum(data1, data2))

    # flat row and flat col
    shape = np.random.randint(20, 100, [2, ])
    cArray1 = NumCpp.NdArray(1, shape[0])
    cArray2 = NumCpp.NdArray(shape[1], 1)
    data1 = np.random.randint(0, 100, [1, shape[0]])
    data2 = np.random.randint(0, 100, [shape[1], 1])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.minimum(
        cArray1, cArray2), np.minimum(data1, data2))

    shape = np.random.randint(20, 100, [2, ])
    cArray1 = NumCpp.NdArrayComplexDouble(1, shape[0])
    cArray2 = NumCpp.NdArrayComplexDouble(shape[1], 1)
    real1 = np.random.randint(1, 100, [1, shape[0]])
    imag1 = np.random.randint(1, 100, [1, shape[0]])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape[1], 1])
    imag2 = np.random.randint(1, 100, [shape[1], 1])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.minimum(
        cArray1, cArray2), np.minimum(data1, data2))

    # flat col and flat row
    shape = np.random.randint(20, 100, [2, ])
    cArray2 = NumCpp.NdArray(1, shape[0])
    cArray1 = NumCpp.NdArray(shape[1], 1)
    data2 = np.random.randint(0, 100, [1, shape[0]])
    data1 = np.random.randint(0, 100, [shape[1], 1])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.minimum(
        cArray1, cArray2), np.minimum(data1, data2))

    shape = np.random.randint(20, 100, [2, ])
    cArray2 = NumCpp.NdArrayComplexDouble(1, shape[0])
    cArray1 = NumCpp.NdArrayComplexDouble(shape[1], 1)
    real2 = np.random.randint(1, 100, [1, shape[0]])
    imag2 = np.random.randint(1, 100, [1, shape[0]])
    data2 = real2 + 1j * imag2
    real1 = np.random.randint(1, 100, [shape[1], 1])
    imag1 = np.random.randint(1, 100, [shape[1], 1])
    data1 = real1 + 1j * imag1
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.minimum(
        cArray1, cArray2), np.minimum(data1, data2))


####################################################################################
def test_mod():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt32(shape)
    cArray2 = NumCpp.NdArrayUInt32(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.mod(
        cArray1, cArray2).getNumpyArray(), np.mod(data1, data2))


####################################################################################
def test_multiply():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.multiply(cArray1, cArray2), data1 * data2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    value = np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.multiply(cArray, value), data * value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    value = np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.multiply(value, cArray), data * value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.multiply(cArray1, cArray2), data1 * data2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(-100, 100) + 1j * np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.multiply(cArray, value), data * value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(-100, 100) + 1j * np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.multiply(value, cArray), data * value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArray(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.multiply(cArray1, cArray2), data1 * data2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.multiply(cArray1, cArray2), data1 * data2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    value = np.random.randint(-100, 100) + 1j * np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.multiply(cArray, value), data * value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    value = np.random.randint(-100, 100) + 1j * np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.multiply(value, cArray), data * value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.multiply(cArray, value), data * value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.multiply(value, cArray), data * value)


####################################################################################
def test_nan_to_num():
    shapeInput = np.random.randint(50, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.size(), ]).astype(float)

    nan_idx = np.random.choice(range(data.size), 10, replace=False)
    pos_inf_idx = np.random.choice(range(data.size), 10, replace=False)
    neg_inf_idx = np.random.choice(range(data.size), 10, replace=False)

    data[nan_idx] = np.nan
    data[pos_inf_idx] = np.inf
    data[neg_inf_idx] = -np.inf
    data = data.reshape(shapeInput)
    cArray.setArray(data)

    nan_replace = float(np.random.randint(100))
    pos_inf_replace = float(np.random.randint(100))
    neg_inf_replace = float(np.random.randint(100))

    assert np.array_equal(NumCpp.nan_to_num(cArray, nan_replace, pos_inf_replace, neg_inf_replace),
                          np.nan_to_num(data, nan=nan_replace, posinf=pos_inf_replace, neginf=neg_inf_replace))


####################################################################################
def test_nanargmax():
    shapeInput = np.random.randint(10, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert NumCpp.nanargmax(
        cArray, NumCpp.Axis.NONE).item() == np.nanargmax(data)

    shapeInput = np.random.randint(10, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanargmax(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(),
                          np.nanargmax(data, axis=0))

    shapeInput = np.random.randint(10, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanargmax(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(),
                          np.nanargmax(data, axis=1))


####################################################################################
def test_nanargmin():
    shapeInput = np.random.randint(10, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert NumCpp.nanargmin(
        cArray, NumCpp.Axis.NONE).item() == np.nanargmin(data)

    shapeInput = np.random.randint(10, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanargmin(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(),
                          np.nanargmin(data, axis=0))

    shapeInput = np.random.randint(10, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanargmin(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(),
                          np.nanargmin(data, axis=1))


####################################################################################
def test_nancumprod():
    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nancumprod(cArray, NumCpp.Axis.NONE).getNumpyArray().flatten(),
                          np.nancumprod(data, axis=None))

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nancumprod(
        cArray, NumCpp.Axis.ROW).getNumpyArray(), np.nancumprod(data, axis=0))

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nancumprod(
        cArray, NumCpp.Axis.COL).getNumpyArray(), np.nancumprod(data, axis=1))


####################################################################################
def test_nancumsum():
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nancumsum(cArray, NumCpp.Axis.NONE).getNumpyArray().flatten(),
                          np.nancumsum(data, axis=None))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nancumsum(
        cArray, NumCpp.Axis.ROW).getNumpyArray(), np.nancumsum(data, axis=0))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nancumsum(
        cArray, NumCpp.Axis.COL).getNumpyArray(), np.nancumsum(data, axis=1))


####################################################################################
def test_nanmax():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert NumCpp.nanmax(cArray, NumCpp.Axis.NONE).item() == np.nanmax(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanmax(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(),
                          np.nanmax(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanmax(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(),
                          np.nanmax(data, axis=1))


####################################################################################
def test_nanmean():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert NumCpp.nanmean(cArray, NumCpp.Axis.NONE).item() == np.nanmean(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanmean(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(),
                          np.nanmean(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanmean(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(),
                          np.nanmean(data, axis=1))


####################################################################################
def test_nanmedian():
    isEven = True
    while isEven:
        shapeInput = np.random.randint(20, 100, [2, ])
        isEven = shapeInput.prod().item() % 2 == 0
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())  # noqa
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert (NumCpp.nanmedian(cArray, NumCpp.Axis.NONE).getNumpyArray().flatten().item() ==
            np.nanmedian(data, axis=None).item())

    # isEven = True
    # while isEven:
    #     shapeInput = np.random.randint(20, 100, [2, ])
    #     isEven = shapeInput[0].item() % 2 == 0
    # shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    # cArray = NumCpp.NdArray(shape)
    # data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    # data = data.flatten()
    # data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    # data = data.reshape(shapeInput)
    # cArray.setArray(data)
    # assert np.array_equal(NumCpp.nanmedian(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(),
    # np.nanmedian(data, axis=0))
    #
    # isEven = True
    # while isEven:
    #     shapeInput = np.random.randint(20, 100, [2, ])
    #     isEven = shapeInput[1].item() % 2 == 0
    # shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    # cArray = NumCpp.NdArray(shape)
    # data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    # data = data.flatten()
    # data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    # data = data.reshape(shapeInput)
    # cArray.setArray(data)
    # assert np.array_equal(NumCpp.nanmedian(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(),
    # np.nanmedian(data, axis=1))


####################################################################################
def test_nanmin():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert NumCpp.nanmin(cArray, NumCpp.Axis.NONE).item() == np.nanmin(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanmin(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(),
                          np.nanmin(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanmin(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(),
                          np.nanmin(data, axis=1))


####################################################################################
def test_nanpercentile():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert (NumCpp.nanpercentile(cArray, percentile, NumCpp.Axis.NONE, 'lower').item() ==
            np.nanpercentile(data, percentile, axis=None, interpolation='lower'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert (NumCpp.nanpercentile(cArray, percentile, NumCpp.Axis.NONE, 'higher').item() ==
            np.nanpercentile(data, percentile, axis=None, interpolation='higher'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert (NumCpp.nanpercentile(cArray, percentile, NumCpp.Axis.NONE, 'nearest').item() ==
            np.nanpercentile(data, percentile, axis=None, interpolation='nearest'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert (NumCpp.nanpercentile(cArray, percentile, NumCpp.Axis.NONE, 'midpoint').item() ==
            np.nanpercentile(data, percentile, axis=None, interpolation='midpoint'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert (NumCpp.nanpercentile(cArray, percentile, NumCpp.Axis.NONE, 'linear').item() ==
            np.nanpercentile(data, percentile, axis=None, interpolation='linear'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(NumCpp.nanpercentile(cArray, percentile, NumCpp.Axis.ROW, 'lower').getNumpyArray().flatten(),
                          np.nanpercentile(data, percentile, axis=0, interpolation='lower'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(NumCpp.nanpercentile(cArray, percentile, NumCpp.Axis.ROW, 'higher').getNumpyArray().flatten(),
                          np.nanpercentile(data, percentile, axis=0, interpolation='higher'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(NumCpp.nanpercentile(cArray,
                                               percentile,
                                               NumCpp.Axis.ROW,
                                               'nearest').getNumpyArray().flatten(),
                          np.nanpercentile(data, percentile, axis=0, interpolation='nearest'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(
        np.round(NumCpp.nanpercentile(cArray,
                                      percentile,
                                      NumCpp.Axis.ROW,
                                      'midpoint').getNumpyArray().flatten(), 9),
        np.round(np.nanpercentile(data, percentile, axis=0, interpolation='midpoint'), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(np.round(NumCpp.nanpercentile(cArray,
                                                        percentile,
                                                        NumCpp.Axis.ROW,
                                                        'linear').getNumpyArray().flatten(), 9),
                          np.round(np.nanpercentile(data, percentile, axis=0, interpolation='linear'), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(NumCpp.nanpercentile(cArray, percentile, NumCpp.Axis.COL, 'lower').getNumpyArray().flatten(),
                          np.nanpercentile(data, percentile, axis=1, interpolation='lower'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(NumCpp.nanpercentile(cArray, percentile, NumCpp.Axis.COL, 'higher').getNumpyArray().flatten(),
                          np.nanpercentile(data, percentile, axis=1, interpolation='higher'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(NumCpp.nanpercentile(cArray,
                                               percentile,
                                               NumCpp.Axis.COL,
                                               'nearest').getNumpyArray().flatten(),
                          np.nanpercentile(data, percentile, axis=1, interpolation='nearest'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(
        np.round(NumCpp.nanpercentile(cArray,
                                      percentile,
                                      NumCpp.Axis.COL,
                                      'midpoint').getNumpyArray().flatten(), 9),
        np.round(np.nanpercentile(data, percentile, axis=1, interpolation='midpoint'), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(
        np.round(NumCpp.nanpercentile(cArray, percentile,
                 NumCpp.Axis.COL, 'linear').getNumpyArray().flatten(), 9),
        np.round(np.nanpercentile(data, percentile, axis=1, interpolation='linear'), 9))


####################################################################################
def test_nanprod():
    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert NumCpp.nanprod(cArray, NumCpp.Axis.NONE).item(
    ) == np.nanprod(data, axis=None)

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanprod(
        cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), np.nanprod(data, axis=0))

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanprod(
        cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), np.nanprod(data, axis=1))


####################################################################################
def test_nans():
    shapeInput = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.nansSquare(shapeInput)
    assert (cArray.shape[0] == shapeInput and cArray.shape[1] == shapeInput and
            cArray.size == shapeInput ** 2 and np.all(np.isnan(cArray)))

    shapeInput = np.random.randint(20, 100, [2, ])
    cArray = NumCpp.nansRowCol(shapeInput[0].item(), shapeInput[1].item())
    assert (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod() and np.all(np.isnan(cArray)))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.nansShape(shape)
    assert (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod() and np.all(np.isnan(cArray)))


####################################################################################
def test_nans_like():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.nans_like(cArray1)
    assert (cArray2.shape().rows == shape.rows and cArray2.shape().cols == shape.cols and
            cArray2.size() == shapeInput.prod() and np.all(np.isnan(cArray2.getNumpyArray())))


####################################################################################
def test_nanstd():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.round(NumCpp.nanstdev(cArray, NumCpp.Axis.NONE).item(),
                    9) == np.round(np.nanstd(data), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.nanstdev(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), 9),
                          np.round(np.nanstd(data, axis=0), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.nanstdev(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), 9),
                          np.round(np.nanstd(data, axis=1), 9))


####################################################################################
def test_nansum():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert NumCpp.nansum(cArray, NumCpp.Axis.NONE).item() == np.nansum(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nansum(
        cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), np.nansum(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nansum(
        cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), np.nansum(data, axis=1))


####################################################################################
def test_nanvar():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.round(NumCpp.nanvar(cArray, NumCpp.Axis.NONE).item(),
                    8) == np.round(np.nanvar(data), 8)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.nanvar(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), 8),
                          np.round(np.nanvar(data, axis=0), 8))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.nanvar(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), 8),
                          np.round(np.nanvar(data, axis=1), 8))


####################################################################################
def test_nbytes():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.nbytes(cArray) == data.size * 8

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.nbytes(cArray) == data.size * 16


####################################################################################
def test_negative():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.negative(cArray).getNumpyArray(), 9),
                          np.round(np.negative(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.negative(cArray).getNumpyArray(), 9),
                          np.round(np.negative(data), 9))


####################################################################################
def test_newbyteorderArray():
    value = np.random.randint(1, 100, [1, ]).item()
    assert (NumCpp.newbyteorderScalar(value, NumCpp.Endian.BIG) ==
            np.asarray([value], dtype=np.uint32).newbyteorder().item())

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        0, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.newbyteorderArray(cArray, NumCpp.Endian.BIG),
                          data.newbyteorder())


####################################################################################
def test_nth_root():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    root = np.random.rand(1).item() * 10
    assert np.array_equal(np.round(NumCpp.nth_rootArray(cArray, root), 9),
                          np.round(np.power(data, 1 / root), 9))


####################################################################################
def test_none():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.none(cArray, NumCpp.Axis.NONE).astype(
        bool).item() == np.logical_not(np.any(data).item())

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.none(cArray, NumCpp.Axis.NONE).astype(
        bool).item() == np.logical_not(np.any(data).item())

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.none(cArray, NumCpp.Axis.ROW).flatten().astype(bool),
                          np.logical_not(np.any(data, axis=0)))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.none(cArray, NumCpp.Axis.ROW).flatten().astype(bool),
                          np.logical_not(np.any(data, axis=0)))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.none(cArray, NumCpp.Axis.COL).flatten().astype(bool),
                          np.logical_not(np.any(data, axis=1)))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.none(cArray, NumCpp.Axis.COL).flatten().astype(bool),
                          np.logical_not(np.any(data, axis=1)))


####################################################################################
def test_nonzero():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    row, col = np.nonzero(data)
    rowC, colC = NumCpp.nonzero(cArray)
    assert (np.array_equal(rowC.getNumpyArray().flatten(), row) and
            np.array_equal(colC.getNumpyArray().flatten(), col))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    row, col = np.nonzero(data)
    rowC, colC = NumCpp.nonzero(cArray)
    assert (np.array_equal(rowC.getNumpyArray().flatten(), row) and
            np.array_equal(colC.getNumpyArray().flatten(), col))


####################################################################################
def test_norm():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.norm(cArray, NumCpp.Axis.NONE).flatten(
    ) == np.linalg.norm(data.flatten())

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.norm(cArray, NumCpp.Axis.NONE).item() is not None

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    norms = NumCpp.norm(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten()
    allPass = True
    for idx, row in enumerate(data.transpose()):
        if norms[idx] != np.linalg.norm(row):
            allPass = False
            break
    assert allPass

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    norms = NumCpp.norm(cArray, NumCpp.Axis.COL).getNumpyArray().flatten()
    allPass = True
    for idx, row in enumerate(data):
        if norms[idx] != np.linalg.norm(row):
            allPass = False
            break
    assert allPass

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    norms = NumCpp.norm(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten()
    assert norms is not None


####################################################################################
def test_not_equal():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.not_equal(
        cArray1, cArray2).getNumpyArray(), np.not_equal(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.not_equal(
        cArray1, cArray2).getNumpyArray(), np.not_equal(data1, data2))


####################################################################################
def test_ones():
    shapeInput = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.onesSquare(shapeInput)
    assert (cArray.shape[0] == shapeInput and cArray.shape[1] == shapeInput and
            cArray.size == shapeInput ** 2 and np.all(cArray == 1))

    shapeInput = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.onesSquareComplex(shapeInput)
    assert (cArray.shape[0] == shapeInput and cArray.shape[1] == shapeInput and
            cArray.size == shapeInput ** 2 and np.all(cArray == complex(1, 0)))

    shapeInput = np.random.randint(20, 100, [2, ])
    cArray = NumCpp.onesRowCol(shapeInput[0].item(), shapeInput[1].item())
    assert (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod() and np.all(cArray == 1))

    shapeInput = np.random.randint(20, 100, [2, ])
    cArray = NumCpp.onesRowColComplex(
        shapeInput[0].item(), shapeInput[1].item())
    assert (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod() and np.all(cArray == complex(1, 0)))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.onesShape(shape)
    assert (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod() and np.all(cArray == 1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.onesShapeComplex(shape)
    assert (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod() and np.all(cArray == complex(1, 0)))


####################################################################################
def test_ones_like():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.ones_like(cArray1)
    assert (cArray2.shape().rows == shape.rows and cArray2.shape().cols == shape.cols and
            cArray2.size() == shapeInput.prod() and np.all(cArray2.getNumpyArray() == 1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.ones_likeComplex(cArray1)
    assert (cArray2.shape().rows == shape.rows and cArray2.shape().cols == shape.cols and
            cArray2.size() == shapeInput.prod() and np.all(cArray2.getNumpyArray() == complex(1, 0)))


####################################################################################
def test_outer():
    size = np.random.randint(1, 100, [1, ]).item()
    shape = NumCpp.Shape(1, size)
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    data2 = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.outer(cArray1, cArray2),
                          np.outer(data1, data2))

    size = np.random.randint(1, 100, [1, ]).item()
    shape = NumCpp.Shape(1, size)
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 50, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 50, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.outer(cArray1, cArray2),
                          np.outer(data1, data2))


####################################################################################
def test_packbits():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 2, [shape.rows, shape.cols]).astype(bool)
    cArray = NumCpp.NdArrayBool(shape)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.packbitsLittleEndian(cArray, NumCpp.Axis.NONE).flatten(),
                          np.packbits(data, axis=None, bitorder='little'))
    assert np.array_equal(NumCpp.packbitsLittleEndian(cArray, NumCpp.Axis.ROW),
                          np.packbits(data, axis=0, bitorder='little'))
    assert np.array_equal(NumCpp.packbitsLittleEndian(cArray, NumCpp.Axis.COL),
                          np.packbits(data, axis=1, bitorder='little'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 2, [shape.rows, shape.cols]).astype(bool)
    cArray = NumCpp.NdArrayBool(shape)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.packbitsBigEndian(cArray, NumCpp.Axis.NONE).flatten(),
                          np.packbits(data, axis=None, bitorder='big'))
    assert np.array_equal(NumCpp.packbitsBigEndian(cArray, NumCpp.Axis.ROW),
                          np.packbits(data, axis=0, bitorder='big'))
    assert np.array_equal(NumCpp.packbitsBigEndian(cArray, NumCpp.Axis.COL),
                          np.packbits(data, axis=1, bitorder='big'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 2, [shape.rows, shape.cols]).astype(np.uint8)
    cArray = NumCpp.NdArrayUInt8(shape)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.packbitsLittleEndian(cArray, NumCpp.Axis.NONE).flatten(),
                          np.packbits(data, axis=None, bitorder='little'))
    assert np.array_equiv(NumCpp.packbitsLittleEndian(cArray, NumCpp.Axis.ROW),
                          np.packbits(data, axis=0, bitorder='little'))
    assert np.array_equiv(NumCpp.packbitsLittleEndian(cArray, NumCpp.Axis.COL),
                          np.packbits(data, axis=1, bitorder='little'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, 2, [shape.rows, shape.cols]).astype(np.uint8)
    cArray = NumCpp.NdArrayUInt8(shape)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.packbitsBigEndian(cArray, NumCpp.Axis.NONE).flatten(),
                          np.packbits(data, axis=None, bitorder='big'))
    assert np.array_equiv(NumCpp.packbitsBigEndian(cArray, NumCpp.Axis.ROW),
                          np.packbits(data, axis=0, bitorder='big'))
    assert np.array_equiv(NumCpp.packbitsBigEndian(cArray, NumCpp.Axis.COL),
                          np.packbits(data, axis=1, bitorder='big'))


####################################################################################
def test_pad():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    padWidth = np.random.randint(1, 10, [1, ]).item()
    padValue = np.random.randint(1, 100, [1, ]).item()
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.pad(cArray, padWidth, padValue).getNumpyArray(),
                          np.pad(data, padWidth, mode='constant', constant_values=padValue))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    padWidth = np.random.randint(1, 10, [1, ]).item()
    padValue = np.random.randint(1, 100, [1, ]).item(
    ) + 1j * np.random.randint(1, 100, [1, ]).item()
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.pad(cArray, padWidth, padValue).getNumpyArray(),
                          np.pad(data, padWidth, mode='constant', constant_values=padValue))


####################################################################################
def test_partition():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    kthElement = np.random.randint(
        0, shapeInput.prod(), [1, ], dtype=np.uint32).item()
    partitionedArray = NumCpp.partition(
        cArray, kthElement, NumCpp.Axis.NONE).getNumpyArray().flatten()
    assert (np.all(partitionedArray[kthElement] <= partitionedArray[kthElement]) and
            np.all(partitionedArray[kthElement:] >= partitionedArray[kthElement]))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    kthElement = np.random.randint(
        0, shapeInput.prod(), [1, ], dtype=np.uint32).item()
    partitionedArray = NumCpp.partition(
        cArray, kthElement, NumCpp.Axis.NONE).getNumpyArray().flatten()
    assert (np.all(partitionedArray[kthElement] <= partitionedArray[kthElement]) and
            np.all(partitionedArray[kthElement:] >= partitionedArray[kthElement]))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    kthElement = np.random.randint(
        0, shapeInput[0], [1, ], dtype=np.uint32).item()
    partitionedArray = NumCpp.partition(
        cArray, kthElement, NumCpp.Axis.ROW).getNumpyArray().transpose()
    allPass = True
    for row in partitionedArray:
        if not (np.all(row[kthElement] <= row[kthElement]) and
                np.all(row[kthElement:] >= row[kthElement])):
            allPass = False
            break
    assert allPass

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    kthElement = np.random.randint(
        0, shapeInput[0], [1, ], dtype=np.uint32).item()
    partitionedArray = NumCpp.partition(
        cArray, kthElement, NumCpp.Axis.ROW).getNumpyArray().transpose()
    allPass = True
    for row in partitionedArray:
        if not (np.all(row[kthElement] <= row[kthElement]) and
                np.all(row[kthElement:] >= row[kthElement])):
            allPass = False
            break
    assert allPass

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    kthElement = np.random.randint(
        0, shapeInput[1], [1, ], dtype=np.uint32).item()
    partitionedArray = NumCpp.partition(
        cArray, kthElement, NumCpp.Axis.COL).getNumpyArray()
    allPass = True
    for row in partitionedArray:
        if not (np.all(row[kthElement] <= row[kthElement]) and
                np.all(row[kthElement:] >= row[kthElement])):
            allPass = False
            break
    assert allPass

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    kthElement = np.random.randint(
        0, shapeInput[1], [1, ], dtype=np.uint32).item()
    partitionedArray = NumCpp.partition(
        cArray, kthElement, NumCpp.Axis.COL).getNumpyArray()
    allPass = True
    for row in partitionedArray:
        if not (np.all(row[kthElement] <= row[kthElement]) and
                np.all(row[kthElement:] >= row[kthElement])):
            allPass = False
            break
    assert allPass


####################################################################################
def test_percentile():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert (NumCpp.percentile(cArray, percentile, NumCpp.Axis.NONE, 'lower').item() ==
            np.percentile(data, percentile, axis=None, interpolation='lower'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert (NumCpp.percentile(cArray, percentile, NumCpp.Axis.NONE, 'higher').item() ==
            np.percentile(data, percentile, axis=None, interpolation='higher'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert (NumCpp.percentile(cArray, percentile, NumCpp.Axis.NONE, 'nearest').item() ==
            np.percentile(data, percentile, axis=None, interpolation='nearest'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert (NumCpp.percentile(cArray, percentile, NumCpp.Axis.NONE, 'midpoint').item() ==
            np.percentile(data, percentile, axis=None, interpolation='midpoint'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert (NumCpp.percentile(cArray, percentile, NumCpp.Axis.NONE, 'linear').item() ==
            np.percentile(data, percentile, axis=None, interpolation='linear'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(NumCpp.percentile(cArray, percentile, NumCpp.Axis.ROW, 'lower').getNumpyArray().flatten(),
                          np.percentile(data, percentile, axis=0, interpolation='lower'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(NumCpp.percentile(cArray, percentile, NumCpp.Axis.ROW, 'higher').getNumpyArray().flatten(),
                          np.percentile(data, percentile, axis=0, interpolation='higher'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(NumCpp.percentile(cArray, percentile, NumCpp.Axis.ROW, 'nearest').getNumpyArray().flatten(),
                          np.percentile(data, percentile, axis=0, interpolation='nearest'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(np.round(NumCpp.percentile(cArray,
                                                     percentile,
                                                     NumCpp.Axis.ROW,
                                                     'midpoint').getNumpyArray().flatten(), 9),
                          np.round(np.percentile(data, percentile, axis=0, interpolation='midpoint'), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(np.round(NumCpp.percentile(cArray,
                                                     percentile,
                                                     NumCpp.Axis.ROW,
                                                     'linear').getNumpyArray().flatten(), 9),
                          np.round(np.percentile(data, percentile, axis=0, interpolation='linear'), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(NumCpp.percentile(cArray, percentile, NumCpp.Axis.COL, 'lower').getNumpyArray().flatten(),
                          np.percentile(data, percentile, axis=1, interpolation='lower'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(NumCpp.percentile(cArray, percentile, NumCpp.Axis.COL, 'higher').getNumpyArray().flatten(),
                          np.percentile(data, percentile, axis=1, interpolation='higher'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(NumCpp.percentile(cArray, percentile, NumCpp.Axis.COL, 'nearest').getNumpyArray().flatten(),
                          np.percentile(data, percentile, axis=1, interpolation='nearest'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(np.round(NumCpp.percentile(cArray,
                                                     percentile,
                                                     NumCpp.Axis.COL,
                                                     'midpoint').getNumpyArray().flatten(), 9),
                          np.round(np.percentile(data, percentile, axis=1, interpolation='midpoint'), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(np.round(NumCpp.percentile(cArray,
                                                     percentile,
                                                     NumCpp.Axis.COL,
                                                     'linear').getNumpyArray().flatten(), 9),
                          np.round(np.percentile(data, percentile, axis=1, interpolation='linear'), 9))


####################################################################################
def test_place():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    cMask = NumCpp.NdArrayBool(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    mask = np.random.randint(0, 2, [shape.rows, shape.cols]).astype(bool)
    cArray.setArray(data)
    cMask.setArray(mask)
    replaceValues = np.random.randint(0, 100, [shape.size() // 2, ])
    cReplaceValues = NumCpp.NdArray(1, replaceValues.size)
    cReplaceValues.setArray(replaceValues)

    assert np.array_equal(cArray.getNumpyArray(), data)

    NumCpp.place(cArray, cMask, cReplaceValues)
    assert not np.array_equal(cArray.getNumpyArray(), data)

    np.place(data, mask, replaceValues)
    assert np.array_equal(cArray.getNumpyArray(), data)


####################################################################################
def test_polar():
    components = np.random.rand(2).astype(float)
    assert NumCpp.polarScalar(components[0], components[1])

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    magArray = NumCpp.NdArray(shape)
    angleArray = NumCpp.NdArray(shape)
    mag = np.random.rand(shape.rows, shape.cols)
    angle = np.random.rand(shape.rows, shape.cols)
    magArray.setArray(mag)
    angleArray.setArray(angle)
    assert NumCpp.polarArray(magArray, angleArray) is not None


####################################################################################
def test_power():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    exponent = np.random.randint(0, 5, [1, ]).item()
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.powerArrayScalar(cArray, exponent), 9),
                          np.round(np.power(data, exponent), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    exponent = np.random.randint(0, 5, [1, ]).item()
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.powerArrayScalar(cArray, exponent), 9),
                          np.round(np.power(data, exponent), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    cExponents = NumCpp.NdArrayUInt8(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    exponents = np.random.randint(
        0, 5, [shape.rows, shape.cols]).astype(np.uint8)
    cArray.setArray(data)
    cExponents.setArray(exponents)
    assert np.array_equal(np.round(NumCpp.powerArrayArray(cArray, cExponents), 9),
                          np.round(np.power(data, exponents), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cExponents = NumCpp.NdArrayUInt8(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    exponents = np.random.randint(
        0, 5, [shape.rows, shape.cols]).astype(np.uint8)
    cArray.setArray(data)
    cExponents.setArray(exponents)
    assert np.array_equal(np.round(NumCpp.powerArrayArray(cArray, cExponents), 9),
                          np.round(np.power(data, exponents), 9))


####################################################################################
def test_powerf():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    exponent = np.random.rand(1).item() * 3
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.powerfArrayScalar(cArray, exponent), 9),
                          np.round(np.power(data, exponent), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    exponent = np.random.rand(1).item() * 3
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.powerfArrayScalar(cArray, exponent), 9),
                          np.round(np.power(data, exponent), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    cExponents = NumCpp.NdArray(shape)
    data = np.random.randint(0, 20, [shape.rows, shape.cols])
    exponents = np.random.rand(shape.rows, shape.cols) * 3
    cArray.setArray(data)
    cExponents.setArray(exponents)
    assert np.array_equal(np.round(NumCpp.powerfArrayArray(cArray, cExponents), 9),
                          np.round(np.power(data, exponents), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cExponents = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    exponents = np.random.rand(shape.rows, shape.cols) * \
        3 + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    cExponents.setArray(exponents)
    assert np.array_equal(np.round(NumCpp.powerfArrayArray(cArray, cExponents), 9),
                          np.round(np.power(data, exponents), 9))


####################################################################################
def test_prod():
    shapeInput = np.random.randint(2, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 5, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    assert NumCpp.prod(cArray, NumCpp.Axis.NONE).item() == data.prod()

    shapeInput = np.random.randint(2, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 5, [shape.rows, shape.cols])
    imag = np.random.randint(1, 5, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.prod(cArray, NumCpp.Axis.NONE).item() == data.prod()

    shapeInput = np.random.randint(2, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 5, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.prod(
        cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), data.prod(axis=0))

    shapeInput = np.random.randint(2, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 5, [shape.rows, shape.cols])
    imag = np.random.randint(1, 5, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.prod(
        cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), data.prod(axis=0))

    shapeInput = np.random.randint(2, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 5, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.prod(
        cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), data.prod(axis=1))

    shapeInput = np.random.randint(2, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 5, [shape.rows, shape.cols])
    imag = np.random.randint(1, 5, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.prod(
        cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), data.prod(axis=1))


####################################################################################
def test_proj():
    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert NumCpp.projScalar(value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cData = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cData.setArray(data)
    assert NumCpp.projArray(cData) is not None


####################################################################################
def test_ptp():
    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.ptp(
        cArray, NumCpp.Axis.NONE).getNumpyArray().item() == data.ptp()

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.ptp(
        cArray, NumCpp.Axis.NONE).getNumpyArray().item() == data.ptp()

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.ptp(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten().astype(np.uint32),
                          data.ptp(axis=0))

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.ptp(cArray, NumCpp.Axis.COL).getNumpyArray().flatten().astype(np.uint32),
                          data.ptp(axis=1))


####################################################################################
def test_put():
    # putFlat
    shapeInput = np.random.randint(2, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    randomIdx = np.random.randint(0, shapeInput.prod(), [1, ]).item()
    randomValue = np.random.randint(1, 500, [1, ]).item()
    NumCpp.put(cArray, randomIdx, randomValue)
    assert cArray.get(randomIdx) == randomValue

    shapeInput = np.random.randint(2, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    randomIdx = np.random.randint(0, shapeInput.prod(), [1, ]).item()
    randomValue = np.random.randint(1, 500, [1, ]).item() + 1j * np.random.randint(1, 500, [1, ]).item()
    NumCpp.put(cArray, randomIdx, randomValue)
    assert cArray.get(randomIdx) == randomValue

    # putRowCol
    shapeInput = np.random.randint(2, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    randomRowIdx = np.random.randint(0, shapeInput[0], [1, ]).item()
    randomColIdx = np.random.randint(0, shapeInput[1], [1, ]).item()
    randomValue = np.random.randint(1, 500, [1, ]).item()
    NumCpp.put(cArray, randomRowIdx, randomColIdx, randomValue)
    assert cArray.get(randomRowIdx, randomColIdx) == randomValue

    shapeInput = np.random.randint(2, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    randomRowIdx = np.random.randint(0, shapeInput[0], [1, ]).item()
    randomColIdx = np.random.randint(0, shapeInput[1], [1, ]).item()
    randomValue = np.random.randint(1, 500, [1, ]).item(
    ) + 1j * np.random.randint(1, 500, [1, ]).item()
    NumCpp.put(cArray, randomRowIdx, randomColIdx, randomValue)
    assert cArray.get(randomRowIdx, randomColIdx) == randomValue

    # putIndices1DValue
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    start = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stop = np.random.randint(start + 1, shapeInput[0], [1, ],).item()
    step = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    inputIndices = np.arange(start, stop, step).astype(np.int32)
    cIndices = NumCpp.NdArrayInt32(1, inputIndices.size)
    cIndices.setArray(inputIndices)
    randomValue = np.random.randint(1, 500, [1, ]).item()
    NumCpp.put(cArray, cIndices, randomValue)
    assert np.all(cArray.get(cIndices).astype(np.uint32) == randomValue)

    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    start = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stop = np.random.randint(start + 1, shapeInput[0], [1, ],).item()
    step = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    inputIndices = np.arange(start, stop, step).astype(np.int32)
    cIndices = NumCpp.NdArrayInt32(1, inputIndices.size)
    cIndices.setArray(inputIndices)
    randomValue = np.random.randint(1, 500, [1, ]).item(
    ) + 1j * np.random.randint(1, 500, [1, ]).item()
    NumCpp.put(cArray, cIndices, randomValue)
    assert np.all(cArray.get(cIndices) == randomValue)

    # putIndices1DValues
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    start = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stop = np.random.randint(start + 1, shapeInput[0], [1, ],).item()
    step = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    inputIndices = np.arange(start, stop, step).astype(np.int32)
    cIndices = NumCpp.NdArrayInt32(1, inputIndices.size)
    cIndices.setArray(inputIndices)
    randomValues = np.random.randint(
        1, 500, [inputIndices.size, ])
    NumCpp.put(cArray, cIndices, randomValues)
    assert np.array_equal(cArray.get(
        cIndices).flatten().astype(np.uint32), randomValues)

    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    start = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stop = np.random.randint(start + 1, shapeInput[0], [1, ],).item()
    step = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    inputIndices = np.arange(start, stop, step).astype(np.int32)
    cIndices = NumCpp.NdArrayInt32(1, inputIndices.size)
    cIndices.setArray(inputIndices)
    randomValues = np.random.randint(1, 500, [inputIndices.size, ]) + 1j * \
        np.random.randint(1, 500, [inputIndices.size, ])
    NumCpp.put(cArray, cIndices, randomValues)
    assert np.array_equal(cArray.get(cIndices).flatten(), randomValues)

    # putSlice1DValue
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    start = np.random.randint(0, shapeInput.prod() // 4, [1, ]).item()
    stop = np.random.randint(start + 1, shapeInput.prod(), [1, ]).item()
    step = np.random.randint(1, shapeInput.prod() // 10, [1, ]).item()
    randomValue = np.random.randint(1, 500, [1, ]).item()
    inputSlice = NumCpp.Slice(start, stop, step)
    NumCpp.put(cArray, inputSlice, randomValue)
    assert np.all(cArray.get(inputSlice).astype(np.uint32) == randomValue)

    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    start = np.random.randint(0, shapeInput.prod() // 4, [1, ]).item()
    stop = np.random.randint(start + 1, shapeInput.prod(), [1, ]).item()
    step = np.random.randint(1, shapeInput.prod() // 10, [1, ]).item()
    randomValue = np.random.randint(1, 500, [1, ]).item(
    ) + 1j * np.random.randint(1, 500, [1, ]).item()
    inputSlice = NumCpp.Slice(start, stop, step)
    NumCpp.put(cArray, inputSlice, randomValue)
    assert np.all(cArray.get(inputSlice) == randomValue)

    # putSlice1DValues
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    start = np.random.randint(0, shapeInput.prod() // 4, [1, ]).item()
    stop = np.random.randint(start + 1, shapeInput.prod(), [1, ]).item()
    step = np.random.randint(1, shapeInput.prod() // 10, [1, ]).item()
    inputSlice = NumCpp.Slice(start, stop, step)
    randomValues = np.random.randint(
        1, 500, [inputSlice.numElements(cArray.size()), ])
    NumCpp.put(cArray, inputSlice, randomValues)
    assert np.array_equal(cArray.get(
        inputSlice).flatten().astype(np.uint32), randomValues)

    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    start = np.random.randint(0, shapeInput.prod() // 4, [1, ]).item()
    stop = np.random.randint(start + 1, shapeInput.prod(), [1, ]).item()
    step = np.random.randint(1, shapeInput.prod() // 10, [1, ]).item()
    inputSlice = NumCpp.Slice(start, stop, step)
    randomValues = np.random.randint(1, 500, [inputSlice.numElements(cArray.size()), ]) + 1j * \
        np.random.randint(1, 500, [inputSlice.numElements(cArray.size()), ])
    NumCpp.put(cArray, inputSlice, randomValues)
    assert np.array_equal(cArray.get(inputSlice).flatten(), randomValues)

    # putIndices2DValue
    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ],).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ],).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ],).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ],).item()
    inputRowIndices = np.arange(startRow, stopRow, stepRow).astype(np.int32)
    inputColIndices = np.arange(startCol, stopCol, stepCol).astype(np.int32)
    cRowIndices = NumCpp.NdArrayInt32(1, inputRowIndices.size)
    cColIndices = NumCpp.NdArrayInt32(1, inputColIndices.size)
    cRowIndices.setArray(inputRowIndices)
    cColIndices.setArray(inputColIndices)
    randomValue = np.random.randint(1, 500, [1, ],).item()
    NumCpp.put(cArray, cRowIndices, cColIndices, randomValue)
    assert np.all(
        cArray.get(cRowIndices, cColIndices).astype(np.uint32) == randomValue)

    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ],).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ],).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ],).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ],).item()
    inputRowIndices = np.arange(startRow, stopRow, stepRow).astype(np.int32)
    inputColIndices = np.arange(startCol, stopCol, stepCol).astype(np.int32)
    cRowIndices = NumCpp.NdArrayInt32(1, inputRowIndices.size)
    cColIndices = NumCpp.NdArrayInt32(1, inputColIndices.size)
    cRowIndices.setArray(inputRowIndices)
    cColIndices.setArray(inputColIndices)
    randomValue = (np.random.randint(1, 500, [1, ],).item() + 1j * np.random.randint(1, 500, [1, ],).item())
    NumCpp.put(cArray, cRowIndices, cColIndices, randomValue)
    assert np.all(cArray.get(cRowIndices, cColIndices) == randomValue)

    # putRowIndicesColSliceValue
    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ],).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ],).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ],).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ],).item()
    inputRowIndices = np.arange(startRow, stopRow, stepRow).astype(np.int32)
    inputColSlice = NumCpp.Slice(startCol, stopCol, stepCol)
    cRowIndices = NumCpp.NdArrayInt32(1, inputRowIndices.size)
    cRowIndices.setArray(inputRowIndices)
    randomValue = np.random.randint(1, 500, [1, ],).item()
    NumCpp.put(cArray, cRowIndices, inputColSlice, randomValue)
    assert np.all(
        cArray.get(cRowIndices, inputColSlice).astype(np.uint32) == randomValue)

    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ],).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ],).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ],).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ],).item()
    inputRowIndices = np.arange(startRow, stopRow, stepRow).astype(np.int32)
    inputColSlice = NumCpp.Slice(startCol, stopCol, stepCol)
    cRowIndices = NumCpp.NdArrayInt32(1, inputRowIndices.size)
    cRowIndices.setArray(inputRowIndices)
    randomValue = (np.random.randint(1, 500, [1, ],).item() + 1j * np.random.randint(1, 500, [1, ],).item())
    NumCpp.put(cArray, cRowIndices, inputColSlice, randomValue)
    assert np.all(cArray.get(cRowIndices, inputColSlice) == randomValue)

    # putRowSliceColIndicesValue
    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ],).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ],).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ],).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ],).item()
    inputRowSlice = NumCpp.Slice(startRow, stopRow, stepRow)
    inputColIndices = np.arange(startCol, stopCol, stepCol).astype(np.int32)
    cColIndices = NumCpp.NdArrayInt32(1, inputColIndices.size)
    cColIndices.setArray(inputColIndices)
    randomValue = np.random.randint(1, 500, [1, ],).item()
    NumCpp.put(cArray, inputRowSlice, cColIndices, randomValue)
    assert np.all(
        cArray.get(inputRowSlice, cColIndices).astype(np.uint32) == randomValue)

    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ],).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ],).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ],).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ],).item()
    inputRowSlice = NumCpp.Slice(startRow, stopRow, stepRow)
    inputColIndices = np.arange(startCol, stopCol, stepCol).astype(np.int32)
    cColIndices = NumCpp.NdArrayInt32(1, inputColIndices.size)
    cColIndices.setArray(inputColIndices)
    randomValue = (np.random.randint(1, 500, [1, ],).item() + 1j * np.random.randint(1, 500, [1, ],).item())
    NumCpp.put(cArray, inputRowSlice, cColIndices, randomValue)
    assert np.all(cArray.get(inputRowSlice, cColIndices) == randomValue)

    # putSlice2DValue
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ]).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ]).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ]).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ]).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ]).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ]).item()
    inputRowSlice = NumCpp.Slice(startRow, stopRow, stepRow)
    inputColSlice = NumCpp.Slice(startCol, stopCol, stepCol)
    randomValue = np.random.randint(1, 500, [1, ]).item()
    NumCpp.put(cArray, inputRowSlice, inputColSlice, randomValue)
    assert np.all(cArray.get(inputRowSlice, inputColSlice).astype(
        np.uint32) == randomValue)

    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ]).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ]).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ]).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ]).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ]).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ]).item()
    inputRowSlice = NumCpp.Slice(startRow, stopRow, stepRow)
    inputColSlice = NumCpp.Slice(startCol, stopCol, stepCol)
    randomValue = np.random.randint(1, 500, [1, ]).item(
    ) + 1j * np.random.randint(1, 500, [1, ]).item()
    NumCpp.put(cArray, inputRowSlice, inputColSlice, randomValue)
    assert np.all(cArray.get(inputRowSlice, inputColSlice) == randomValue)

    # putIndices2DValueRow
    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    rowIdx = np.random.randint(0, shapeInput[0], [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ],).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ],).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ],).item()
    inputColIndices = np.arange(startCol, stopCol, stepCol).astype(np.int32)
    cColIndices = NumCpp.NdArrayInt32(1, inputColIndices.size)
    cColIndices.setArray(inputColIndices)
    randomValue = np.random.randint(1, 500, [1, ],).item()
    NumCpp.put(cArray, rowIdx, cColIndices, randomValue)
    assert np.all(
        cArray.get(rowIdx, cColIndices).astype(np.uint32) == randomValue)

    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    rowIdx = np.random.randint(0, shapeInput[0], [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ],).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ],).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ],).item()
    inputColIndices = np.arange(startCol, stopCol, stepCol).astype(np.int32)
    cColIndices = NumCpp.NdArrayInt32(1, inputColIndices.size)
    cColIndices.setArray(inputColIndices)
    randomValue = (np.random.randint(1, 500, [1, ],).item() + 1j * np.random.randint(1, 500, [1, ],).item())
    NumCpp.put(cArray, rowIdx, cColIndices, randomValue)
    assert np.all(cArray.get(rowIdx, cColIndices) == randomValue)

    # putSlice2DValueRow
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    rowIdx = np.random.randint(0, shapeInput[0], [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ]).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ]).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ]).item()
    inputColSlice = NumCpp.Slice(startCol, stopCol, stepCol)
    randomValue = np.random.randint(1, 500, [1, ]).item()
    NumCpp.put(cArray, rowIdx, inputColSlice, randomValue)
    assert np.all(cArray.get(rowIdx, inputColSlice).astype(
        np.uint32) == randomValue)

    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    rowIdx = np.random.randint(0, shapeInput[0], [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ]).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ]).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ]).item()
    inputColSlice = NumCpp.Slice(startCol, stopCol, stepCol)
    randomValue = np.random.randint(1, 500, [1, ]).item(
    ) + 1j * np.random.randint(1, 500, [1, ]).item()
    NumCpp.put(cArray, rowIdx, inputColSlice, randomValue)
    assert np.all(cArray.get(rowIdx, inputColSlice) == randomValue)

    # putIndices2DValueCol
    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ],).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    colIdx = np.random.randint(0, shapeInput[1], [1, ],).item()
    inputRowIndices = np.arange(startRow, stopRow, stepRow).astype(np.int32)
    cRowIndices = NumCpp.NdArrayInt32(1, inputRowIndices.size)
    cRowIndices.setArray(inputRowIndices)
    randomValue = np.random.randint(1, 500, [1, ],).item()
    NumCpp.put(cArray, cRowIndices, colIdx, randomValue)
    assert np.all(
        cArray.get(cRowIndices, colIdx).astype(np.uint32) == randomValue)

    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ],).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    colIdx = np.random.randint(0, shapeInput[1], [1, ],).item()
    inputRowIndices = np.arange(startRow, stopRow, stepRow).astype(np.int32)
    cRowIndices = NumCpp.NdArrayInt32(1, inputRowIndices.size)
    cRowIndices.setArray(inputRowIndices)
    randomValue = (np.random.randint(1, 500, [1, ],).item() + 1j * np.random.randint(1, 500, [1, ],).item())
    NumCpp.put(cArray, cRowIndices, colIdx, randomValue)
    assert np.all(cArray.get(cRowIndices, colIdx) == randomValue)

    # putSlice2DValueCol
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ]).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ]).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ]).item()
    colIdx = np.random.randint(0, shapeInput[1], [1, ],).item()
    inputRowSlice = NumCpp.Slice(startRow, stopRow, stepRow)
    randomValue = np.random.randint(1, 500, [1, ]).item()
    NumCpp.put(cArray, inputRowSlice, colIdx, randomValue)
    assert np.all(cArray.get(inputRowSlice, colIdx).astype(
        np.uint32) == randomValue)

    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ]).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ]).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ]).item()
    colIdx = np.random.randint(0, shapeInput[1], [1, ],).item()
    inputRowSlice = NumCpp.Slice(startRow, stopRow, stepRow)
    randomValue = np.random.randint(1, 500, [1, ]).item(
    ) + 1j * np.random.randint(1, 500, [1, ]).item()
    NumCpp.put(cArray, inputRowSlice, colIdx, randomValue)
    assert np.all(cArray.get(inputRowSlice, colIdx) == randomValue)

    # putIndices2DValues
    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ],).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ],).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ],).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ],).item()
    inputRowIndices = np.arange(startRow, stopRow, stepRow).astype(np.int32)
    inputColIndices = np.arange(startCol, stopCol, stepCol).astype(np.int32)
    cRowIndices = NumCpp.NdArrayInt32(1, inputRowIndices.size)
    cColIndices = NumCpp.NdArrayInt32(1, inputColIndices.size)
    cRowIndices.setArray(inputRowIndices)
    cColIndices.setArray(inputColIndices)
    randomValues = np.random.randint(1, 500, [inputRowIndices.size, inputColIndices.size])
    NumCpp.put(cArray, cRowIndices, cColIndices, randomValues)
    assert np.array_equal(
        cArray.get(cRowIndices, cColIndices).astype(np.uint32), randomValues)

    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ],).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ],).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ],).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ],).item()
    inputRowIndices = np.arange(startRow, stopRow, stepRow).astype(np.int32)
    inputColIndices = np.arange(startCol, stopCol, stepCol).astype(np.int32)
    cRowIndices = NumCpp.NdArrayInt32(1, inputRowIndices.size)
    cColIndices = NumCpp.NdArrayInt32(1, inputColIndices.size)
    cRowIndices.setArray(inputRowIndices)
    cColIndices.setArray(inputColIndices)
    randomValues = np.random.randint(1, 500, [inputRowIndices.size, inputColIndices.size])
    NumCpp.put(cArray, cRowIndices, cColIndices, randomValues)
    assert np.array_equal(cArray.get(cRowIndices, cColIndices), randomValues)

    # putRowIndicesColSliceValues
    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ],).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ],).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ],).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ],).item()
    inputRowIndices = np.arange(startRow, stopRow, stepRow).astype(np.int32)
    inputColSlice = NumCpp.Slice(startCol, stopCol, stepCol)
    cRowIndices = NumCpp.NdArrayInt32(1, inputRowIndices.size)
    cRowIndices.setArray(inputRowIndices)
    randomValues = np.random.randint(1, 500, [inputRowIndices.size, inputColSlice.numElements(shape.cols)])
    NumCpp.put(cArray, cRowIndices, inputColSlice, randomValues)
    assert np.array_equal(
        cArray.get(cRowIndices, inputColSlice).astype(np.uint32), randomValues)

    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ],).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ],).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ],).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ],).item()
    inputRowIndices = np.arange(startRow, stopRow, stepRow).astype(np.int32)
    inputColSlice = NumCpp.Slice(startCol, stopCol, stepCol)
    cRowIndices = NumCpp.NdArrayInt32(1, inputRowIndices.size)
    cRowIndices.setArray(inputRowIndices)
    randomValues = np.random.randint(1, 500, [inputRowIndices.size, inputColSlice.numElements(shape.cols)])
    NumCpp.put(cArray, cRowIndices, inputColSlice, randomValues)
    assert np.array_equal(cArray.get(cRowIndices, inputColSlice), randomValues)

    # putRowSliceColIndicesValues
    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ],).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ],).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ],).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ],).item()
    inputRowSlice = NumCpp.Slice(startRow, stopRow, stepRow)
    inputColIndices = np.arange(startCol, stopCol, stepCol).astype(np.int32)
    cColIndices = NumCpp.NdArrayInt32(1, inputColIndices.size)
    cColIndices.setArray(inputColIndices)
    randomValues = np.random.randint(1, 500, [inputRowSlice.numElements(shape.rows), inputColIndices.size])
    NumCpp.put(cArray, inputRowSlice, cColIndices, randomValues)
    assert np.array_equal(
        cArray.get(inputRowSlice, cColIndices).astype(np.uint32), randomValues)

    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ],).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ],).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ],).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ],).item()
    inputRowSlice = NumCpp.Slice(startRow, stopRow, stepRow)
    inputColIndices = np.arange(startCol, stopCol, stepCol).astype(np.int32)
    cColIndices = NumCpp.NdArrayInt32(1, inputColIndices.size)
    cColIndices.setArray(inputColIndices)
    randomValues = np.random.randint(1, 500, [inputRowSlice.numElements(shape.rows), inputColIndices.size])
    NumCpp.put(cArray, inputRowSlice, cColIndices, randomValues)
    assert np.array_equal(cArray.get(inputRowSlice, cColIndices), randomValues)

    # putSlice2DValues
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ]).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ]).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ]).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ]).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ]).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ]).item()
    inputRowSlice = NumCpp.Slice(startRow, stopRow, stepRow)
    inputColSlice = NumCpp.Slice(startCol, stopCol, stepCol)
    randomValues = np.random.randint(1, 500, [inputRowSlice.numElements(shape.rows),
                                              inputColSlice.numElements(shape.cols)])
    NumCpp.put(cArray, inputRowSlice, inputColSlice, randomValues)
    assert np.array_equal(cArray.get(
        inputRowSlice, inputColSlice).astype(np.uint32), randomValues)

    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ]).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ]).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ]).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ]).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ]).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ]).item()
    inputRowSlice = NumCpp.Slice(startRow, stopRow, stepRow)
    inputColSlice = NumCpp.Slice(startCol, stopCol, stepCol)
    valueShape = [inputRowSlice.numElements(
        shape.rows), inputColSlice.numElements(shape.cols)]
    randomValues = np.random.randint(
        1, 500, valueShape) + 1j * np.random.randint(1, 500, valueShape)
    NumCpp.put(cArray, inputRowSlice, inputColSlice, randomValues)
    assert np.array_equal(cArray.get(
        inputRowSlice, inputColSlice), randomValues)

    # putIndices2DValuesRow
    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    idxRow = np.random.randint(0, shapeInput[0], [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ],).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ],).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ],).item()
    inputColIndices = np.arange(startCol, stopCol, stepCol).astype(np.int32)
    cColIndices = NumCpp.NdArrayInt32(1, inputColIndices.size)
    cColIndices.setArray(inputColIndices)
    randomValues = np.random.randint(1, 500, [1, inputColIndices.size])
    NumCpp.put(cArray, idxRow, cColIndices, randomValues)
    assert np.array_equal(
        cArray.get(idxRow, cColIndices).astype(np.uint32), randomValues)

    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    idxRow = np.random.randint(0, shapeInput[0], [1, ],).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ],).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ],).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ],).item()
    inputColIndices = np.arange(startCol, stopCol, stepCol).astype(np.int32)
    cColIndices = NumCpp.NdArrayInt32(1, inputColIndices.size)
    cColIndices.setArray(inputColIndices)
    randomValues = np.random.randint(1, 500, [1, inputColIndices.size])
    NumCpp.put(cArray, idxRow, cColIndices, randomValues)
    assert np.array_equal(cArray.get(idxRow, cColIndices), randomValues)

    # putSlice2DValuesRow
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    idxRow = np.random.randint(0, shapeInput[0], [1, ]).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ]).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ]).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ]).item()
    inputColSlice = NumCpp.Slice(startCol, stopCol, stepCol)
    randomValues = np.random.randint(1, 500, [1, inputColSlice.numElements(shape.cols)])
    NumCpp.put(cArray, idxRow, inputColSlice, randomValues)
    assert np.array_equal(cArray.get(
        idxRow, inputColSlice).astype(np.uint32), randomValues)

    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    idxRow = np.random.randint(0, shapeInput[0], [1, ]).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ]).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ]).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ]).item()
    inputColSlice = NumCpp.Slice(startCol, stopCol, stepCol)
    valueShape = [1, inputColSlice.numElements(shape.cols)]
    randomValues = np.random.randint(
        1, 500, valueShape) + 1j * np.random.randint(1, 500, valueShape)
    NumCpp.put(cArray, idxRow, inputColSlice, randomValues)
    assert np.array_equal(cArray.get(
        idxRow, inputColSlice), randomValues)

    # putIndices2DValuesCol
    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ],).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    idxCol = np.random.randint(0, shapeInput[1], [1, ],).item()
    inputRowIndices = np.arange(startRow, stopRow, stepRow).astype(np.int32)
    cRowIndices = NumCpp.NdArrayInt32(1, inputRowIndices.size)
    cRowIndices.setArray(inputRowIndices)
    randomValues = np.random.randint(1, 500, [inputRowIndices.size, 1])
    NumCpp.put(cArray, cRowIndices, idxCol, randomValues)
    assert np.array_equal(
        cArray.get(cRowIndices, idxCol).astype(np.uint32), randomValues)

    shapeInput = np.random.randint(100, 500, [2, ],)
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ],).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ],).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ],).item()
    idxCol = np.random.randint(0, shapeInput[1], [1, ],).item()
    inputRowIndices = np.arange(startRow, stopRow, stepRow).astype(np.int32)
    cRowIndices = NumCpp.NdArrayInt32(1, inputRowIndices.size)
    cRowIndices.setArray(inputRowIndices)
    randomValues = np.random.randint(1, 500, [inputRowIndices.size, 1])
    NumCpp.put(cArray, cRowIndices, idxCol, randomValues)
    assert np.array_equal(cArray.get(cRowIndices, idxCol), randomValues)

    # putSlice2DValuesCol
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ]).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ]).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ]).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1, ]).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1, ]).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1, ]).item()
    inputRowSlice = NumCpp.Slice(startRow, stopRow, stepRow)
    inputColSlice = NumCpp.Slice(startCol, stopCol, stepCol)
    randomValues = np.random.randint(1, 500, [inputRowSlice.numElements(shape.rows),
                                              inputColSlice.numElements(shape.cols)])
    NumCpp.put(cArray, inputRowSlice, inputColSlice, randomValues)
    assert np.array_equal(cArray.get(
        inputRowSlice, inputColSlice).astype(np.uint32), randomValues)

    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1, ]).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1, ]).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1, ]).item()
    idxCol = np.random.randint(0, shapeInput[1], [1, ]).item()
    inputRowSlice = NumCpp.Slice(startRow, stopRow, stepRow)
    valueShape = [inputRowSlice.numElements(
        shape.rows), 1]
    randomValues = np.random.randint(
        1, 500, valueShape) + 1j * np.random.randint(1, 500, valueShape)
    NumCpp.put(cArray, inputRowSlice, idxCol, randomValues)
    assert np.array_equal(cArray.get(
        inputRowSlice, idxCol), randomValues)


####################################################################################
def test_rad2deg():
    value = np.abs(np.random.rand(1).item()) * 2 * np.pi
    assert np.round(NumCpp.rad2degScalar(value),
                    8) == np.round(np.rad2deg(value), 8)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 2 * np.pi
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.rad2degArray(
        cArray), 8), np.round(np.rad2deg(data), 8))


####################################################################################
def test_radians():
    value = np.abs(np.random.rand(1).item()) * 360
    assert np.round(NumCpp.radiansScalar(value),
                    9) == np.round(np.radians(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 360
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.radiansArray(
        cArray), 9), np.round(np.radians(data), 9))


####################################################################################
def test_ravel():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    cArray2 = NumCpp.ravel(cArray)
    assert np.array_equal(cArray2.getNumpyArray().flatten(), np.ravel(data))


####################################################################################
def test_real():
    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.realScalar(value), 9) == np.round(np.real(value), 9)  # noqa

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.realArray(
        cArray), 9), np.round(np.real(data), 9))


####################################################################################
def test_reciprocal():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.reciprocal(
        cArray), 9), np.round(np.reciprocal(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.reciprocal(
        cArray), 9), np.round(np.reciprocal(data), 9))


####################################################################################
def test_remainder():
    # numpy and cmath remainders are calculated differently, so convert for testing purposes
    values = np.random.rand(2) * 100
    values = np.sort(values)
    res = NumCpp.remainderScalar(values[1].item(), values[0].item())
    if res < 0:
        res += values[0].item()
    assert np.round(res, 9) == np.round(np.remainder(values[1], values[0]), 9)

    # numpy and cmath remainders are calculated differently, so convert for testing purposes
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.rand(shape.rows, shape.cols) * 100 + 10
    data2 = data1 - np.random.rand(shape.rows, shape.cols) * 10
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    res = NumCpp.remainderArray(cArray1, cArray2)
    res[res < 0] = res[res < 0] + data2[res < 0]
    assert np.array_equal(np.round(res, 9), np.round(
        np.remainder(data1, data2), 9))


####################################################################################
def test_replace():
    shapeInput = np.random.randint(2, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    oldValue = np.random.randint(1, 100, 1).item()
    newValue = np.random.randint(1, 100, 1).item()
    dataCopy = data.copy()
    dataCopy[dataCopy == oldValue] = newValue
    assert np.array_equal(NumCpp.replace(cArray, oldValue, newValue), dataCopy)

    shapeInput = np.random.randint(2, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    oldValue = np.random.randint(1, 100, 1).item(
    ) + 1j * np.random.randint(1, 100, 1).item()
    newValue = np.random.randint(1, 100, 1).item(
    ) + 1j * np.random.randint(1, 100, 1).item()
    dataCopy = data.copy()
    dataCopy[dataCopy == oldValue] = newValue
    assert np.array_equal(NumCpp.replace(cArray, oldValue, newValue), dataCopy)


####################################################################################
def test_reshape():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    newShape = data.size
    NumCpp.reshape(cArray, newShape)
    assert np.array_equal(cArray.getNumpyArray(), data.reshape(1, newShape))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    newShape = NumCpp.Shape(shapeInput[1].item(), shapeInput[0].item())
    NumCpp.reshape(cArray, newShape)
    assert np.array_equal(cArray.getNumpyArray(),
                          data.reshape(shapeInput[::-1]))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    newShape = NumCpp.Shape(shapeInput[1].item(), shapeInput[0].item())
    NumCpp.reshapeList(cArray, newShape)
    assert np.array_equal(cArray.getNumpyArray(),
                          data.reshape(shapeInput[::-1]))

    shapeInput = np.random.randint(2, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    newNumCols = np.random.choice(np.array(list(factors(data.size))), 1).item()
    NumCpp.reshape(cArray, -1, newNumCols)
    assert np.array_equal(cArray.getNumpyArray(), data.reshape(-1, newNumCols))

    shapeInput = np.random.randint(2, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    newNumRows = np.random.choice(np.array(list(factors(data.size))), 1).item()
    NumCpp.reshape(cArray, newNumRows, -1)
    assert np.array_equal(cArray.getNumpyArray(), data.reshape(newNumRows, -1))


####################################################################################
def test_resize():
    shapeInput1 = np.random.randint(1, 100, [2, ])
    shapeInput2 = np.random.randint(1, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput1[0].item(), shapeInput1[1].item())
    shape2 = NumCpp.Shape(shapeInput2[0].item(), shapeInput2[1].item())
    cArray = NumCpp.NdArray(shape1)
    data = np.random.randint(
        1, 100, [shape1.rows, shape1.cols], dtype=np.uint32)
    cArray.setArray(data)
    NumCpp.resizeFast(cArray, shape2)
    assert cArray.shape().rows == shape2.rows
    assert cArray.shape().cols == shape2.cols

    shapeInput1 = np.random.randint(1, 100, [2, ])
    shapeInput2 = np.random.randint(1, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput1[0].item(), shapeInput1[1].item())
    shape2 = NumCpp.Shape(shapeInput2[0].item(), shapeInput2[1].item())
    cArray = NumCpp.NdArray(shape1)
    data = np.random.randint(
        1, 100, [shape1.rows, shape1.cols], dtype=np.uint32)
    cArray.setArray(data)
    NumCpp.resizeSlow(cArray, shape2)
    assert cArray.shape().rows == shape2.rows
    assert cArray.shape().cols == shape2.cols


####################################################################################
def test_right_shift():
    shapeInput = np.random.randint(20, 100, [2, ])
    bitsToshift = np.random.randint(1, 32, [1, ]).item()
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(1, np.iinfo(np.uint32).max, [
                             shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.right_shift(cArray, bitsToshift).getNumpyArray(),
                          np.right_shift(data, bitsToshift))


####################################################################################
def test_rint():
    value = np.abs(np.random.rand(1).item()) * 2 * np.pi
    assert NumCpp.rintScalar(value) == np.rint(value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 2 * np.pi
    cArray.setArray(data)
    assert np.array_equal(NumCpp.rintArray(cArray), np.rint(data))


####################################################################################
def test_rms():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert (np.round(NumCpp.rms(cArray, NumCpp.Axis.NONE).getNumpyArray().flatten().item(), 9) ==
            np.round(np.sqrt(np.mean(np.square(data), axis=None)).item(), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert (np.round(NumCpp.rms(cArray, NumCpp.Axis.NONE).getNumpyArray().flatten().item(), 9) ==
            np.round(np.sqrt(np.mean(np.square(data), axis=None)).item(), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.rms(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), 9),
                          np.round(np.sqrt(np.mean(np.square(data), axis=0)), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.rms(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), 9),
                          np.round(np.sqrt(np.mean(np.square(data), axis=0)), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.rms(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), 9),
                          np.round(np.sqrt(np.mean(np.square(data), axis=1)), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.rms(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), 9),
                          np.round(np.sqrt(np.mean(np.square(data), axis=1)), 9))


####################################################################################
def test_roll():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    amount = np.random.randint(0, data.size, [1, ]).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.roll(cArray, amount, NumCpp.Axis.NONE).getNumpyArray(),
                          np.roll(data, amount, axis=None))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    amount = np.random.randint(0, shape.cols, [1, ]).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.roll(cArray, amount, NumCpp.Axis.ROW).getNumpyArray(),
                          np.roll(data, amount, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    amount = np.random.randint(0, shape.rows, [1, ]).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.roll(cArray, amount, NumCpp.Axis.COL).getNumpyArray(),
                          np.roll(data, amount, axis=1))


####################################################################################
def test_rot90():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    amount = np.random.randint(1, 4, [1, ]).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.rot90(
        cArray, amount).getNumpyArray(), np.rot90(data, amount))


####################################################################################
def test_round():
    value = np.abs(np.random.rand(1).item()) * 2 * np.pi
    assert NumCpp.roundScalar(value, 10) == np.round(value, 10)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 2 * np.pi
    cArray.setArray(data)
    assert np.array_equal(NumCpp.roundArray(cArray, 9), np.round(data, 9))


####################################################################################
def test_row_stack():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[0].item(
    ) + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
    shape3 = NumCpp.Shape(shapeInput[0].item(
    ) + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
    shape4 = NumCpp.Shape(shapeInput[0].item(
    ) + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    cArray3 = NumCpp.NdArray(shape3)
    cArray4 = NumCpp.NdArray(shape4)
    data1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    data4 = np.random.randint(1, 100, [shape4.rows, shape4.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    assert np.array_equal(NumCpp.row_stack(cArray1, cArray2, cArray3, cArray4),
                          np.row_stack([data1, data2, data3, data4]))


####################################################################################
def test_select():
    # vector of pointers
    shape = np.random.randint(10, 100, [2, ])
    numChoices = np.random.randint(1, 10)
    default = np.random.randint(100)

    condlist = []
    choicelist = []
    for i in range(numChoices):
        values = np.random.rand(*shape)
        condlist.append(values > np.percentile(values, 75))
        choicelist.append(np.random.randint(0, 100, shape).astype(float))

    assert np.array_equal(NumCpp.select(condlist, choicelist, default),
                          np.select(condlist, choicelist, default))

    # vector of arrays
    shape = np.random.randint(10, 100, [2, ])
    numChoices = np.random.randint(1, 10)
    default = np.random.randint(100)

    condlist = []
    choicelist = []
    for i in range(numChoices):
        values = np.random.rand(*shape)
        condlist.append(values > np.percentile(values, 75))
        choicelist.append(np.random.randint(0, 100, shape).astype(float))

    assert np.array_equal(NumCpp.selectVector(condlist, choicelist, default),
                          np.select(condlist, choicelist, default))

    # initializer list
    shape = np.random.randint(10, 100, [2, ])
    numChoices = 3
    default = np.random.randint(100)

    condlist = []
    choicelist = []
    for i in range(numChoices):
        values = np.random.rand(*shape)
        condlist.append(values > np.percentile(values, 75))
        choicelist.append(np.random.randint(0, 100, shape).astype(float))

    assert np.array_equal(NumCpp.select(*condlist, *choicelist, default),
                          np.select(condlist, choicelist, default))


####################################################################################
def test_setdiff1d():
    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt32(shape)
    cArray2 = NumCpp.NdArrayUInt32(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.setdiff1d(cArray1, cArray2).getNumpyArray().flatten(),
                          np.setdiff1d(data1, data2))

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.setdiff1d(cArray1, cArray2).getNumpyArray().flatten(),
                          np.setdiff1d(data1, data2))


####################################################################################
def test_shape():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    assert cArray.shape().rows == shape.rows and cArray.shape().cols == shape.cols


####################################################################################
def test_sign():
    value = np.random.randn(1).item() * 100
    assert NumCpp.signScalar(value) == np.sign(value)

    value = np.random.randn(1).item() * 100 + 1j * \
        np.random.randn(1).item() * 100
    assert NumCpp.signScalar(value) == np.sign(value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    assert np.array_equal(NumCpp.signArray(cArray), np.sign(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.signArray(cArray), np.sign(data))


####################################################################################
def test_signbit():
    value = np.random.randn(1).item() * 100
    assert NumCpp.signbitScalar(value) == np.signbit(value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    assert np.array_equal(NumCpp.signbitArray(cArray), np.signbit(data))


####################################################################################
def test_sin():
    value = np.random.randn(1).item()
    assert np.round(NumCpp.sinScalar(value), 9) == np.round(np.sin(value), 9)

    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.sinScalar(value), 9) == np.round(np.sin(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.sinArray(
        cArray), 9), np.round(np.sin(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.sinArray(
        cArray), 9), np.round(np.sin(data), 9))


####################################################################################
def test_sinc():
    value = np.random.randn(1)
    assert np.round(NumCpp.sincScalar(value.item()),
                    9) == np.round(np.sinc(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.sincArray(
        cArray), 9), np.round(np.sinc(data), 9))


####################################################################################
def test_sinh():
    value = np.random.randn(1).item()
    assert np.round(NumCpp.sinhScalar(value), 9) == np.round(np.sinh(value), 9)

    value = np.random.randn(1).item() + 1j * np.random.randn(1).item()
    assert np.round(NumCpp.sinhScalar(value), 9) == np.round(np.sinh(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.sinhArray(
        cArray), 9), np.round(np.sinh(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.randn(shape.rows, shape.cols) + \
        1j * np.random.randn(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.sinhArray(
        cArray), 9), np.round(np.sinh(data), 9))


####################################################################################
def test_size():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    assert cArray.size() == shapeInput.prod().item()


####################################################################################
def test_sort():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    d = data.flatten()
    d.sort()
    assert np.array_equal(NumCpp.sort(
        cArray, NumCpp.Axis.NONE).getNumpyArray().flatten(), d)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    d = data.flatten()
    d.sort()
    assert np.array_equal(NumCpp.sort(
        cArray, NumCpp.Axis.NONE).getNumpyArray().flatten(), d)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    pSorted = np.sort(data, axis=0)
    cSorted = NumCpp.sort(cArray, NumCpp.Axis.ROW).getNumpyArray()
    assert np.array_equal(cSorted, pSorted)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    pSorted = np.sort(data, axis=0)
    cSorted = NumCpp.sort(cArray, NumCpp.Axis.ROW).getNumpyArray()
    assert np.array_equal(cSorted, pSorted)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    pSorted = np.sort(data, axis=1)
    cSorted = NumCpp.sort(cArray, NumCpp.Axis.COL).getNumpyArray()
    assert np.array_equal(cSorted, pSorted)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    pSorted = np.sort(data, axis=1)
    cSorted = NumCpp.sort(cArray, NumCpp.Axis.COL).getNumpyArray()
    assert np.array_equal(cSorted, pSorted)


def test_split():
    shapeInput = np.random.randint(80, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    numSplits = np.random.randint(2, 5)
    indices = np.sort(np.random.choice(range(shape.rows), numSplits, replace=False))
    cIndices = NumCpp.NdArrayInt32(indices.size)
    cIndices.setArray(indices.astype(np.int32))
    result = np.split(data, indices, axis=0)
    cResult = NumCpp.split(cArray, cIndices, NumCpp.Axis.ROW)
    assert len(cResult) == len(result)
    for i in range(len(result)):
        assert np.array_equal(cResult[i], result[i])

    shapeInput = np.random.randint(80, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    numSplits = np.random.randint(2, 5)
    indices = np.sort(np.random.choice(range(shape.cols), numSplits, replace=False))
    cIndices = NumCpp.NdArrayInt32(indices.size)
    cIndices.setArray(indices.astype(np.int32))
    result = np.split(data, indices, axis=1)
    cResult = NumCpp.split(cArray, cIndices, NumCpp.Axis.COL)
    assert len(cResult) == len(result)
    for i in range(len(result)):
        assert np.array_equal(cResult[i], result[i])


####################################################################################
def test_sqrt():
    value = np.random.randint(1, 100, [1, ]).item()
    assert np.round(NumCpp.sqrtScalar(value), 9) == np.round(np.sqrt(value), 9)

    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.sqrtScalar(value), 9) == np.round(np.sqrt(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.sqrtArray(
        cArray), 9), np.round(np.sqrt(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.sqrtArray(
        cArray), 9), np.round(np.sqrt(data), 9))


####################################################################################
def test_square():
    value = np.random.randint(1, 100, [1, ]).item()
    assert np.round(NumCpp.squareScalar(value),
                    9) == np.round(np.square(value), 9)

    value = np.random.randint(1, 100, [1, ]).item(
    ) + 1j * np.random.randint(1, 100, [1, ]).item()
    assert np.round(NumCpp.squareScalar(value),
                    9) == np.round(np.square(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.squareArray(
        cArray), 9), np.round(np.square(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.squareArray(
        cArray), 9), np.round(np.square(data), 9))


####################################################################################
def test_stack():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    cArray3 = NumCpp.NdArray(shape)
    cArray4 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data3 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data4 = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    assert np.array_equal(NumCpp.stack(cArray1, cArray2, cArray3, cArray4, NumCpp.Axis.ROW),
                          np.vstack([data1, data2, data3, data4]))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    cArray3 = NumCpp.NdArray(shape)
    cArray4 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data3 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data4 = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    assert np.array_equal(NumCpp.stack(cArray1, cArray2, cArray3, cArray4, NumCpp.Axis.COL),
                          np.hstack([data1, data2, data3, data4]))


####################################################################################
def test_stdev():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.round(NumCpp.stdev(cArray, NumCpp.Axis.NONE).item(),
                    9) == np.round(np.std(data), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.stdev(cArray, NumCpp.Axis.NONE).item() is not None

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.stdev(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), 9),
                          np.round(np.std(data, axis=0), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.stdev(cArray, NumCpp.Axis.ROW) is not None

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.stdev(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), 9),
                          np.round(np.std(data, axis=1), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.stdev(cArray, NumCpp.Axis.COL) is not None


####################################################################################
def test_subtract():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.subtract(cArray1, cArray2), data1 - data2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    value = np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.subtract(cArray, value), data - value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    value = np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.subtract(value, cArray), value - data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.subtract(cArray1, cArray2), data1 - data2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(-100, 100) + 1j * np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.subtract(cArray, value), data - value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(-100, 100) + 1j * np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.subtract(value, cArray), value - data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArray(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.subtract(cArray1, cArray2), data1 - data2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.subtract(cArray1, cArray2), data1 - data2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    value = np.random.randint(-100, 100) + 1j * np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.subtract(cArray, value), data - value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    value = np.random.randint(-100, 100) + 1j * np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.subtract(value, cArray), value - data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.subtract(cArray, value), data - value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(-100, 100)
    assert np.array_equal(NumCpp.subtract(value, cArray), value - data)


####################################################################################
def test_sum():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.sum(cArray, NumCpp.Axis.NONE).item() == np.sum(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.sum(cArray, NumCpp.Axis.NONE).item() == np.sum(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.sum(
        cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), np.sum(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.sum(
        cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), np.sum(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.sum(
        cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), np.sum(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.sum(
        cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), np.sum(data, axis=1))


####################################################################################
def test_swap():
    shapeInput1 = np.random.randint(20, 100, [2, ])
    shapeInput2 = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput1[0].item(), shapeInput1[1].item())
    shape2 = NumCpp.Shape(shapeInput2[0].item(), shapeInput2[1].item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    data1 = np.random.randint(0, 100, [shape1.rows, shape1.cols]).astype(float)
    data2 = np.random.randint(0, 100, [shape2.rows, shape2.cols]).astype(float)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    NumCpp.swap(cArray1, cArray2)
    assert (np.array_equal(cArray1.getNumpyArray(), data2) and
            np.array_equal(cArray2.getNumpyArray(), data1))


####################################################################################
def test_swapaxes():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.swapaxes(cArray).getNumpyArray(), data.T)


####################################################################################
def test_swapRows():
    shapeInput = np.random.randint(10, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, shapeInput)
    cArray.setArray(data)
    rowIdx1 = np.random.randint(0, shape.rows)
    rowIdx2 = np.random.randint(0, shape.rows)
    cArrayNp = NumCpp.swapRows(cArray, rowIdx1, rowIdx2).getNumpyArray()
    assert np.array_equal(cArrayNp[rowIdx1, :], data[rowIdx2, :])
    assert np.array_equal(cArrayNp[rowIdx2, :], data[rowIdx1, :])


####################################################################################
def test_swapCols():
    shapeInput = np.random.randint(10, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, shapeInput)
    cArray.setArray(data)
    colIdx1 = np.random.randint(0, shape.cols)
    colIdx2 = np.random.randint(0, shape.cols)
    cArrayNp = NumCpp.swapCols(cArray, colIdx1, colIdx2).getNumpyArray()
    assert np.array_equal(cArrayNp[:, colIdx1], data[:, colIdx2])
    assert np.array_equal(cArrayNp[:, colIdx2], data[:, colIdx1])


####################################################################################
def test_tan():
    value = np.random.rand(1).item() * np.pi
    assert np.round(NumCpp.tanScalar(value), 9) == np.round(np.tan(value), 9)

    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.tanScalar(value), 9) == np.round(np.tan(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.tanArray(
        cArray), 9), np.round(np.tan(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.tanArray(
        cArray), 9), np.round(np.tan(data), 9))


####################################################################################
def test_take():
    shapeInput = np.random.randint(10, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, shapeInput)
    cArray.setArray(data)

    indices = np.random.choice(range(data.size), data.size // 5, replace=False).astype(np.uint32)
    cIndices = NumCpp.NdArrayUInt32(1, indices.size)
    cIndices.setArray(indices)
    assert np.array_equal(NumCpp.take(cArray, cIndices, NumCpp.Axis.NONE).flatten(),
                          np.take(data, indices).flatten())

    rowIndices = np.sort(np.random.choice(range(shape.rows), shape.rows // 2, replace=False).astype(np.uint32))
    cRowIndices = NumCpp.NdArrayUInt32(1, rowIndices.size)
    cRowIndices.setArray(rowIndices)
    assert np.array_equal(NumCpp.take(cArray, cRowIndices, NumCpp.Axis.ROW),
                          np.take(data, rowIndices, axis=0))

    colIndices = np.sort(np.random.choice(range(shape.cols), shape.cols // 2, replace=False).astype(np.uint32))
    cColIndices = NumCpp.NdArrayUInt32(1, colIndices.size)
    cColIndices.setArray(colIndices)
    assert np.array_equal(NumCpp.take(cArray, cColIndices, NumCpp.Axis.COL),
                          np.take(data, colIndices, axis=1))


####################################################################################
def test_tanh():
    value = np.random.rand(1).item() * np.pi
    assert np.round(NumCpp.tanhScalar(value), 9) == np.round(np.tanh(value), 9)

    components = np.random.rand(2).astype(float)
    value = complex(components[0], components[1])
    assert np.round(NumCpp.tanhScalar(value), 9) == np.round(np.tanh(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.tanhArray(
        cArray), 9), np.round(np.tanh(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * \
        np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.tanhArray(
        cArray), 9), np.round(np.tanh(data), 9))


####################################################################################
def test_tile():
    shapeInput = np.random.randint(1, 10, [2, ])
    shapeRepeat = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shapeR = NumCpp.Shape(shapeRepeat[0].item(), shapeRepeat[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.tileRectangle(
        cArray, shapeR.rows, shapeR.cols), np.tile(data, shapeRepeat))

    shapeInput = np.random.randint(1, 10, [2, ])
    shapeRepeat = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shapeR = NumCpp.Shape(shapeRepeat[0].item(), shapeRepeat[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.tileShape(
        cArray, shapeR), np.tile(data, shapeRepeat))


####################################################################################
def test_tofile():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    tempDir = tempfile.gettempdir()
    filename = os.path.join(tempDir, 'temp.bin')
    NumCpp.tofile(cArray, filename)
    assert os.path.exists(filename)
    data2 = np.fromfile(filename, float).reshape(shapeInput)
    assert np.array_equal(data, data2)
    os.remove(filename)

    # delimiter = ' '
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    tempDir = tempfile.gettempdir()
    filename = os.path.join(tempDir, 'temp.txt')
    NumCpp.tofile(cArray, filename, ' ')
    assert os.path.exists(filename)
    data2 = np.fromfile(filename, dtype=float, sep=' ').reshape(shapeInput)
    assert np.array_equal(data, data2)
    os.remove(filename)

    # delimiter = '\t'
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    tempDir = tempfile.gettempdir()
    filename = os.path.join(tempDir, 'temp.txt')
    NumCpp.tofile(cArray, filename, '\t')
    assert os.path.exists(filename)
    data2 = np.fromfile(filename, dtype=float, sep='\t').reshape(shapeInput)
    assert np.array_equal(data, data2)
    os.remove(filename)

    # delimiter = '\n'
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    tempDir = tempfile.gettempdir()
    filename = os.path.join(tempDir, 'temp.txt')
    NumCpp.tofile(cArray, filename, '\n')
    assert os.path.exists(filename)
    data2 = np.fromfile(filename, dtype=float, sep='\n').reshape(shapeInput)
    assert np.array_equal(data, data2)
    os.remove(filename)

    # delimiter = ','
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    tempDir = tempfile.gettempdir()
    filename = os.path.join(tempDir, 'temp.txt')
    NumCpp.tofile(cArray, filename, ',')
    assert os.path.exists(filename)
    data2 = np.fromfile(filename, dtype=float, sep=',').reshape(shapeInput)
    assert np.array_equal(data, data2)
    os.remove(filename)

    # delimiter = '|'
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    tempDir = tempfile.gettempdir()
    filename = os.path.join(tempDir, 'temp.txt')
    NumCpp.tofile(cArray, filename, '|')
    assert os.path.exists(filename)
    data2 = np.fromfile(filename, dtype=float, sep='|').reshape(shapeInput)
    assert np.array_equal(data, data2)
    os.remove(filename)


####################################################################################
def test_toStlVector():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    out = np.asarray(NumCpp.toStlVector(cArray))
    assert np.array_equal(out, data.flatten())


####################################################################################
def test_trace():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows, [1, ]).item()
    assert np.array_equal(NumCpp.trace(
        cArray, offset, NumCpp.Axis.ROW), data.trace(offset, axis1=1, axis2=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows, [1, ]).item()
    assert np.array_equal(NumCpp.trace(
        cArray, offset, NumCpp.Axis.COL), data.trace(offset, axis1=0, axis2=1))


####################################################################################
def test_transpose():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.transpose(
        cArray).getNumpyArray(), np.transpose(data))


####################################################################################
def test_trapz():
    shape = NumCpp.Shape(np.random.randint(10, 20, [1, ]).item(), 1)
    cArray = NumCpp.NdArray(shape)
    coeffs = np.random.randint(0, 10, [2, ])
    dx = np.random.rand(1).item()
    data = np.array([x ** 2 - coeffs[0] * x + coeffs[1]
                    for x in range(shape.size())])
    cArray.setArray(data)
    integralC = NumCpp.trapzDx(cArray, dx, NumCpp.Axis.NONE).item()
    integralPy = np.trapz(data, dx=dx)
    assert np.round(integralC, 8) == np.round(integralPy, 8)

    shape = NumCpp.Shape(np.random.randint(
        10, 20, [1, ]).item(), np.random.randint(10, 20, [1, ]).item())
    cArray = NumCpp.NdArray(shape)
    coeffs = np.random.randint(0, 10, [2, ])
    dx = np.random.rand(1).item()
    data = np.array([x ** 2 - coeffs[0] * x - coeffs[1]
                    for x in range(shape.size())]).reshape(shape.rows, shape.cols)
    cArray.setArray(data)
    integralC = NumCpp.trapzDx(cArray, dx, NumCpp.Axis.ROW).flatten()
    integralPy = np.trapz(data, dx=dx, axis=0)
    assert np.array_equal(np.round(integralC, 8), np.round(integralPy, 8))

    shape = NumCpp.Shape(np.random.randint(
        10, 20, [1, ]).item(), np.random.randint(10, 20, [1, ]).item())
    cArray = NumCpp.NdArray(shape)
    coeffs = np.random.randint(0, 10, [2, ])
    dx = np.random.rand(1).item()
    data = np.array([x ** 2 - coeffs[0] * x - coeffs[1]
                    for x in range(shape.size())]).reshape(shape.rows, shape.cols)
    cArray.setArray(data)
    integralC = NumCpp.trapzDx(cArray, dx, NumCpp.Axis.COL).flatten()
    integralPy = np.trapz(data, dx=dx, axis=1)
    assert np.array_equal(np.round(integralC, 8), np.round(integralPy, 8))

    shape = NumCpp.Shape(1, np.random.randint(10, 20, [1, ]).item())
    cArrayY = NumCpp.NdArray(shape)
    cArrayX = NumCpp.NdArray(shape)
    coeffs = np.random.randint(0, 10, [2, ])
    dx = np.random.rand(shape.rows, shape.cols)
    data = np.array([x ** 2 - coeffs[0] * x + coeffs[1]
                    for x in range(shape.size())])
    cArrayY.setArray(data)
    cArrayX.setArray(dx)
    integralC = NumCpp.trapz(cArrayY, cArrayX, NumCpp.Axis.NONE).item()
    integralPy = np.trapz(data, x=dx)
    assert np.round(integralC, 8) == np.round(integralPy, 8)

    shape = NumCpp.Shape(np.random.randint(
        10, 20, [1, ]).item(), np.random.randint(10, 20, [1, ]).item())
    cArrayY = NumCpp.NdArray(shape)
    cArrayX = NumCpp.NdArray(shape)
    coeffs = np.random.randint(0, 10, [2, ])
    dx = np.random.rand(shape.rows, shape.cols)
    data = np.array([x ** 2 - coeffs[0] * x + coeffs[1]
                    for x in range(shape.size())]).reshape(shape.rows, shape.cols)
    cArrayY.setArray(data)
    cArrayX.setArray(dx)
    integralC = NumCpp.trapz(cArrayY, cArrayX, NumCpp.Axis.ROW).flatten()
    integralPy = np.trapz(data, x=dx, axis=0)
    assert np.array_equal(np.round(integralC, 8), np.round(integralPy, 8))

    shape = NumCpp.Shape(np.random.randint(
        10, 20, [1, ]).item(), np.random.randint(10, 20, [1, ]).item())
    cArrayY = NumCpp.NdArray(shape)
    cArrayX = NumCpp.NdArray(shape)
    coeffs = np.random.randint(0, 10, [2, ])
    dx = np.random.rand(shape.rows, shape.cols)
    data = np.array([x ** 2 - coeffs[0] * x + coeffs[1]
                    for x in range(shape.size())]).reshape(shape.rows, shape.cols)
    cArrayY.setArray(data)
    cArrayX.setArray(dx)
    integralC = NumCpp.trapz(cArrayY, cArrayX, NumCpp.Axis.COL).flatten()
    integralPy = np.trapz(data, x=dx, axis=1)
    assert np.array_equal(np.round(integralC, 8), np.round(integralPy, 8))


####################################################################################
def test_tril():
    squareSize = np.random.randint(10, 100, [1, ]).item()
    offset = np.random.randint(0, squareSize // 2, [1, ]).item()
    assert np.array_equal(NumCpp.trilSquare(squareSize, offset),
                          np.tri(squareSize, k=offset))

    squareSize = np.random.randint(10, 100, [1, ]).item()
    offset = np.random.randint(0, squareSize // 2, [1, ]).item()
    assert np.array_equal(NumCpp.trilSquareComplex(squareSize, offset),
                          np.tri(squareSize, k=offset) + 1j * np.zeros([squareSize, squareSize]))

    shapeInput = np.random.randint(10, 100, [2, ])
    offset = np.random.randint(0, np.min(shapeInput) // 2, [1, ]).item()
    assert np.array_equal(NumCpp.trilRect(shapeInput[0].item(), shapeInput[1].item(), offset),
                          np.tri(shapeInput[0].item(), shapeInput[1].item(), k=offset))

    shapeInput = np.random.randint(10, 100, [2, ])
    offset = np.random.randint(0, np.min(shapeInput) // 2, [1, ]).item()
    assert np.array_equal(NumCpp.trilRectComplex(shapeInput[0].item(), shapeInput[1].item(), offset),
                          np.tri(shapeInput[0].item(), shapeInput[1].item(), k=offset) + 1j * np.zeros(shapeInput))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows // 2, [1, ]).item()
    assert np.array_equal(NumCpp.trilArray(cArray, offset),
                          np.tril(data, k=offset))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows // 2, [1, ]).item()
    assert np.array_equal(NumCpp.trilArray(cArray, offset),
                          np.tril(data, k=offset))


####################################################################################
def test_triu():
    squareSize = np.random.randint(10, 100, [1, ]).item()
    offset = np.random.randint(0, squareSize // 2, [1, ]).item()
    assert np.array_equal(NumCpp.triuSquare(squareSize, offset),
                          np.tri(squareSize, k=-offset).T)

    squareSize = np.random.randint(10, 100, [1, ]).item()
    offset = np.random.randint(0, squareSize // 2, [1, ]).item()
    assert np.array_equal(NumCpp.triuSquareComplex(squareSize, offset),
                          np.tri(squareSize, k=-offset).T + 1j * np.zeros([squareSize, squareSize]))

    shapeInput = np.random.randint(10, 100, [2, ])
    offset = np.random.randint(0, np.min(shapeInput), [1, ]).item()
    # NOTE: numpy triu appears to have a bug... just check that NumCpp runs without error
    assert NumCpp.triuRect(shapeInput[0].item(
    ), shapeInput[1].item(), offset) is not None

    shapeInput = np.random.randint(10, 100, [2, ])
    offset = np.random.randint(0, np.min(shapeInput), [1, ]).item()
    # NOTE: numpy triu appears to have a bug... just check that NumCpp runs without error
    assert NumCpp.triuRectComplex(
        shapeInput[0].item(), shapeInput[1].item(), offset) is not None

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows // 2, [1, ]).item()
    assert np.array_equal(NumCpp.triuArray(
        cArray, offset), np.triu(data, k=offset))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows // 2, [1, ]).item()
    assert np.array_equal(NumCpp.triuArray(
        cArray, offset), np.triu(data, k=offset))


####################################################################################
def test_trim_zeros():
    numElements = np.random.randint(50, 100, [1, ]).item()
    offsetBeg = np.random.randint(0, 10, [1, ]).item()
    offsetEnd = np.random.randint(10, numElements, [1, ]).item()
    shape = NumCpp.Shape(1, numElements)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    data[0, :offsetBeg] = 0
    data[0, -offsetEnd:] = 0
    cArray.setArray(data)
    assert np.array_equal(NumCpp.trim_zeros(cArray, 'f').getNumpyArray().flatten(),
                          np.trim_zeros(data.flatten(), 'f'))

    numElements = np.random.randint(50, 100, [1, ]).item()
    offsetBeg = np.random.randint(0, 10, [1, ]).item()
    offsetEnd = np.random.randint(10, numElements, [1, ]).item()
    shape = NumCpp.Shape(1, numElements)
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    data[0, :offsetBeg] = complex(0, 0)
    data[0, -offsetEnd:] = complex(0, 0)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.trim_zeros(cArray, 'f').getNumpyArray().flatten(),
                          np.trim_zeros(data.flatten(), 'f'))

    numElements = np.random.randint(50, 100, [1, ]).item()
    offsetBeg = np.random.randint(0, 10, [1, ]).item()
    offsetEnd = np.random.randint(10, numElements, [1, ]).item()
    shape = NumCpp.Shape(1, numElements)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    data[0, :offsetBeg] = 0
    data[0, -offsetEnd:] = 0
    cArray.setArray(data)
    assert np.array_equal(NumCpp.trim_zeros(cArray, 'b').getNumpyArray().flatten(),
                          np.trim_zeros(data.flatten(), 'b'))

    numElements = np.random.randint(50, 100, [1, ]).item()
    offsetBeg = np.random.randint(0, 10, [1, ]).item()
    offsetEnd = np.random.randint(10, numElements, [1, ]).item()
    shape = NumCpp.Shape(1, numElements)
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    data[0, :offsetBeg] = complex(0, 0)
    data[0, -offsetEnd:] = complex(0, 0)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.trim_zeros(cArray, 'b').getNumpyArray().flatten(),
                          np.trim_zeros(data.flatten(), 'b'))

    numElements = np.random.randint(50, 100, [1, ]).item()
    offsetBeg = np.random.randint(0, 10, [1, ]).item()
    offsetEnd = np.random.randint(10, numElements, [1, ]).item()
    shape = NumCpp.Shape(1, numElements)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    data[0, :offsetBeg] = 0
    data[0, -offsetEnd:] = 0
    cArray.setArray(data)
    assert np.array_equal(NumCpp.trim_zeros(cArray, 'fb').getNumpyArray().flatten(),
                          np.trim_zeros(data.flatten(), 'fb'))

    numElements = np.random.randint(50, 100, [1, ]).item()
    offsetBeg = np.random.randint(0, 10, [1, ]).item()
    offsetEnd = np.random.randint(10, numElements, [1, ]).item()
    shape = NumCpp.Shape(1, numElements)
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    data[0, :offsetBeg] = complex(0, 0)
    data[0, -offsetEnd:] = complex(0, 0)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.trim_zeros(cArray, 'fb').getNumpyArray().flatten(),
                          np.trim_zeros(data.flatten(), 'fb'))


####################################################################################
def test_trunc():
    value = np.random.rand(1).item() * np.pi
    assert NumCpp.truncScalar(value) == np.trunc(value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.truncArray(cArray), np.trunc(data))


####################################################################################
def test_union1d():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt32(shape)
    cArray2 = NumCpp.NdArrayUInt32(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.union1d(
        cArray1, cArray2).getNumpyArray().flatten(), np.union1d(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.union1d(
        cArray1, cArray2).getNumpyArray().flatten(), np.union1d(data1, data2))


####################################################################################
def test_unique():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.unique(
        cArray).getNumpyArray().flatten(), np.unique(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.unique(
        cArray).getNumpyArray().flatten(), np.unique(data))


####################################################################################
def test_unpackbits():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, np.iinfo(np.uint8).max + 1, [shape.rows, shape.cols]).astype(np.uint8)
    cArray = NumCpp.NdArrayUInt8(shape)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.unpackbitsLittleEndian(cArray, NumCpp.Axis.NONE).flatten(),
                          np.unpackbits(data, axis=None, bitorder='little'))
    assert np.array_equal(NumCpp.unpackbitsLittleEndian(cArray, NumCpp.Axis.ROW),
                          np.unpackbits(data, axis=0, bitorder='little'))
    assert np.array_equal(NumCpp.unpackbitsLittleEndian(cArray, NumCpp.Axis.COL),
                          np.unpackbits(data, axis=1, bitorder='little'))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(0, np.iinfo(np.uint8).max + 1, [shape.rows, shape.cols]).astype(np.uint8)
    cArray = NumCpp.NdArrayUInt8(shape)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.unpackbitsBigEndian(cArray, NumCpp.Axis.NONE).flatten(),
                          np.unpackbits(data, axis=None, bitorder='big'))
    assert np.array_equal(NumCpp.unpackbitsBigEndian(cArray, NumCpp.Axis.ROW),
                          np.unpackbits(data, axis=0, bitorder='big'))
    assert np.array_equal(NumCpp.unpackbitsBigEndian(cArray, NumCpp.Axis.COL),
                          np.unpackbits(data, axis=1, bitorder='big'))


####################################################################################
def test_unwrap():
    value = np.random.randn(1).item() * 3 * np.pi
    assert np.round(NumCpp.unwrapScalar(value), 9) == np.round(
        np.arctan2(np.sin(value), np.cos(value)), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.unwrapArray(cArray), 9), np.round(
        np.arctan2(np.sin(data), np.cos(data)), 9))


####################################################################################
def test_vander():
    size = np.random.randint(5, 10)
    cArray = NumCpp.NdArray(1, size)
    data = np.random.randint(0, 10, [size, ])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.vander(cArray, True), 8),
                          np.round(np.vander(data, increasing=True), 8))
    assert np.array_equal(np.round(NumCpp.vander(cArray, False), 8),
                          np.round(np.vander(data, increasing=False), 8))
    assert np.array_equal(np.round(NumCpp.vander(cArray, size + 1, True), 8),
                          np.round(np.vander(data, size + 1, increasing=True), 8))
    assert np.array_equal(np.round(NumCpp.vander(cArray, size + 1, True), 8),
                          np.round(np.vander(data, size + 1, increasing=True), 8))


####################################################################################
def test_var():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.round(NumCpp.var(cArray, NumCpp.Axis.NONE).item(),
                    8) == np.round(np.var(data), 8)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.var(cArray, NumCpp.Axis.NONE) is not None

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.var(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), 8),
                          np.round(np.var(data, axis=0), 8))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.var(cArray, NumCpp.Axis.ROW) is not None

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.var(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), 8),
                          np.round(np.var(data, axis=1), 8))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.var(cArray, NumCpp.Axis.COL) is not None


####################################################################################
def test_vsplit():
    shapeInput = np.random.randint(80, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    numSplits = np.random.randint(2, 5)
    indices = np.sort(np.random.choice(range(shape.rows), numSplits, replace=False))
    cIndices = NumCpp.NdArrayInt32(indices.size)
    cIndices.setArray(indices.astype(np.int32))
    result = np.vsplit(data, indices)
    cResult = NumCpp.vsplit(cArray, cIndices)
    assert len(cResult) == len(result)
    for i in range(len(result)):
        assert np.array_equal(cResult[i], result[i])

####################################################################################


def test_vstack():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[0].item(
    ) + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
    shape3 = NumCpp.Shape(shapeInput[0].item(
    ) + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
    shape4 = NumCpp.Shape(shapeInput[0].item(
    ) + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    cArray3 = NumCpp.NdArray(shape3)
    cArray4 = NumCpp.NdArray(shape4)
    data1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    data4 = np.random.randint(1, 100, [shape4.rows, shape4.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    assert np.array_equal(NumCpp.vstack(cArray1, cArray2, cArray3, cArray4),
                          np.vstack([data1, data2, data3, data4]))


####################################################################################
def test_where():
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArrayMask = NumCpp.NdArrayBool(shape)
    cArrayA = NumCpp.NdArray(shape)
    cArrayB = NumCpp.NdArray(shape)
    dataMask = np.random.randint(0, 2, [shape.rows, shape.cols], dtype=bool)
    dataA = np.random.randint(1, 100, [shape.rows, shape.cols])
    dataB = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArrayMask.setArray(dataMask)
    cArrayA.setArray(dataA)
    cArrayB.setArray(dataB)
    assert np.array_equal(NumCpp.where(
        cArrayMask, cArrayA, cArrayB), np.where(dataMask, dataA, dataB))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArrayMask = NumCpp.NdArrayBool(shape)
    dataMask = np.random.randint(0, 2, [shape.rows, shape.cols], dtype=bool)
    cArrayMask.setArray(dataMask)
    cArrayA = NumCpp.NdArrayComplexDouble(shape)
    realA = np.random.randint(1, 100, [shape.rows, shape.cols])
    imagA = np.random.randint(1, 100, [shape.rows, shape.cols])
    dataA = realA + 1j * imagA
    cArrayA.setArray(dataA)
    cArrayB = NumCpp.NdArrayComplexDouble(shape)
    realB = np.random.randint(1, 100, [shape.rows, shape.cols])
    imagB = np.random.randint(1, 100, [shape.rows, shape.cols])
    dataB = realB + 1j * imagB
    cArrayB.setArray(dataB)
    assert np.array_equal(NumCpp.where(
        cArrayMask, cArrayA, cArrayB), np.where(dataMask, dataA, dataB))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArrayMask = NumCpp.NdArrayBool(shape)
    cArrayA = NumCpp.NdArray(shape)
    dataMask = np.random.randint(0, 2, [shape.rows, shape.cols], dtype=bool)
    dataA = np.random.randint(1, 100, [shape.rows, shape.cols])
    dataB = np.random.randint(1, 100)
    cArrayMask.setArray(dataMask)
    cArrayA.setArray(dataA)
    assert np.array_equal(NumCpp.where(
        cArrayMask, cArrayA, dataB), np.where(dataMask, dataA, dataB))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArrayMask = NumCpp.NdArrayBool(shape)
    dataMask = np.random.randint(0, 2, [shape.rows, shape.cols], dtype=bool)
    cArrayMask.setArray(dataMask)
    cArrayA = NumCpp.NdArrayComplexDouble(shape)
    realA = np.random.randint(1, 100, [shape.rows, shape.cols])
    imagA = np.random.randint(1, 100, [shape.rows, shape.cols])
    dataA = realA + 1j * imagA
    cArrayA.setArray(dataA)
    realB = np.random.randint(1, 100)
    imagB = np.random.randint(1, 100)
    dataB = realB + 1j * imagB
    assert np.array_equal(NumCpp.where(
        cArrayMask, cArrayA, dataB), np.where(dataMask, dataA, dataB))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArrayMask = NumCpp.NdArrayBool(shape)
    cArrayB = NumCpp.NdArray(shape)
    dataMask = np.random.randint(0, 2, [shape.rows, shape.cols], dtype=bool)
    dataB = np.random.randint(1, 100, [shape.rows, shape.cols])
    dataA = np.random.randint(1, 100)
    cArrayMask.setArray(dataMask)
    cArrayB.setArray(dataB)
    assert np.array_equal(NumCpp.where(
        cArrayMask, dataA, cArrayB), np.where(dataMask, dataA, dataB))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArrayMask = NumCpp.NdArrayBool(shape)
    dataMask = np.random.randint(0, 2, [shape.rows, shape.cols], dtype=bool)
    cArrayMask.setArray(dataMask)
    cArrayB = NumCpp.NdArrayComplexDouble(shape)
    realB = np.random.randint(1, 100, [shape.rows, shape.cols])
    imagB = np.random.randint(1, 100, [shape.rows, shape.cols])
    dataB = realB + 1j * imagB
    cArrayB.setArray(dataB)
    realA = np.random.randint(1, 100)
    imagA = np.random.randint(1, 100)
    dataA = realA + 1j * imagA
    assert np.array_equal(NumCpp.where(
        cArrayMask, dataA, cArrayB), np.where(dataMask, dataA, dataB))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArrayMask = NumCpp.NdArrayBool(shape)
    dataMask = np.random.randint(0, 2, [shape.rows, shape.cols], dtype=bool)
    cArrayMask.setArray(dataMask)
    dataB = np.random.randint(1, 100)
    dataA = np.random.randint(1, 100)
    assert np.array_equal(NumCpp.where(
        cArrayMask, dataA, dataB), np.where(dataMask, dataA, dataB))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArrayMask = NumCpp.NdArrayBool(shape)
    dataMask = np.random.randint(0, 2, [shape.rows, shape.cols], dtype=bool)
    cArrayMask.setArray(dataMask)
    realB = np.random.randint(1, 100)
    imagB = np.random.randint(1, 100)
    dataB = realB + 1j * imagB
    realA = np.random.randint(1, 100)
    imagA = np.random.randint(1, 100)
    dataA = realA + 1j * imagA
    assert np.array_equal(NumCpp.where(
        cArrayMask, dataA, dataB), np.where(dataMask, dataA, dataB))


####################################################################################
def test_zeros():
    shapeInput = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.zerosSquare(shapeInput)
    assert (cArray.shape[0] == shapeInput and cArray.shape[1] == shapeInput and
            cArray.size == shapeInput ** 2 and np.all(cArray == 0))

    shapeInput = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.zerosSquareComplex(shapeInput)
    assert (cArray.shape[0] == shapeInput and cArray.shape[1] == shapeInput and
            cArray.size == shapeInput ** 2 and np.all(cArray == complex(0, 0)))

    shapeInput = np.random.randint(20, 100, [2, ])
    cArray = NumCpp.zerosRowCol(shapeInput[0].item(), shapeInput[1].item())
    assert (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod() and np.all(cArray == 0))

    shapeInput = np.random.randint(20, 100, [2, ])
    cArray = NumCpp.zerosRowColComplex(
        shapeInput[0].item(), shapeInput[1].item())
    assert (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod() and np.all(cArray == complex(0, 0)))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.zerosShape(shape)
    assert (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod() and np.all(cArray == 0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.zerosShapeComplex(shape)
    assert (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod() and np.all(cArray == complex(0, 0)))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.zeros_like(cArray1)
    assert (cArray2.shape().rows == shape.rows and cArray2.shape().cols == shape.cols and
            cArray2.size() == shapeInput.prod() and np.all(cArray2.getNumpyArray() == 0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.zeros_likeComplex(cArray1)
    assert (cArray2.shape().rows == shape.rows and cArray2.shape().cols == shape.cols and
            cArray2.size() == shapeInput.prod() and np.all(cArray2.getNumpyArray() == complex(0, 0)))
