import os
import getpass
import numpy as np
import scipy.ndimage.measurements as meas
from functools import reduce
import sys
sys.path.append(os.path.abspath(r'../lib'))
import NumCpp  # noqa E402


####################################################################################
def factors(n):
    return set(reduce(list.__add__,
                      ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


####################################################################################
def test_functions():
    np.random.seed(6666)

    randValue = np.random.randint(-100, -1, [1, ]).astype(np.double).item()
    assert NumCpp.absScaler(randValue) == np.abs(randValue)

    components = np.random.randint(-100, -1, [2, ]).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.absScaler(value), 9) == np.round(np.abs(value), 9)

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
    assert np.array_equal(np.round(NumCpp.absArray(cArray), 9), np.round(np.abs(data), 9))

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
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.alen(cArray) == shape.rows

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.all(cArray, NumCpp.Axis.NONE).astype(np.bool).item() == np.all(data).item()

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.all(cArray, NumCpp.Axis.NONE).astype(np.bool).item() == np.all(data).item()

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.all(cArray, NumCpp.Axis.ROW).flatten().astype(np.bool), np.all(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.all(cArray, NumCpp.Axis.ROW).flatten().astype(np.bool), np.all(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.all(cArray, NumCpp.Axis.COL).flatten().astype(np.bool), np.all(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.all(cArray, NumCpp.Axis.COL).flatten().astype(np.bool), np.all(data, axis=1))

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
    assert NumCpp.allclose(cArray1, cArray2, tolerance) and not NumCpp.allclose(cArray1, cArray3, tolerance)

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
    assert np.array_equal(NumCpp.amax(cArray, NumCpp.Axis.ROW).flatten(), np.max(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.amax(cArray, NumCpp.Axis.ROW).flatten(), np.max(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.amax(cArray, NumCpp.Axis.COL).flatten(), np.max(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.amax(cArray, NumCpp.Axis.COL).flatten(), np.max(data, axis=1))

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
    assert np.array_equal(NumCpp.amin(cArray, NumCpp.Axis.ROW).flatten(), np.min(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.amin(cArray, NumCpp.Axis.ROW).flatten(), np.min(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.amin(cArray, NumCpp.Axis.COL).flatten(), np.min(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.amin(cArray, NumCpp.Axis.COL).flatten(), np.min(data, axis=1))

    components = np.random.randint(-100, -1, [2, ]).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.angleScaler(value), 9) == np.round(np.angle(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols]) + \
        1j * np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.angleArray(cArray), 9), np.round(np.angle(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.any(cArray, NumCpp.Axis.NONE).astype(np.bool).item() == np.any(data).item()

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.any(cArray, NumCpp.Axis.NONE).astype(np.bool).item() == np.any(data).item()

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.any(cArray, NumCpp.Axis.ROW).flatten().astype(np.bool), np.any(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.any(cArray, NumCpp.Axis.ROW).flatten().astype(np.bool), np.any(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.any(cArray, NumCpp.Axis.COL).flatten().astype(np.bool), np.any(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.any(cArray, NumCpp.Axis.COL).flatten().astype(np.bool), np.any(data, axis=1))

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
    shape2 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + NumCppols)
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    data1 = np.random.randint(0, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(0, 100, [shape2.rows, shape2.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.append(cArray1, cArray2, NumCpp.Axis.COL).getNumpyArray(),
                          np.append(data1, data2, axis=1))

    start = np.random.randn(1).item()
    stop = np.random.randn(1).item() * 100
    step = np.abs(np.random.randn(1).item())
    if stop < start:
        step *= -1
    data = np.arange(start, stop, step)
    assert np.array_equal(np.round(NumCpp.arange(start, stop, step).flatten(), 9), np.round(data, 9))

    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.arccosScaler(value), 9) == np.round(np.arccos(value), 9)

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.arccosScaler(value), 9) == np.round(np.arccos(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.arccosArray(cArray), 9), np.round(np.arccos(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.arccosArray(cArray), 9), np.round(np.arccos(data), 9))

    value = np.abs(np.random.rand(1).item()) + 1
    assert np.round(NumCpp.arccoshScaler(value), 9) == np.round(np.arccosh(value), 9)

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.arccoshScaler(value), 9) == np.round(np.arccosh(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.arccoshArray(cArray), 9), np.round(np.arccosh(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.arccoshArray(cArray), 9), np.round(np.arccosh(data), 9))

    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.arcsinScaler(value), 9) == np.round(np.arcsin(value), 9)

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.arcsinScaler(value), 9) == np.round(np.arcsin(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.arcsinArray(cArray), 9), np.round(np.arcsin(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    np.array_equal(np.round(NumCpp.arcsinArray(cArray), 9), np.round(np.arcsin(data), 9))

    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.arcsinhScaler(value), 9) == np.round(np.arcsinh(value), 9)

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.arcsinhScaler(value), 9) == np.round(np.arcsinh(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.arcsinhArray(cArray), 9), np.round(np.arcsinh(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    np.array_equal(np.round(NumCpp.arcsinhArray(cArray), 9), np.round(np.arcsinh(data), 9))

    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.arctanScaler(value), 9) == np.round(np.arctan(value), 9)

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.arctanScaler(value), 9) == np.round(np.arctan(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.arctanArray(cArray), 9), np.round(np.arctan(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    np.array_equal(np.round(NumCpp.arctanArray(cArray), 9), np.round(np.arctan(data), 9))

    xy = NumCpp.uniformOnSphere(1, 2).getNumpyArray().flatten()
    assert np.round(NumCpp.arctan2Scaler(xy[1], xy[0]), 9) == np.round(np.arctan2(xy[1], xy[0]), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArrayX = NumCpp.NdArray(shape)
    cArrayY = NumCpp.NdArray(shape)
    xy = NumCpp.uniformOnSphere(np.prod(shapeInput).item(), 2).getNumpyArray()
    xData = xy[:, 0].reshape(shapeInput)
    yData = xy[:, 1].reshape(shapeInput)
    cArrayX.setArray(xData)
    cArrayY.setArray(yData)
    assert np.array_equal(np.round(NumCpp.arctan2Array(cArrayY, cArrayX), 9), np.round(np.arctan2(yData, xData), 9))

    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.arctanhScaler(value), 9) == np.round(np.arctanh(value), 9)

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.arctanhScaler(value), 9) == np.round(np.arctanh(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.arctanhArray(cArray), 9), np.round(np.arctanh(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    np.array_equal(np.round(NumCpp.arctanhArray(cArray), 9), np.round(np.arctanh(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmax(cArray, NumCpp.Axis.NONE).item(), np.argmax(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmax(cArray, NumCpp.Axis.NONE).item(), np.argmax(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmax(cArray, NumCpp.Axis.ROW).flatten(), np.argmax(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmax(cArray, NumCpp.Axis.ROW).flatten(), np.argmax(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmax(cArray, NumCpp.Axis.COL).flatten(), np.argmax(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmax(cArray, NumCpp.Axis.COL).flatten(), np.argmax(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmin(cArray, NumCpp.Axis.NONE).item(), np.argmin(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmin(cArray, NumCpp.Axis.NONE).item(), np.argmin(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmin(cArray, NumCpp.Axis.ROW).flatten(), np.argmin(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmin(cArray, NumCpp.Axis.ROW).flatten(), np.argmin(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmin(cArray, NumCpp.Axis.COL).flatten(), np.argmin(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.argmin(cArray, NumCpp.Axis.COL).flatten(), np.argmin(data, axis=1))

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
        if not np.array_equal(row[cIdx[idx, :]], row[pIdx[idx, :]]):
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

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    randValue = np.random.randint(0, 100, [1, ]).item()
    data2 = data > randValue
    cArray.setArray(data2)
    assert np.array_equal(NumCpp.argwhere(cArray).flatten(), np.argwhere(data.flatten() > randValue).flatten())

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    randValue = np.random.randint(0, 100, [1, ]).item()
    data2 = data > randValue
    cArray.setArray(data2)
    assert np.array_equal(NumCpp.argwhere(cArray).flatten(), np.argwhere(data.flatten() > randValue).flatten())

    value = np.abs(np.random.rand(1).item()) * np.random.randint(1, 10, [1, ]).item()
    numDecimalsRound = np.random.randint(0, 10, [1, ]).astype(np.uint8).item()
    assert NumCpp.aroundScaler(value, numDecimalsRound) == np.round(value, numDecimalsRound)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * np.random.randint(1, 10, [1, ]).item()
    cArray.setArray(data)
    numDecimalsRound = np.random.randint(0, 10, [1, ]).astype(np.uint8).item()
    assert np.array_equal(NumCpp.aroundArray(cArray, numDecimalsRound), np.round(data, numDecimalsRound))

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
    assert NumCpp.array_equal(cArray1, cArray2) and not NumCpp.array_equal(cArray1, cArray3)

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
    assert NumCpp.array_equal(cArray1, cArray2) and not NumCpp.array_equal(cArray1, cArray3)

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
    cArray2.setArray(data1.reshape([shapeInput1[1].item(), shapeInput1[0].item()]))
    cArray3.setArray(data3)
    assert NumCpp.array_equiv(cArray1, cArray2) and not NumCpp.array_equiv(cArray1, cArray3)

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
    cArray2.setArray(data1.reshape([shapeInput1[1].item(), shapeInput1[0].item()]))
    cArray3.setArray(data3)
    assert NumCpp.array_equiv(cArray1, cArray2) and not NumCpp.array_equiv(cArray1, cArray3)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    assert np.array_equal(NumCpp.asarrayArray1D(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayArray1D(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    assert np.array_equal(NumCpp.asarrayArray1DCopy(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayArray1DCopy(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayArray2D(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayArray2D(*values), data)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayArray2DCopy(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayArray2DCopy(*values), data)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    assert np.array_equal(NumCpp.asarrayVector1D(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayVector1D(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    assert np.array_equal(NumCpp.asarrayVector1DCopy(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayVector1DCopy(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayVector2D(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayVector2D(*values), data)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayVectorArray2D(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayVectorArray2D(*values), data)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayVectorArray2DCopy(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayVectorArray2DCopy(*values), data)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    assert np.array_equal(NumCpp.asarrayDeque1D(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayDeque1D(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayDeque2D(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayDeque2D(*values), data)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    assert np.array_equal(NumCpp.asarrayList(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayList(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    assert np.array_equal(NumCpp.asarrayIterators(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayIterators(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    assert np.array_equal(NumCpp.asarrayPointerIterators(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayPointerIterators(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    assert np.array_equal(NumCpp.asarrayPointer(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayPointer(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayPointer2D(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayPointer2D(*values), data)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    assert np.array_equal(NumCpp.asarrayPointerShell(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayPointerShell(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayPointerShell2D(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayPointerShell2D(*values), data)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    assert np.array_equal(NumCpp.asarrayPointerShellTakeOwnership(*values).flatten(), values)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    assert np.array_equal(NumCpp.asarrayPointerShellTakeOwnership(*values).flatten(), values)

    values = np.random.randint(0, 100, [2, ]).astype(np.double)
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayPointerShell2DTakeOwnership(*values), data)

    real = np.random.randint(0, 100, [2, ]).astype(np.double)
    imag = np.random.randint(0, 100, [2, ]).astype(np.double)
    values = real + 1j * imag
    data = np.vstack([values, values])
    assert np.array_equal(NumCpp.asarrayPointerShell2DTakeOwnership(*values), data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.round(NumCpp.average(cArray, NumCpp.Axis.NONE).item(), 9) == np.round(np.average(data), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.round(NumCpp.average(cArray, NumCpp.Axis.NONE).item(), 9) == np.round(np.average(data), 9)

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

    value = np.random.randint(0, np.iinfo(np.uint64).max, [1, ], dtype=np.uint64).item()
    assert NumCpp.binaryRepr(np.uint64(value)) == np.binary_repr(value, np.iinfo(np.uint64).bits)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.bincount(cArray, 0).flatten(), np.bincount(data.flatten(), minlength=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    cArray.setArray(data)
    minLength = int(np.max(data) + 10)
    assert np.array_equal(NumCpp.bincount(cArray, minLength).flatten(),
                          np.bincount(data.flatten(), minlength=minLength))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    cWeights = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    weights = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    cArray.setArray(data)
    cWeights.setArray(weights)
    assert np.array_equal(NumCpp.bincountWeighted(cArray, cWeights, 0).flatten(),
                          np.bincount(data.flatten(), minlength=0, weights=weights.flatten()))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    cWeights = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    weights = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    cArray.setArray(data)
    cWeights.setArray(weights)
    minLength = int(np.max(data) + 10)
    assert np.array_equal(NumCpp.bincountWeighted(cArray, cWeights, minLength).flatten(),
                          np.bincount(data.flatten(), minlength=minLength, weights=weights.flatten()))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt64(shape)
    cArray2 = NumCpp.NdArrayUInt64(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.bitwise_and(cArray1, cArray2), np.bitwise_and(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt64(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.bitwise_not(cArray), np.bitwise_not(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt64(shape)
    cArray2 = NumCpp.NdArrayUInt64(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.bitwise_or(cArray1, cArray2), np.bitwise_or(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt64(shape)
    cArray2 = NumCpp.NdArrayUInt64(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.bitwise_xor(cArray1, cArray2), np.bitwise_xor(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt64(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.byteswap(cArray).shape, shapeInput)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.cbrtArray(cArray), 9), np.round(np.cbrt(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols).astype(np.double) * 1000
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.ceilArray(cArray), 9), np.round(np.ceil(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols).astype(np.double) * 1000
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.centerOfMass(cArray, NumCpp.Axis.NONE).flatten(), 9),
                          np.round(meas.center_of_mass(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols).astype(np.double) * 1000
    cArray.setArray(data)

    coms = list()
    for col in range(data.shape[1]):
        coms.append(np.round(meas.center_of_mass(data[:, col])[0], 9))

    assert np.array_equal(np.round(NumCpp.centerOfMass(cArray, NumCpp.Axis.ROW).flatten(), 9), np.round(coms, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols).astype(np.double) * 1000
    cArray.setArray(data)

    coms = list()
    for row in range(data.shape[0]):
        coms.append(np.round(meas.center_of_mass(data[row, :])[0], 9))

    assert np.array_equal(np.round(NumCpp.centerOfMass(cArray, NumCpp.Axis.COL).flatten(), 9), np.round(coms, 9))

    value = np.random.randint(0, 100, [1, ]).item()
    minValue = np.random.randint(0, 10, [1, ]).item()
    maxValue = np.random.randint(90, 100, [1, ]).item()
    assert NumCpp.clipScaler(value, minValue, maxValue) == np.clip(value, minValue, maxValue)

    value = np.random.randint(0, 100, [1, ]).item() + 1j * np.random.randint(0, 100, [1, ]).item()
    minValue = np.random.randint(0, 10, [1, ]).item() + 1j * np.random.randint(0, 10, [1, ]).item()
    maxValue = np.random.randint(90, 100, [1, ]).item() + 1j * np.random.randint(0, 100, [1, ]).item()
    assert NumCpp.clipScaler(value, minValue, maxValue) == np.clip(value, minValue, maxValue)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    minValue = np.random.randint(0, 10, [1, ]).item()
    maxValue = np.random.randint(90, 100, [1, ]).item()
    assert np.array_equal(NumCpp.clipArray(cArray, minValue, maxValue), np.clip(data, minValue, maxValue))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    minValue = np.random.randint(0, 10, [1, ]).item() + 1j * np.random.randint(0, 10, [1, ]).item()
    maxValue = np.random.randint(90, 100, [1, ]).item() + 1j * np.random.randint(0, 100, [1, ]).item()
    assert np.array_equal(NumCpp.clipArray(cArray, minValue, maxValue), np.clip(data, minValue, maxValue))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1, ]).item())
    shape3 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1, ]).item())
    shape4 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1, ]).item())
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

    real = np.random.rand(1).astype(np.double).item()
    value = np.complex(real)
    assert np.round(NumCpp.complexScaler(real), 9) == np.round(value, 9)

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.complexScaler(components[0], components[1]), 9) == np.round(value, 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    realArray = NumCpp.NdArray(shape)
    real = np.random.rand(shape.rows, shape.cols)
    realArray.setArray(real)
    assert np.array_equal(np.round(NumCpp.complexArray(realArray), 9), np.round(real + 1j * np.zeros_like(real), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    realArray = NumCpp.NdArray(shape)
    imagArray = NumCpp.NdArray(shape)
    real = np.random.rand(shape.rows, shape.cols)
    imag = np.random.rand(shape.rows, shape.cols)
    realArray.setArray(real)
    imagArray.setArray(imag)
    assert np.array_equal(np.round(NumCpp.complexArray(realArray, imagArray), 9), np.round(real + 1j * imag, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1, ]).item())
    shape3 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1, ]).item())
    shape4 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1, ]).item())
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
    shape2 = NumCpp.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
    shape3 = NumCpp.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
    shape4 = NumCpp.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
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
    shape2 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1, ]).item())
    shape3 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1, ]).item())
    shape4 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1, ]).item())
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

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.conjScaler(value), 9) == np.round(np.conj(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.conjArray(cArray), 9), np.round(np.conj(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    value = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    assert NumCpp.contains(cArray, value, NumCpp.Axis.NONE).getNumpyArray().item() == (value in data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    value = np.random.randint(0, 100, [1, ]).item() + 1j * np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    assert NumCpp.contains(cArray, value, NumCpp.Axis.NONE).getNumpyArray().item() == (value in data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    value = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    truth = list()
    for row in data:
        truth.append(value in row)
    assert np.array_equal(NumCpp.contains(cArray, value, NumCpp.Axis.COL).getNumpyArray().flatten(), np.asarray(truth))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    value = np.random.randint(0, 100, [1, ]).item() + 1j * np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    truth = list()
    for row in data:
        truth.append(value in row)
    assert np.array_equal(NumCpp.contains(cArray, value, NumCpp.Axis.COL).getNumpyArray().flatten(), np.asarray(truth))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    value = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    truth = list()
    for row in data.T:
        truth.append(value in row)
    assert np.array_equal(NumCpp.contains(cArray, value, NumCpp.Axis.ROW).getNumpyArray().flatten(), np.asarray(truth))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    value = np.random.randint(0, 100, [1, ]).item() + 1j * np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    truth = list()
    for row in data.T:
        truth.append(value in row)
    assert np.array_equal(NumCpp.contains(cArray, value, NumCpp.Axis.ROW).getNumpyArray().flatten(), np.asarray(truth))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.copy(cArray), data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.copysign(cArray1, cArray2), np.copysign(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray()
    data1 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    assert np.array_equal(NumCpp.copyto(cArray2, cArray1), data1)

    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.cosScaler(value), 9) == np.round(np.cos(value), 9)

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.cosScaler(value), 9) == np.round(np.cos(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.cosArray(cArray), 9), np.round(np.cos(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.cosArray(cArray), 9), np.round(np.cos(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.coshArray(cArray), 9), np.round(np.cosh(data), 9))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 3, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert NumCpp.count_nonzero(cArray, NumCpp.Axis.NONE) == np.count_nonzero(data)

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 3, [shape.rows, shape.cols])
    imag = np.random.randint(1, 3, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.count_nonzero(cArray, NumCpp.Axis.NONE) == np.count_nonzero(data)

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 3, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.count_nonzero(cArray, NumCpp.Axis.ROW).flatten(), np.count_nonzero(data, axis=0))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 3, [shape.rows, shape.cols])
    imag = np.random.randint(1, 3, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.count_nonzero(cArray, NumCpp.Axis.ROW).flatten(), np.count_nonzero(data, axis=0))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 3, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.count_nonzero(cArray, NumCpp.Axis.COL).flatten(), np.count_nonzero(data, axis=1))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 3, [shape.rows, shape.cols])
    imag = np.random.randint(1, 3, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.count_nonzero(cArray, NumCpp.Axis.COL).flatten(), np.count_nonzero(data, axis=1))

    shape = NumCpp.Shape(1, 2)
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert NumCpp.cross(cArray1, cArray2, NumCpp.Axis.NONE).item() == np.cross(data1, data2).item()

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
    assert NumCpp.cross(cArray1, cArray2, NumCpp.Axis.NONE).item() == np.cross(data1, data2).item()

    shape = NumCpp.Shape(2, np.random.randint(1, 100, [1, ]).item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
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
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
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
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
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
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
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
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
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

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.cube(cArray), 9), np.round(data * data * data, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.cube(cArray), 9), np.round(data * data * data, 9))

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumprod(cArray, NumCpp.Axis.NONE).flatten(), data.cumprod())

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 4, [shape.rows, shape.cols])
    imag = np.random.randint(1, 4, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumprod(cArray, NumCpp.Axis.NONE).flatten(), data.cumprod())

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumprod(cArray, NumCpp.Axis.ROW), data.cumprod(axis=0))

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 4, [shape.rows, shape.cols])
    imag = np.random.randint(1, 4, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumprod(cArray, NumCpp.Axis.ROW), data.cumprod(axis=0))

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumprod(cArray, NumCpp.Axis.COL), data.cumprod(axis=1))

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 4, [shape.rows, shape.cols])
    imag = np.random.randint(1, 4, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumprod(cArray, NumCpp.Axis.COL), data.cumprod(axis=1))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumsum(cArray, NumCpp.Axis.NONE).flatten(), data.cumsum())

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag = np.random.randint(1, 50, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumsum(cArray, NumCpp.Axis.NONE).flatten(), data.cumsum())

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumsum(cArray, NumCpp.Axis.ROW), data.cumsum(axis=0))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag = np.random.randint(1, 50, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumsum(cArray, NumCpp.Axis.ROW), data.cumsum(axis=0))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumsum(cArray, NumCpp.Axis.COL), data.cumsum(axis=1))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag = np.random.randint(1, 50, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.cumsum(cArray, NumCpp.Axis.COL), data.cumsum(axis=1))

    value = np.abs(np.random.rand(1).item()) * 360
    assert np.round(NumCpp.deg2radScaler(value), 9) == np.round(np.deg2rad(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 360
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.deg2radArray(cArray), 9), np.round(np.deg2rad(data), 9))

    value = np.abs(np.random.rand(1).item()) * 2 * np.pi
    assert np.round(NumCpp.degreesScaler(value), 9) == np.round(np.degrees(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 2 * np.pi
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.degreesArray(cArray), 9), np.round(np.degrees(data), 9))

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
    indices = NumCpp.Slice(0, 100, 4)
    indicesPy = slice(0, 99, 4)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.deleteIndicesSlice(cArray, indices, NumCpp.Axis.ROW),
                          np.delete(data, indicesPy, axis=0))

    shapeInput = np.asarray([100, 100])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    indices = NumCpp.Slice(0, 100, 4)
    indicesPy = slice(0, 99, 4)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.deleteIndicesSlice(cArray, indices, NumCpp.Axis.COL),
                          np.delete(data, indicesPy, axis=1))

    shapeInput = np.asarray([100, 100])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    index = np.random.randint(0, shape.size(), [1, ]).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.deleteIndicesScaler(cArray, index, NumCpp.Axis.NONE).flatten(),
                          np.delete(data, index, axis=None))

    shapeInput = np.asarray([100, 100])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    index = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.deleteIndicesScaler(cArray, index, NumCpp.Axis.ROW), np.delete(data, index, axis=0))

    shapeInput = np.asarray([100, 100])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    index = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.deleteIndicesScaler(cArray, index, NumCpp.Axis.COL), np.delete(data, index, axis=1))

    shapeInput = np.random.randint(2, 25, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    k = np.random.randint(0, np.min(shapeInput), [1, ]).item()
    elements = np.random.randint(1, 100, shapeInput)
    cElements = NumCpp.NdArray(shape)
    cElements.setArray(elements)
    assert np.array_equal(NumCpp.diag(cElements, k).flatten(), np.diag(elements, k))

    shapeInput = np.random.randint(2, 25, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    k = np.random.randint(0, np.min(shapeInput), [1, ]).item()
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    elements = real + 1j * imag
    cElements = NumCpp.NdArrayComplexDouble(shape)
    cElements.setArray(elements)
    assert np.array_equal(NumCpp.diag(cElements, k).flatten(), np.diag(elements, k))

    numElements = np.random.randint(2, 25, [1, ]).item()
    shape = NumCpp.Shape(1, numElements)
    k = np.random.randint(0, 10, [1, ]).item()
    elements = np.random.randint(1, 100, [numElements, ])
    cElements = NumCpp.NdArray(shape)
    cElements.setArray(elements)
    assert np.array_equal(NumCpp.diagflat(cElements, k), np.diagflat(elements, k))

    numElements = np.random.randint(2, 25, [1, ]).item()
    shape = NumCpp.Shape(1, numElements)
    k = np.random.randint(0, 10, [1, ]).item()
    real = np.random.randint(1, 100, [numElements, ])
    imag = np.random.randint(1, 100, [numElements, ])
    elements = real + 1j * imag
    cElements = NumCpp.NdArrayComplexDouble(shape)
    cElements.setArray(elements)
    assert np.array_equal(NumCpp.diagflat(cElements, k), np.diagflat(elements, k))

    numElements = np.random.randint(1, 25, [1, ]).item()
    shape = NumCpp.Shape(1, numElements)
    k = np.random.randint(0, 10, [1, ]).item()
    elements = np.random.randint(1, 100, [numElements, ])
    cElements = NumCpp.NdArray(shape)
    cElements.setArray(elements)
    assert np.array_equal(NumCpp.diagflat(cElements, k), np.diagflat(elements, k))

    numElements = np.random.randint(1, 25, [1, ]).item()
    shape = NumCpp.Shape(1, numElements)
    k = np.random.randint(0, 10, [1, ]).item()
    real = np.random.randint(1, 100, [numElements, ])
    imag = np.random.randint(1, 100, [numElements, ])
    elements = real + 1j * imag
    cElements = NumCpp.NdArrayComplexDouble(shape)
    cElements.setArray(elements)
    assert np.array_equal(NumCpp.diagflat(cElements, k), np.diagflat(elements, k))

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

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.diff(cArray, NumCpp.Axis.NONE).flatten(),
                          np.diff(data.flatten()))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag = np.random.randint(1, 50, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.diff(cArray, NumCpp.Axis.NONE).flatten(),
                          np.diff(data.flatten()))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.diff(cArray, NumCpp.Axis.ROW), np.diff(data, axis=0))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag = np.random.randint(1, 50, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.diff(cArray, NumCpp.Axis.ROW), np.diff(data, axis=0))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.diff(cArray, NumCpp.Axis.COL).astype(np.uint32), np.diff(data, axis=1))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 50, [shape.rows, shape.cols])
    imag = np.random.randint(1, 50, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.diff(cArray, NumCpp.Axis.COL), np.diff(data, axis=1))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(np.double)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(np.round(NumCpp.divide(cArray1, cArray2), 9), np.round(np.divide(data1, data2), 9))

    shapeInput = np.random.randint(1, 50, [2, ])
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
    assert np.array_equal(np.round(NumCpp.divide(cArray1, cArray2), 9), np.round(np.divide(data1, data2), 9))

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

    shapeInput = np.random.randint(1, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[1].item(), np.random.randint(1, 100, [1, ]).item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    data1 = np.random.randint(1, 50, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 50, [shape2.rows, shape2.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.dot(cArray1, cArray2), np.dot(data1, data2))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[1].item(), np.random.randint(1, 100, [1, ]).item())
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

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if sys.platform == 'linux':
        tempDir = r'/home/' + getpass.getuser() + r'/Desktop/'
    else:
        tempDir = r'C:\Temp'
    if not os.path.exists(tempDir):
        os.mkdir(tempDir)
    tempFile = os.path.join(tempDir, 'NdArrayDump.bin')
    NumCpp.dump(cArray, tempFile)
    assert os.path.exists(tempFile)
    data2 = np.fromfile(tempFile, dtype=np.double).reshape(shapeInput)
    assert np.array_equal(data, data2)
    os.remove(tempFile)

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    if sys.platform == 'linux':
        tempDir = r'/home/' + getpass.getuser() + r'/Desktop/'
    else:
        tempDir = r'C:\Temp'
    if not os.path.exists(tempDir):
        os.mkdir(tempDir)
    tempFile = os.path.join(tempDir, 'NdArrayDump.bin')
    NumCpp.dump(cArray, tempFile)
    assert os.path.exists(tempFile)
    data2 = np.fromfile(tempFile, dtype=np.complex).reshape(shapeInput)
    assert np.array_equal(data, data2)
    os.remove(tempFile)

    shapeInput = np.random.randint(1, 100, [2, ])
    cArray = NumCpp.emptyRowCol(shapeInput[0].item(), shapeInput[1].item())
    assert cArray.shape[0] == shapeInput[0]
    assert cArray.shape[1] == shapeInput[1]
    assert cArray.size == shapeInput.prod()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.emptyShape(shape)
    assert cArray.shape[0] == shape.rows
    assert cArray.shape[1] == shape.cols
    assert cArray.size == shapeInput.prod()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.empty_like(cArray1)
    assert cArray2.shape().rows == shape.rows
    assert cArray2.shape().cols == shape.cols
    assert cArray2.size() == shapeInput.prod()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    assert NumCpp.endianess(cArray) == NumCpp.Endian.NATIVE

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 10, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 10, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.equal(cArray1, cArray2), np.equal(data1, data2))

    shapeInput = np.random.randint(1, 100, [2, ])
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
    assert np.array_equal(NumCpp.equal(cArray1, cArray2), np.equal(data1, data2))

    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.expScaler(value), 9) == np.round(np.exp(value), 9)

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.expScaler(value), 9) == np.round(np.exp(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.expArray(cArray), 9), np.round(np.exp(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.expArray(cArray), 9), np.round(np.exp(data), 9))

    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.exp2Scaler(value), 9) == np.round(np.exp2(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.exp2Array(cArray), 9), np.round(np.exp2(data), 9))

    value = np.abs(np.random.rand(1).item())
    assert np.round(NumCpp.expm1Scaler(value), 9) == np.round(np.expm1(value), 9)

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.expm1Scaler(value), 9) == np.round(np.expm1(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.expm1Array(cArray), 9), np.round(np.expm1(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.rand(shape.rows, shape.cols)
    imag = np.random.rand(shape.rows, shape.cols)
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.expm1Array(cArray), 9), np.round(np.expm1(data), 9))

    shapeInput = np.random.randint(1, 100, [1, ]).item()
    randK = np.random.randint(0, shapeInput, [1, ]).item()
    assert np.array_equal(NumCpp.eye1D(shapeInput, randK), np.eye(shapeInput, k=randK))

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
    assert np.array_equal(NumCpp.eyeShape(cShape, randK), np.eye(shapeInput[0].item(), shapeInput[1].item(), k=randK))

    shapeInput = np.random.randint(10, 100, [2, ])
    cShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    randK = np.random.randint(0, np.min(shapeInput), [1, ]).item()
    assert np.array_equal(NumCpp.eyeShapeComplex(cShape, randK),
                          np.eye(shapeInput[0].item(), shapeInput[1].item(), k=randK) +
                          1j * np.zeros(shapeInput))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    NumCpp.fillDiagonal(cArray, 666)
    np.fill_diagonal(data, 666)
    assert np.array_equal(cArray.getNumpyArray(), data)

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

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    value = data.mean()
    cMask = NumCpp.operatorGreater(cArray ,value)
    cMaskArray = NumCpp.NdArrayBool(cMask.shape[0], cMask.shape[1])
    cMaskArray.setArray(cMask)
    idxs = NumCpp.findN(cMaskArray, 8).astype(np.int64)
    idxsPy = np.nonzero((data > value).flatten())[0]
    assert np.array_equal(idxs.flatten(), idxsPy[:8])

    value = np.random.randn(1).item() * 100
    assert NumCpp.fixScaler(value) == np.fix(value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    assert np.array_equal(NumCpp.fixArray(cArray), np.fix(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.flatten(cArray).getNumpyArray(), np.resize(data, [1, data.size]))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.flatnonzero(cArray).getNumpyArray().flatten(), np.flatnonzero(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.flip(cArray, NumCpp.Axis.NONE).getNumpyArray(),
                          np.flip(data.reshape(1, data.size), axis=1).reshape(shapeInput))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.fliplr(cArray).getNumpyArray(), np.fliplr(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.flipud(cArray).getNumpyArray(), np.flipud(data))

    value = np.random.randn(1).item() * 100
    assert NumCpp.floorScaler(value) == np.floor(value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    assert np.array_equal(NumCpp.floorArray(cArray), np.floor(data))

    value1 = np.random.randn(1).item() * 100 + 1000
    value2 = np.random.randn(1).item() * 100 + 1000
    assert NumCpp.floor_divideScaler(value1, value2) == np.floor_divide(value1, value2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data2 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.floor_divideArray(cArray1, cArray2), np.floor_divide(data1, data2))

    value1 = np.random.randn(1).item() * 100 + 1000
    value2 = np.random.randn(1).item() * 100 + 1000
    assert NumCpp.fmaxScaler(value1, value2) == np.fmax(value1, value2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data2 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.fmaxArray(cArray1, cArray2), np.fmax(data1, data2))

    value1 = np.random.randn(1).item() * 100 + 1000
    value2 = np.random.randn(1).item() * 100 + 1000
    assert NumCpp.fminScaler(value1, value2) == np.fmin(value1, value2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data2 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.fminArray(cArray1, cArray2), np.fmin(data1, data2))

    value1 = np.random.randint(1, 100, [1, ]).item() * 100 + 1000
    value2 = np.random.randint(1, 100, [1, ]).item() * 100 + 1000
    assert NumCpp.fmodScaler(value1, value2) == np.fmod(value1, value2)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt32(shape)
    cArray2 = NumCpp.NdArrayUInt32(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]) * 100 + 1000
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]) * 100 + 1000
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.fmodArray(cArray1, cArray2), np.fmod(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    tempDir = r'C:\Temp'
    if not os.path.exists(tempDir):
        os.mkdir(tempDir)
    tempFile = os.path.join(tempDir, 'NdArrayDump.bin')
    NumCpp.dump(cArray, tempFile)
    assert os.path.isfile(tempFile)
    data2 = NumCpp.fromfile(tempFile, '').reshape(shape)
    assert np.array_equal(data, data2)
    os.remove(tempFile)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if sys.platform == 'linux':
        tempDir = r'/home/' + getpass.getuser() + r'/Desktop/'
    else:
        tempDir = r'C:\Temp'
    if not os.path.exists(tempDir):
        os.mkdir(tempDir)
    tempFile = os.path.join(tempDir, 'NdArrayDump')
    NumCpp.tofile(cArray, tempFile, '\n')
    assert os.path.exists(tempFile + '.txt')
    data2 = NumCpp.fromfile(tempFile + '.txt', '\n').reshape(shape)
    assert np.array_equal(data, data2)
    os.remove(tempFile + '.txt')

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

    shapeInput = np.random.randint(1, 100, [1, ]).item()
    value = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.fullSquare(shapeInput, value)
    assert (cArray.shape[0] == shapeInput and cArray.shape[1] == shapeInput and
            cArray.size == shapeInput**2 and np.all(cArray == value))

    shapeInput = np.random.randint(1, 100, [1, ]).item()
    value = np.random.randint(1, 100, [1, ]).item() + 1j * np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.fullSquareComplex(shapeInput, value)
    assert (cArray.shape[0] == shapeInput and cArray.shape[1] == shapeInput and
            cArray.size == shapeInput**2 and np.all(cArray == value))

    shapeInput = np.random.randint(1, 100, [2, ])
    value = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.fullRowCol(shapeInput[0].item(), shapeInput[1].item(), value)
    assert (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod() and np.all(cArray == value))

    shapeInput = np.random.randint(1, 100, [2, ])
    value = np.random.randint(1, 100, [1, ]).item() + 1j * np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.fullRowColComplex(shapeInput[0].item(), shapeInput[1].item(), value)
    assert (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod() and np.all(cArray == value))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    value = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.fullShape(shape, value)
    assert (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod() and np.all(cArray == value))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    value = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.fullShape(shape, value)
    assert (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod() and np.all(cArray == value))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    value = np.random.randint(1, 100, [1, ]).item() + 1j * np.random.randint(1, 100, [1, ]).item()
    cArray2 = NumCpp.full_likeComplex(cArray1, value)
    assert (cArray2.shape().rows == shape.rows and cArray2.shape().cols == shape.cols and
            cArray2.size() == shapeInput.prod() and np.all(cArray2.getNumpyArray() == value))

    value1 = np.random.randint(1, 1000, [1, ]).item()
    value2 = np.random.randint(1, 1000, [1, ]).item()
    assert NumCpp.gcdScaler(value1, value2) == np.gcd(value1, value2)

    size = np.random.randint(20, 100, [1, ]).item()
    cArray = NumCpp.NdArrayUInt32(1, size)
    data = np.random.randint(1, 1000, [size, ], dtype=np.uint32)
    cArray.setArray(data)
    assert NumCpp.gcdArray(cArray) == np.gcd.reduce(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 1000, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.gradient(cArray, NumCpp.Axis.ROW), np.gradient(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 1000, [shape.rows, shape.cols])
    imag = np.random.randint(1, 1000, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.gradient(cArray, NumCpp.Axis.ROW), np.gradient(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 1000, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.gradient(cArray, NumCpp.Axis.COL), np.gradient(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 1000, [shape.rows, shape.cols])
    imag = np.random.randint(1, 1000, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.gradient(cArray, NumCpp.Axis.COL), np.gradient(data, axis=1))

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

    shape = NumCpp.Shape(1024, 1024)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(1024, 1024) * np.random.randint(1, 10, [1, ]).item() + np.random.randint(1, 10, [1, ]).item()
    cArray.setArray(data)
    numBins = np.random.randint(10, 30, [1, ]).item()
    histogram, bins = NumCpp.histogram(cArray, numBins)
    h, b = np.histogram(data, numBins)
    assert np.array_equal(histogram.getNumpyArray().flatten().astype(np.int32), h)
    assert np.array_equal(np.round(bins.getNumpyArray().flatten(), 9), np.round(b, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1, ]).item())
    shape3 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1, ]).item())
    shape4 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1, ]).item())
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

    value1 = np.random.randn(1).item() * 100 + 1000
    value2 = np.random.randn(1).item() * 100 + 1000
    assert NumCpp.hypotScaler(value1, value2) == np.hypot(value1, value2)

    value1 = np.random.randn(1).item() * 100 + 1000
    value2 = np.random.randn(1).item() * 100 + 1000
    value3 = np.random.randn(1).item() * 100 + 1000
    assert (np.round(NumCpp.hypotScalerTriple(value1, value2, value3), 9) ==
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

    squareSize = np.random.randint(10, 100, [1, ]).item()
    assert np.array_equal(NumCpp.identity(squareSize).getNumpyArray(), np.identity(squareSize))

    squareSize = np.random.randint(10, 100, [1, ]).item()
    assert np.array_equal(NumCpp.identityComplex(squareSize).getNumpyArray(),
                          np.identity(squareSize) + 1j * np.zeros([squareSize, squareSize]))

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.imagScaler(value), 9) == np.round(np.imag(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.imagArray(cArray), 9), np.round(np.imag(data), 9))

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

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt32(shape)
    cArray2 = NumCpp.NdArrayUInt32(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.intersect1d(cArray1, cArray2).getNumpyArray().flatten(), np.intersect1d(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.invert(cArray).getNumpyArray(), np.invert(data))

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

    value = np.random.randn(1).item() * 100 + 1000
    assert NumCpp.isinfScaler(value) == np.isinf(value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data[data > 1000] = np.inf
    cArray.setArray(data)
    assert np.array_equal(NumCpp.isinfArray(cArray), np.isinf(data))

    value = np.random.randn(1).item() * 100 + 1000
    assert NumCpp.isnanScaler(value) == np.isnan(value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data[data > 1000] = np.nan
    cArray.setArray(data)
    assert np.array_equal(NumCpp.isnanArray(cArray), np.isnan(data))

    value1 = np.random.randint(1, 1000, [1, ]).item()
    value2 = np.random.randint(1, 1000, [1, ]).item()
    assert NumCpp.lcmScaler(value1, value2) == np.lcm(value1, value2)

    size = np.random.randint(2, 10, [1, ]).item()
    cArray = NumCpp.NdArrayUInt32(1, size)
    data = np.random.randint(1, 100, [size, ])
    cArray.setArray(data)
    assert NumCpp.lcmArray(cArray) == np.lcm.reduce(data)

    value1 = np.random.randn(1).item() * 100
    value2 = np.random.randint(1, 20, [1, ]).item()
    assert np.round(NumCpp.ldexpScaler(value1, value2), 9) == np.round(np.ldexp(value1, value2), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArrayUInt8(shape)
    data1 = np.random.randn(shape.rows, shape.cols) * 100
    data2 = np.random.randint(1, 20, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(np.round(NumCpp.ldexpArray(cArray1, cArray2), 9), np.round(np.ldexp(data1, data2), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    bitsToshift = np.random.randint(1, 32, [1, ]).item()
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(1, np.iinfo(np.uint32).max, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.left_shift(cArray, bitsToshift).getNumpyArray(),
                          np.left_shift(data, bitsToshift))

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

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if sys.platform == 'linux':
        tempDir = r'/home/' + getpass.getuser() + r'/Desktop/'
    else:
        tempDir = r'C:\Temp'
    if not os.path.exists(tempDir):
        os.mkdir(tempDir)
    tempFile = os.path.join(tempDir, 'NdArrayDump.bin')
    NumCpp.dump(cArray, tempFile)
    assert os.path.isfile(tempFile)
    data2 = NumCpp.load(tempFile).reshape(shape)
    assert np.array_equal(data, data2)
    os.remove(tempFile)

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

    value = np.random.randn(1).item() * 100 + 1000
    assert np.round(NumCpp.logScaler(value), 9) == np.round(np.log(value), 9)

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.logScaler(value), 9) == np.round(np.log(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.logArray(cArray), 9), np.round(np.log(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.logArray(cArray), 9), np.round(np.log(data), 9))

    value = np.random.randn(1).item() * 100 + 1000
    assert np.round(NumCpp.log10Scaler(value), 9) == np.round(np.log10(value), 9)

    components = np.random.randn(2).astype(np.double) * 100 + 100
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.log10Scaler(value), 9) == np.round(np.log10(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.log10Array(cArray), 9), np.round(np.log10(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.log10Array(cArray), 9), np.round(np.log10(data), 9))

    value = np.random.randn(1).item() * 100 + 1000
    assert np.round(NumCpp.log1pScaler(value), 9) == np.round(np.log1p(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.log1pArray(cArray), 9), np.round(np.log1p(data), 9))

    value = np.random.randn(1).item() * 100 + 1000
    assert np.round(NumCpp.log2Scaler(value), 9) == np.round(np.log2(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.log2Array(cArray), 9), np.round(np.log2(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 20, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 20, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.logical_and(cArray1, cArray2).getNumpyArray(), np.logical_and(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 20, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.logical_not(cArray).getNumpyArray(), np.logical_not(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 20, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 20, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.logical_or(cArray1, cArray2).getNumpyArray(), np.logical_or(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 20, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 20, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.logical_xor(cArray1, cArray2).getNumpyArray(), np.logical_xor(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[1].item(), shapeInput[0].item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    data1 = np.random.randint(0, 20, [shape1.rows, shape1.cols])
    data2 = np.random.randint(0, 20, [shape2.rows, shape2.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.matmul(cArray1, cArray2), np.matmul(data1, data2))

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
    assert np.array_equal(NumCpp.matmul(cArray1, cArray2), np.matmul(data1, data2))

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

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.maximum(cArray1, cArray2), np.maximum(data1, data2))

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
    assert np.array_equal(NumCpp.maximum(cArray1, cArray2), np.maximum(data1, data2))

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

    isEven = True
    while isEven:
        shapeInput = np.random.randint(20, 100, [2, ])
        isEven = shapeInput.prod().item() % 2 == 0
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.median(cArray, NumCpp.Axis.NONE).getNumpyArray().flatten().item() == np.median(data, axis=None).item()

    isEven = True
    while isEven:
        shapeInput = np.random.randint(20, 100, [2, ])
        isEven = shapeInput.prod().item() % 2 == 0
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.median(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), np.median(data, axis=0))

    isEven = True
    while isEven:
        shapeInput = np.random.randint(20, 100, [2, ])
        isEven = shapeInput.prod().item() % 2 == 0
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.median(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), np.median(data, axis=1))

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
    meshData = NumCpp.meshgrid(iSlice, jSlice)
    assert np.array_equal(meshData.first.getNumpyArray(), iMesh)
    assert np.array_equal(meshData.second.getNumpyArray(), jMesh)

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

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.minimum(cArray1, cArray2), np.minimum(data1, data2))

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
    assert np.array_equal(NumCpp.minimum(cArray1, cArray2), np.minimum(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt32(shape)
    cArray2 = NumCpp.NdArrayUInt32(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.mod(cArray1, cArray2).getNumpyArray(), np.mod(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.multiply(cArray1, cArray2), np.multiply(data1, data2))

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
    assert np.array_equal(NumCpp.multiply(cArray1, cArray2), np.multiply(data1, data2))

    shapeInput = np.random.randint(10, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert NumCpp.nanargmax(cArray, NumCpp.Axis.NONE).item() == np.nanargmax(data)

    shapeInput = np.random.randint(10, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanargmax(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(),
                          np.nanargmax(data, axis=0))

    shapeInput = np.random.randint(10, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanargmax(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(),
                          np.nanargmax(data, axis=1))

    shapeInput = np.random.randint(10, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert NumCpp.nanargmin(cArray, NumCpp.Axis.NONE).item() == np.nanargmin(data)

    shapeInput = np.random.randint(10, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanargmin(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(),
                          np.nanargmin(data, axis=0))

    shapeInput = np.random.randint(10, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanargmin(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(),
                          np.nanargmin(data, axis=1))

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nancumprod(cArray, NumCpp.Axis.NONE).getNumpyArray().flatten(),
                          np.nancumprod(data, axis=None))

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nancumprod(cArray, NumCpp.Axis.ROW).getNumpyArray(), np.nancumprod(data, axis=0))

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nancumprod(cArray, NumCpp.Axis.COL).getNumpyArray(), np.nancumprod(data, axis=1))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nancumsum(cArray, NumCpp.Axis.NONE).getNumpyArray().flatten(),
                          np.nancumsum(data, axis=None))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nancumsum(cArray, NumCpp.Axis.ROW).getNumpyArray(), np.nancumsum(data, axis=0))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nancumsum(cArray, NumCpp.Axis.COL).getNumpyArray(), np.nancumsum(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert NumCpp.nanmax(cArray, NumCpp.Axis.NONE).item() == np.nanmax(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanmax(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(),
                          np.nanmax(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanmax(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(),
                          np.nanmax(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert NumCpp.nanmean(cArray, NumCpp.Axis.NONE).item() == np.nanmean(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanmean(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(),
                          np.nanmean(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanmean(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(),
                          np.nanmean(data, axis=1))

    isEven = True
    while isEven:
        shapeInput = np.random.randint(20, 100, [2, ])
        isEven = shapeInput.prod().item() % 2 == 0
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    # data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    # data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    # data = data.flatten()
    # data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    # data = data.reshape(shapeInput)
    # cArray.setArray(data)
    # assert np.array_equal(NumCpp.nanmedian(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(),
    # np.nanmedian(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert NumCpp.nanmin(cArray, NumCpp.Axis.NONE).item() == np.nanmin(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanmin(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(),
                          np.nanmin(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanmin(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(),
                          np.nanmin(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    percentile = np.random.rand(1).item() * 100
    assert np.array_equal(
            np.round(NumCpp.nanpercentile(cArray, percentile, NumCpp.Axis.COL, 'linear').getNumpyArray().flatten(), 9),
            np.round(np.nanpercentile(data, percentile, axis=1, interpolation='linear'), 9))

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert NumCpp.nanprod(cArray, NumCpp.Axis.NONE).item() == np.nanprod(data, axis=None)

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanprod(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), np.nanprod(data, axis=0))

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nanprod(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), np.nanprod(data, axis=1))

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

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.nans_like(cArray1)
    assert (cArray2.shape().rows == shape.rows and cArray2.shape().cols == shape.cols and
            cArray2.size() == shapeInput.prod() and np.all(np.isnan(cArray2.getNumpyArray())))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.round(NumCpp.nanstdev(cArray, NumCpp.Axis.NONE).item(), 9) == np.round(np.nanstd(data), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.nanstdev(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), 9),
                          np.round(np.nanstd(data, axis=0), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.nanstdev(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), 9),
                          np.round(np.nanstd(data, axis=1), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert NumCpp.nansum(cArray, NumCpp.Axis.NONE).item() == np.nansum(data)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nansum(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), np.nansum(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.nansum(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), np.nansum(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.round(NumCpp.nanvar(cArray, NumCpp.Axis.NONE).item(), 9) == np.round(np.nanvar(data), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.nanvar(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), 9),
                          np.round(np.nanvar(data, axis=0), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data = data.flatten()
    data[np.random.randint(0, shape.size(), [shape.size() // 10, ])] = np.nan
    data = data.reshape(shapeInput)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.nanvar(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), 9),
                          np.round(np.nanvar(data, axis=1), 9))

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

    value = np.random.randint(1, 100, [1, ]).item()
    assert (NumCpp.newbyteorderScaler(value, NumCpp.Endian.BIG) ==
            np.asarray([value], dtype=np.uint32).newbyteorder().item())

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.newbyteorderArray(cArray, NumCpp.Endian.BIG),
                          data.newbyteorder())

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.none(cArray, NumCpp.Axis.NONE).astype(np.bool).item() == np.logical_not(np.any(data).item())

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.none(cArray, NumCpp.Axis.NONE).astype(np.bool).item() == np.logical_not(np.any(data).item())

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.none(cArray, NumCpp.Axis.ROW).flatten().astype(np.bool),
                          np.logical_not(np.any(data, axis=0)))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.none(cArray, NumCpp.Axis.ROW).flatten().astype(np.bool),
                          np.logical_not(np.any(data, axis=0)))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.none(cArray, NumCpp.Axis.COL).flatten().astype(np.bool),
                          np.logical_not(np.any(data, axis=1)))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.none(cArray, NumCpp.Axis.COL).flatten().astype(np.bool),
                          np.logical_not(np.any(data, axis=1)))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    row, col = np.nonzero(data)
    rowCol = NumCpp.nonzero(cArray)
    rowC = rowCol.first.getNumpyArray().flatten()
    colC = rowCol.second.getNumpyArray().flatten()
    assert np.array_equal(rowC, row) and np.array_equal(colC, col)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    row, col = np.nonzero(data)
    rowCol = NumCpp.nonzero(cArray)
    rowC = rowCol.first.getNumpyArray().flatten()
    colC = rowCol.second.getNumpyArray().flatten()
    assert np.array_equal(rowC, row) and np.array_equal(colC, col)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.norm(cArray, NumCpp.Axis.NONE).flatten() == np.linalg.norm(data.flatten())

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

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.not_equal(cArray1, cArray2).getNumpyArray(), np.not_equal(data1, data2))

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
    assert np.array_equal(NumCpp.not_equal(cArray1, cArray2).getNumpyArray(), np.not_equal(data1, data2))

    shapeInput = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.onesSquare(shapeInput)
    assert (cArray.shape[0] == shapeInput and cArray.shape[1] == shapeInput and
            cArray.size == shapeInput ** 2 and np.all(cArray == 1))

    shapeInput = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.onesSquareComplex(shapeInput)
    assert (cArray.shape[0] == shapeInput and cArray.shape[1] == shapeInput and
            cArray.size == shapeInput ** 2 and np.all(cArray == np.complex(1, 0)))

    shapeInput = np.random.randint(20, 100, [2, ])
    cArray = NumCpp.onesRowCol(shapeInput[0].item(), shapeInput[1].item())
    assert (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod() and np.all(cArray == 1))

    shapeInput = np.random.randint(20, 100, [2, ])
    cArray = NumCpp.onesRowColComplex(shapeInput[0].item(), shapeInput[1].item())
    assert (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod() and np.all(cArray == np.complex(1, 0)))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.onesShape(shape)
    assert (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod() and np.all(cArray == 1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.onesShapeComplex(shape)
    assert (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod() and np.all(cArray == np.complex(1, 0)))

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
            cArray2.size() == shapeInput.prod() and np.all(cArray2.getNumpyArray() == np.complex(1, 0)))

    size = np.random.randint(1, 100, [1, ]).item()
    shape = NumCpp.Shape(1, size)
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    data2 = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.outer(cArray1, cArray2), np.outer(data1, data2))

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
    assert np.array_equal(NumCpp.outer(cArray1, cArray2), np.outer(data1, data2))

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
    padValue = np.random.randint(1, 100, [1, ]).item() + 1j * np.random.randint(1, 100, [1, ]).item()
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.pad(cArray, padWidth, padValue).getNumpyArray(),
                          np.pad(data, padWidth, mode='constant', constant_values=padValue))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    kthElement = np.random.randint(0, shapeInput.prod(), [1, ], dtype=np.uint32).item()
    partitionedArray = NumCpp.partition(cArray, kthElement, NumCpp.Axis.NONE).getNumpyArray().flatten()
    assert (np.all(partitionedArray[kthElement] <= partitionedArray[kthElement]) and
            np.all(partitionedArray[kthElement:] >= partitionedArray[kthElement]))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    kthElement = np.random.randint(0, shapeInput.prod(), [1, ], dtype=np.uint32).item()
    partitionedArray = NumCpp.partition(cArray, kthElement, NumCpp.Axis.NONE).getNumpyArray().flatten()
    assert (np.all(partitionedArray[kthElement] <= partitionedArray[kthElement]) and
            np.all(partitionedArray[kthElement:] >= partitionedArray[kthElement]))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    kthElement = np.random.randint(0, shapeInput[0], [1, ], dtype=np.uint32).item()
    partitionedArray = NumCpp.partition(cArray, kthElement, NumCpp.Axis.ROW).getNumpyArray().transpose()
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
    kthElement = np.random.randint(0, shapeInput[0], [1, ], dtype=np.uint32).item()
    partitionedArray = NumCpp.partition(cArray, kthElement, NumCpp.Axis.ROW).getNumpyArray().transpose()
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
    kthElement = np.random.randint(0, shapeInput[1], [1, ], dtype=np.uint32).item()
    partitionedArray = NumCpp.partition(cArray, kthElement, NumCpp.Axis.COL).getNumpyArray()
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
    kthElement = np.random.randint(0, shapeInput[1], [1, ], dtype=np.uint32).item()
    partitionedArray = NumCpp.partition(cArray, kthElement, NumCpp.Axis.COL).getNumpyArray()
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

    components = np.random.rand(2).astype(np.double)
    assert NumCpp.polarScaler(components[0], components[1])

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    magArray = NumCpp.NdArray(shape)
    angleArray = NumCpp.NdArray(shape)
    mag = np.random.rand(shape.rows, shape.cols)
    angle = np.random.rand(shape.rows, shape.cols)
    magArray.setArray(mag)
    angleArray.setArray(angle)
    assert NumCpp.polarArray(magArray, angleArray) is not None

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    exponent = np.random.randint(0, 5, [1, ]).item()
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.powerArrayScaler(cArray, exponent), 9),
                          np.round(np.power(data, exponent), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    exponent = np.random.randint(0, 5, [1, ]).item()
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.powerArrayScaler(cArray, exponent), 9),
                          np.round(np.power(data, exponent), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    cExponents = NumCpp.NdArrayUInt8(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    exponents = np.random.randint(0, 5, [shape.rows, shape.cols]).astype(np.uint8)
    cArray.setArray(data)
    cExponents.setArray(exponents)
    assert np.array_equal(np.round(NumCpp.powerArrayArray(cArray, cExponents), 9),
                          np.round(np.power(data, exponents), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cExponents = NumCpp.NdArrayUInt8(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    exponents = np.random.randint(0, 5, [shape.rows, shape.cols]).astype(np.uint8)
    cArray.setArray(data)
    cExponents.setArray(exponents)
    assert np.array_equal(np.round(NumCpp.powerArrayArray(cArray, cExponents), 9),
                          np.round(np.power(data, exponents), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    exponent = np.random.rand(1).item() * 3
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.powerfArrayScaler(cArray, exponent), 9),
                          np.round(np.power(data, exponent), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    exponent = np.random.rand(1).item() * 3
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.powerfArrayScaler(cArray, exponent), 9),
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
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    exponents = np.random.rand(shape.rows, shape.cols) * 3 + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    cExponents.setArray(exponents)
    assert np.array_equal(np.round(NumCpp.powerfArrayArray(cArray, cExponents), 9),
                          np.round(np.power(data, exponents), 9))

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert NumCpp.prod(cArray, NumCpp.Axis.NONE).item() == data.prod()

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 15, [shape.rows, shape.cols])
    imag = np.random.randint(1, 15, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.prod(cArray, NumCpp.Axis.NONE).item() == data.prod()

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.prod(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), data.prod(axis=0))

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 15, [shape.rows, shape.cols])
    imag = np.random.randint(1, 15, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.prod(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), data.prod(axis=0))

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.prod(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), data.prod(axis=1))

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 15, [shape.rows, shape.cols])
    imag = np.random.randint(1, 15, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.prod(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), data.prod(axis=1))

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert NumCpp.projScaler(value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cData = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cData.setArray(data)
    assert NumCpp.projArray(cData) is not None

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.ptp(cArray, NumCpp.Axis.NONE).getNumpyArray().item() == data.ptp()

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.ptp(cArray, NumCpp.Axis.NONE).getNumpyArray().item() == data.ptp()

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

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    numIndices = np.random.randint(0, shape.size())
    indices = np.asarray(range(numIndices))
    values = np.random.randint(1, 500, [numIndices, ])
    cIndices = NumCpp.NdArrayUInt32(1, numIndices)
    cValues = NumCpp.NdArray(1, numIndices)
    cIndices.setArray(indices)
    cValues.setArray(values)
    NumCpp.put(cArray, cIndices, cValues)
    data.put(indices, values)
    assert np.array_equal(cArray.getNumpyArray(), data)

    value = np.abs(np.random.rand(1).item()) * 2 * np.pi
    assert np.round(NumCpp.rad2degScaler(value), 9) == np.round(np.rad2deg(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 2 * np.pi
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.rad2degArray(cArray), 9), np.round(np.rad2deg(data), 9))

    value = np.abs(np.random.rand(1).item()) * 360
    assert np.round(NumCpp.radiansScaler(value), 9) == np.round(np.radians(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 360
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.radiansArray(cArray), 9), np.round(np.radians(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    cArray2 = NumCpp.ravel(cArray)
    assert np.array_equal(cArray2.getNumpyArray().flatten(), np.ravel(data))

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.realScaler(value), 9) == np.round(np.real(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.realArray(cArray), 9), np.round(np.real(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.reciprocal(cArray), 9), np.round(np.reciprocal(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.double)
    imag = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.double)
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.reciprocal(cArray), 9), np.round(np.reciprocal(data), 9))

    # numpy and cmath remainders are calculated differently, so convert for testing purposes
    values = np.random.rand(2) * 100
    values = np.sort(values)
    res = NumCpp.remainderScaler(values[1].item(), values[0].item())
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
    assert np.array_equal(np.round(res, 9), np.round(np.remainder(data1, data2), 9))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    oldValue = np.random.randint(1, 100, 1).item()
    newValue = np.random.randint(1, 100, 1).item()
    dataCopy = data.copy()
    dataCopy[dataCopy == oldValue] = newValue
    assert np.array_equal(NumCpp.replace(cArray, oldValue, newValue), dataCopy)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    oldValue = np.random.randint(1, 100, 1).item() + 1j * np.random.randint(1, 100, 1).item()
    newValue = np.random.randint(1, 100, 1).item() + 1j * np.random.randint(1, 100, 1).item()
    dataCopy = data.copy()
    dataCopy[dataCopy == oldValue] = newValue
    assert np.array_equal(NumCpp.replace(cArray, oldValue, newValue), dataCopy)

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
    assert np.array_equal(cArray.getNumpyArray(), data.reshape(shapeInput[::-1]))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    newShape = NumCpp.Shape(shapeInput[1].item(), shapeInput[0].item())
    NumCpp.reshapeList(cArray, newShape)
    assert np.array_equal(cArray.getNumpyArray(), data.reshape(shapeInput[::-1]))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    newNumCols = np.random.choice(np.array(list(factors(data.size))), 1).item()
    NumCpp.reshape(cArray, -1, newNumCols)
    assert np.array_equal(cArray.getNumpyArray(), data.reshape(-1, newNumCols))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    newNumRows = np.random.choice(np.array(list(factors(data.size))), 1).item()
    NumCpp.reshape(cArray, newNumRows, -1)
    assert np.array_equal(cArray.getNumpyArray(), data.reshape(newNumRows, -1))

    shapeInput1 = np.random.randint(1, 100, [2, ])
    shapeInput2 = np.random.randint(1, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput1[0].item(), shapeInput1[1].item())
    shape2 = NumCpp.Shape(shapeInput2[0].item(), shapeInput2[1].item())
    cArray = NumCpp.NdArray(shape1)
    data = np.random.randint(1, 100, [shape1.rows, shape1.cols], dtype=np.uint32)
    cArray.setArray(data)
    NumCpp.resizeFast(cArray, shape2)
    assert cArray.shape().rows == shape2.rows
    assert cArray.shape().cols == shape2.cols

    shapeInput1 = np.random.randint(1, 100, [2, ])
    shapeInput2 = np.random.randint(1, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput1[0].item(), shapeInput1[1].item())
    shape2 = NumCpp.Shape(shapeInput2[0].item(), shapeInput2[1].item())
    cArray = NumCpp.NdArray(shape1)
    data = np.random.randint(1, 100, [shape1.rows, shape1.cols], dtype=np.uint32)
    cArray.setArray(data)
    NumCpp.resizeSlow(cArray, shape2)
    assert cArray.shape().rows == shape2.rows
    assert cArray.shape().cols == shape2.cols

    shapeInput = np.random.randint(20, 100, [2, ])
    bitsToshift = np.random.randint(1, 32, [1, ]).item()
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(1, np.iinfo(np.uint32).max, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.right_shift(cArray, bitsToshift).getNumpyArray(),
                          np.right_shift(data, bitsToshift))

    value = np.abs(np.random.rand(1).item()) * 2 * np.pi
    assert NumCpp.rintScaler(value) == np.rint(value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 2 * np.pi
    cArray.setArray(data)
    assert np.array_equal(NumCpp.rintArray(cArray), np.rint(data))

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

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    amount = np.random.randint(1, 4, [1, ]).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.rot90(cArray, amount).getNumpyArray(), np.rot90(data, amount))

    value = np.abs(np.random.rand(1).item()) * 2 * np.pi
    assert NumCpp.roundScaler(value, 10) == np.round(value, 10)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 2 * np.pi
    cArray.setArray(data)
    assert np.array_equal(NumCpp.roundArray(cArray, 9), np.round(data, 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
    shape3 = NumCpp.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
    shape4 = NumCpp.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
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

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt32(shape)
    cArray2 = NumCpp.NdArrayUInt32(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.setdiff1d(cArray1, cArray2).getNumpyArray().flatten(),
                          np.setdiff1d(data1, data2))

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexUint32(shape)
    cArray2 = NumCpp.NdArrayComplexUint32(shape)
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

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    assert cArray.shape().rows == shape.rows and cArray.shape().cols == shape.cols

    value = np.random.randn(1).item() * 100
    assert NumCpp.signScaler(value) == np.sign(value)

    value = np.random.randn(1).item() * 100 + 1j * np.random.randn(1).item() * 100
    assert NumCpp.signScaler(value) == np.sign(value)

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

    value = np.random.randn(1).item() * 100
    assert NumCpp.signbitScaler(value) == np.signbit(value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    assert np.array_equal(NumCpp.signbitArray(cArray), np.signbit(data))

    value = np.random.randn(1).item()
    assert np.round(NumCpp.sinScaler(value), 9) == np.round(np.sin(value), 9)

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.sinScaler(value), 9) == np.round(np.sin(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.sinArray(cArray), 9), np.round(np.sin(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.sinArray(cArray), 9), np.round(np.sin(data), 9))

    value = np.random.randn(1)
    assert np.round(NumCpp.sincScaler(value.item()), 9) == np.round(np.sinc(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.sincArray(cArray), 9), np.round(np.sinc(data), 9))

    value = np.random.randn(1).item()
    assert np.round(NumCpp.sinhScaler(value), 9) == np.round(np.sinh(value), 9)

    value = np.random.randn(1).item() + 1j * np.random.randn(1).item()
    assert np.round(NumCpp.sinhScaler(value), 9) == np.round(np.sinh(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.sinhArray(cArray), 9), np.round(np.sinh(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.randn(shape.rows, shape.cols) + 1j * np.random.randn(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.sinhArray(cArray), 9), np.round(np.sinh(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    assert cArray.size() == shapeInput.prod().item()

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    d = data.flatten()
    d.sort()
    assert np.array_equal(NumCpp.sort(cArray, NumCpp.Axis.NONE).getNumpyArray().flatten(), d)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    d = data.flatten()
    d.sort()
    assert np.array_equal(NumCpp.sort(cArray, NumCpp.Axis.NONE).getNumpyArray().flatten(), d)

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

    value = np.random.randint(1, 100, [1, ]).item()
    assert np.round(NumCpp.sqrtScaler(value), 9) == np.round(np.sqrt(value), 9)

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.sqrtScaler(value), 9) == np.round(np.sqrt(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.sqrtArray(cArray), 9), np.round(np.sqrt(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.sqrtArray(cArray), 9), np.round(np.sqrt(data), 9))

    value = np.random.randint(1, 100, [1, ]).item()
    assert np.round(NumCpp.squareScaler(value), 9) == np.round(np.square(value), 9)

    value = np.random.randint(1, 100, [1, ]).item() + 1j * np.random.randint(1, 100, [1, ]).item()
    assert np.round(NumCpp.squareScaler(value), 9) == np.round(np.square(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.squareArray(cArray), 9), np.round(np.square(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.squareArray(cArray), 9), np.round(np.square(data), 9))

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

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.round(NumCpp.stdev(cArray, NumCpp.Axis.NONE).item(), 9) == np.round(np.std(data), 9)

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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.sum(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), np.sum(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.sum(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), np.sum(data, axis=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.sum(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), np.sum(data, axis=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.sum(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), np.sum(data, axis=1))

    shapeInput1 = np.random.randint(20, 100, [2, ])
    shapeInput2 = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput1[0].item(), shapeInput1[1].item())
    shape2 = NumCpp.Shape(shapeInput2[0].item(), shapeInput2[1].item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    data1 = np.random.randint(0, 100, [shape1.rows, shape1.cols]).astype(np.double)
    data2 = np.random.randint(0, 100, [shape2.rows, shape2.cols]).astype(np.double)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    NumCpp.swap(cArray1, cArray2)
    assert (np.array_equal(cArray1.getNumpyArray(), data2) and
            np.array_equal(cArray2.getNumpyArray(), data1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.swapaxes(cArray).getNumpyArray(), data.T)

    value = np.random.rand(1).item() * np.pi
    assert np.round(NumCpp.tanScaler(value), 9) == np.round(np.tan(value), 9)

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.tanScaler(value), 9) == np.round(np.tan(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.tanArray(cArray), 9), np.round(np.tan(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.tanArray(cArray), 9), np.round(np.tan(data), 9))

    value = np.random.rand(1).item() * np.pi
    assert np.round(NumCpp.tanhScaler(value), 9) == np.round(np.tanh(value), 9)

    components = np.random.rand(2).astype(np.double)
    value = np.complex(components[0], components[1])
    assert np.round(NumCpp.tanhScaler(value), 9) == np.round(np.tanh(value), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.tanhArray(cArray), 9), np.round(np.tanh(data), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1j * np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.tanhArray(cArray), 9), np.round(np.tanh(data), 9))

    shapeInput = np.random.randint(1, 10, [2, ])
    shapeRepeat = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shapeR = NumCpp.Shape(shapeRepeat[0].item(), shapeRepeat[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.tileRectangle(cArray, shapeR.rows, shapeR.cols), np.tile(data, shapeRepeat))

    shapeInput = np.random.randint(1, 10, [2, ])
    shapeRepeat = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shapeR = NumCpp.Shape(shapeRepeat[0].item(), shapeRepeat[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.tileShape(cArray, shapeR), np.tile(data, shapeRepeat))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if sys.platform == 'linux':
        tempDir = r'/home/' + getpass.getuser() + r'/Desktop/'
        filename = os.path.join(tempDir, 'temp.bin')
    else:
        filename = r'C:\Temp\temp.bin'
    NumCpp.tofile(cArray, filename, '')
    assert os.path.exists(filename)
    data2 = np.fromfile(filename, np.double).reshape(shapeInput)
    assert np.array_equal(data, data2)
    os.remove(filename)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if sys.platform == 'linux':
        tempDir = r'/home/' + getpass.getuser() + r'/Desktop/'
        filename = os.path.join(tempDir, 'temp.txt')
    else:
        filename = r'C:\Temp\temp.txt'
    NumCpp.tofile(cArray, filename, '\n')
    assert os.path.exists(filename)
    data2 = np.fromfile(filename, dtype=np.double, sep='\n').reshape(shapeInput)
    assert np.array_equal(data, data2)
    os.remove(filename)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    out = np.asarray(NumCpp.toStlVector(cArray))
    assert np.array_equal(out, data.flatten())

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows, [1, ]).item()
    assert np.array_equal(NumCpp.trace(cArray, offset, NumCpp.Axis.ROW), data.trace(offset, axis1=1, axis2=0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows, [1, ]).item()
    assert np.array_equal(NumCpp.trace(cArray, offset, NumCpp.Axis.COL), data.trace(offset, axis1=0, axis2=1))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.transpose(cArray).getNumpyArray(), np.transpose(data))

    shape = NumCpp.Shape(np.random.randint(10, 20, [1, ]).item(), 1)
    cArray = NumCpp.NdArray(shape)
    coeffs = np.random.randint(0, 10, [2, ])
    dx = np.random.rand(1).item()
    data = np.array([x ** 2 - coeffs[0] * x + coeffs[1] for x in range(shape.size())])
    cArray.setArray(data)
    integralC = NumCpp.trapzDx(cArray, dx, NumCpp.Axis.NONE).item()
    integralPy = np.trapz(data, dx=dx)
    assert np.round(integralC, 9) == np.round(integralPy, 9)

    shape = NumCpp.Shape(np.random.randint(10, 20, [1, ]).item(), np.random.randint(10, 20, [1, ]).item())
    cArray = NumCpp.NdArray(shape)
    coeffs = np.random.randint(0, 10, [2, ])
    dx = np.random.rand(1).item()
    data = np.array([x ** 2 - coeffs[0] * x - coeffs[1] for x in range(shape.size())]).reshape(shape.rows, shape.cols)
    cArray.setArray(data)
    integralC = NumCpp.trapzDx(cArray, dx, NumCpp.Axis.ROW).flatten()
    integralPy = np.trapz(data, dx=dx, axis=0)
    assert np.array_equal(np.round(integralC, 8), np.round(integralPy, 8))

    shape = NumCpp.Shape(np.random.randint(10, 20, [1, ]).item(), np.random.randint(10, 20, [1, ]).item())
    cArray = NumCpp.NdArray(shape)
    coeffs = np.random.randint(0, 10, [2, ])
    dx = np.random.rand(1).item()
    data = np.array([x ** 2 - coeffs[0] * x - coeffs[1] for x in range(shape.size())]).reshape(shape.rows, shape.cols)
    cArray.setArray(data)
    integralC = NumCpp.trapzDx(cArray, dx, NumCpp.Axis.COL).flatten()
    integralPy = np.trapz(data, dx=dx, axis=1)
    assert np.array_equal(np.round(integralC, 8), np.round(integralPy, 8))

    shape = NumCpp.Shape(1, np.random.randint(10, 20, [1, ]).item())
    cArrayY = NumCpp.NdArray(shape)
    cArrayX = NumCpp.NdArray(shape)
    coeffs = np.random.randint(0, 10, [2, ])
    dx = np.random.rand(shape.rows, shape.cols)
    data = np.array([x ** 2 - coeffs[0] * x + coeffs[1] for x in range(shape.size())])
    cArrayY.setArray(data)
    cArrayX.setArray(dx)
    integralC = NumCpp.trapz(cArrayY, cArrayX, NumCpp.Axis.NONE).item()
    integralPy = np.trapz(data, x=dx)
    assert np.round(integralC, 9) == np.round(integralPy, 9)

    shape = NumCpp.Shape(np.random.randint(10, 20, [1, ]).item(), np.random.randint(10, 20, [1, ]).item())
    cArrayY = NumCpp.NdArray(shape)
    cArrayX = NumCpp.NdArray(shape)
    coeffs = np.random.randint(0, 10, [2, ])
    dx = np.random.rand(shape.rows, shape.cols)
    data = np.array([x ** 2 - coeffs[0] * x + coeffs[1] for x in range(shape.size())]).reshape(shape.rows, shape.cols)
    cArrayY.setArray(data)
    cArrayX.setArray(dx)
    integralC = NumCpp.trapz(cArrayY, cArrayX, NumCpp.Axis.ROW).flatten()
    integralPy = np.trapz(data, x=dx, axis=0)
    assert np.array_equal(np.round(integralC, 8), np.round(integralPy, 8))

    shape = NumCpp.Shape(np.random.randint(10, 20, [1, ]).item(), np.random.randint(10, 20, [1, ]).item())
    cArrayY = NumCpp.NdArray(shape)
    cArrayX = NumCpp.NdArray(shape)
    coeffs = np.random.randint(0, 10, [2, ])
    dx = np.random.rand(shape.rows, shape.cols)
    data = np.array([x ** 2 - coeffs[0] * x + coeffs[1] for x in range(shape.size())]).reshape(shape.rows, shape.cols)
    cArrayY.setArray(data)
    cArrayX.setArray(dx)
    integralC = NumCpp.trapz(cArrayY, cArrayX, NumCpp.Axis.COL).flatten()
    integralPy = np.trapz(data, x=dx, axis=1)
    assert np.array_equal(np.round(integralC, 8), np.round(integralPy, 8))

    squareSize = np.random.randint(10, 100, [1, ]).item()
    offset = np.random.randint(0, squareSize, [1, ]).item()
    assert np.array_equal(NumCpp.trilSquare(squareSize, offset),
                          np.tri(squareSize, k=offset))

    squareSize = np.random.randint(10, 100, [1, ]).item()
    offset = np.random.randint(0, squareSize, [1, ]).item()
    assert np.array_equal(NumCpp.trilSquareComplex(squareSize, offset),
                          np.tri(squareSize, k=offset) + 1j * np.zeros([squareSize, squareSize]))

    shapeInput = np.random.randint(10, 100, [2, ])
    offset = np.random.randint(0, np.min(shapeInput), [1, ]).item()
    assert np.array_equal(NumCpp.trilRect(shapeInput[0].item(), shapeInput[1].item(), offset),
                          np.tri(shapeInput[0].item(), shapeInput[1].item(), k=offset))

    shapeInput = np.random.randint(10, 100, [2, ])
    offset = np.random.randint(0, np.min(shapeInput), [1, ]).item()
    assert np.array_equal(NumCpp.trilRectComplex(shapeInput[0].item(), shapeInput[1].item(), offset),
                          np.tri(shapeInput[0].item(), shapeInput[1].item(), k=offset) + 1j * np.zeros(shapeInput))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows, [1, ]).item()
    assert np.array_equal(NumCpp.trilArray(cArray, offset),
                          np.tril(data, k=offset))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows, [1, ]).item()
    assert np.array_equal(NumCpp.trilArray(cArray, offset),
                          np.tril(data, k=offset))

    squareSize = np.random.randint(10, 100, [1, ]).item()
    offset = np.random.randint(0, squareSize, [1, ]).item()
    assert np.array_equal(NumCpp.triuSquare(squareSize, offset),
                          np.tri(squareSize, k=-offset).T)

    squareSize = np.random.randint(10, 100, [1, ]).item()
    offset = np.random.randint(0, squareSize, [1, ]).item()
    assert np.array_equal(NumCpp.triuSquareComplex(squareSize, offset),
                          np.tri(squareSize, k=-offset).T + 1j * np.zeros([squareSize, squareSize]))

    shapeInput = np.random.randint(10, 100, [2, ])
    offset = np.random.randint(0, np.min(shapeInput), [1, ]).item()
    # NOTE: numpy triu appears to have a bug... just check that NumCpp runs without error
    assert NumCpp.triuRect(shapeInput[0].item(), shapeInput[1].item(), offset) is not None

    shapeInput = np.random.randint(10, 100, [2, ])
    offset = np.random.randint(0, np.min(shapeInput), [1, ]).item()
    # NOTE: numpy triu appears to have a bug... just check that NumCpp runs without error
    assert NumCpp.triuRectComplex(shapeInput[0].item(), shapeInput[1].item(), offset) is not None

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows, [1, ]).item()
    assert np.array_equal(NumCpp.triuArray(cArray, offset), np.triu(data, k=offset))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows, [1, ]).item()
    assert np.array_equal(NumCpp.triuArray(cArray, offset), np.triu(data, k=offset))
        
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
    data[0, :offsetBeg] = np.complex(0, 0)
    data[0, -offsetEnd:] = np.complex(0, 0)
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
    data[0, :offsetBeg] = np.complex(0, 0)
    data[0, -offsetEnd:] = np.complex(0, 0)
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
    data[0, :offsetBeg] = np.complex(0, 0)
    data[0, -offsetEnd:] = np.complex(0, 0)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.trim_zeros(cArray, 'fb').getNumpyArray().flatten(),
                          np.trim_zeros(data.flatten(), 'fb'))

    value = np.random.rand(1).item() * np.pi
    assert NumCpp.truncScaler(value) == np.trunc(value)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.truncArray(cArray), np.trunc(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt32(shape)
    cArray2 = NumCpp.NdArrayUInt32(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.union1d(cArray1, cArray2).getNumpyArray().flatten(), np.union1d(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexUint32(shape)
    cArray2 = NumCpp.NdArrayComplexUint32(shape)
    real1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.union1d(cArray1, cArray2).getNumpyArray().flatten(), np.union1d(data1, data2))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.unique(cArray).getNumpyArray().flatten(), np.unique(data))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexUint32(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.unique(cArray).getNumpyArray().flatten(), np.unique(data))

    value = np.random.randn(1).item() * 3 * np.pi
    assert np.round(NumCpp.unwrapScaler(value), 9) == np.round(np.arctan2(np.sin(value), np.cos(value)), 9)

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.unwrapArray(cArray), 9), np.round(np.arctan2(np.sin(data), np.cos(data)), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.round(NumCpp.var(cArray, NumCpp.Axis.NONE).item(), 9) == np.round(np.var(data), 9)

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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.var(cArray, NumCpp.Axis.ROW).getNumpyArray().flatten(), 9),
                          np.round(np.var(data, axis=0), 9))

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
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.var(cArray, NumCpp.Axis.COL).getNumpyArray().flatten(), 9),
                          np.round(np.var(data, axis=1), 9))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert NumCpp.var(cArray, NumCpp.Axis.COL) is not None

    shapeInput = np.random.randint(20, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
    shape3 = NumCpp.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
    shape4 = NumCpp.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1, ]).item(), shapeInput[1].item())
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
    assert np.array_equal(NumCpp.where(cArrayMask, cArrayA, cArrayB), np.where(dataMask, dataA, dataB))

    shapeInput = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.zerosSquare(shapeInput)
    assert (cArray.shape[0] == shapeInput and cArray.shape[1] == shapeInput and
            cArray.size == shapeInput ** 2 and np.all(cArray == 0))

    shapeInput = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.zerosSquareComplex(shapeInput)
    assert (cArray.shape[0] == shapeInput and cArray.shape[1] == shapeInput and
            cArray.size == shapeInput ** 2 and np.all(cArray == np.complex(0, 0)))

    shapeInput = np.random.randint(20, 100, [2, ])
    cArray = NumCpp.zerosRowCol(shapeInput[0].item(), shapeInput[1].item())
    assert (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod() and np.all(cArray == 0))

    shapeInput = np.random.randint(20, 100, [2, ])
    cArray = NumCpp.zerosRowColComplex(shapeInput[0].item(), shapeInput[1].item())
    assert (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod() and np.all(cArray == np.complex(0, 0)))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.zerosShape(shape)
    assert (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod() and np.all(cArray == 0))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.zerosShapeComplex(shape)
    assert (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod() and np.all(cArray == np.complex(0, 0)))

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
            cArray2.size() == shapeInput.prod() and np.all(cArray2.getNumpyArray() == np.complex(0, 0)))
