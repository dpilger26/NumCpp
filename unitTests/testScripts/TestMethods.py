import os
import numpy as np
from termcolor import colored
import sys
sys.path.append(r'../build/x64/Release')
import NumC

####################################################################################
def doTest():
    print(colored('Testing Methods Module', 'magenta'))

    print(colored('Testing abs scalar', 'cyan'))
    randValue = np.random.randint(-100, -1, [1,]).astype(np.double).item()
    if NumC.abs(randValue) == np.abs(randValue):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing abs array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.abs(cArray), np.abs(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing add', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.add(cArray1, cArray2), data1 + data2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing alen array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if NumC.alen(cArray) == shape.rows:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing all: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if NumC.all(cArray, NumC.Axis.NONE).astype(np.bool).item() == np.all(data).item():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing all: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.all(cArray, NumC.Axis.ROW).flatten().astype(np.bool), np.all(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing all: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.all(cArray, NumC.Axis.COL).flatten().astype(np.bool), np.all(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing allclose', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    cArray3 = NumC.NdArray(shape)
    tolerance = 1e-5
    data1 = np.random.randn(shape.rows, shape.cols)
    data2 = data1 + tolerance / 10
    data3 = data1 + 1
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    if NumC.allclose(cArray1, cArray2, tolerance) and not NumC.allclose(cArray1, cArray3, tolerance):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing amax: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if NumC.amax(cArray, NumC.Axis.NONE).item() == np.max(data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing amax: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.amax(cArray, NumC.Axis.ROW).flatten(), np.max(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing amax: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.amax(cArray, NumC.Axis.COL).flatten(), np.max(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing amin: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if NumC.amin(cArray, NumC.Axis.NONE).item() == np.min(data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing amin: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.amin(cArray, NumC.Axis.ROW).flatten(), np.min(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing amin: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.amin(cArray, NumC.Axis.COL).flatten(), np.min(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing any: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if NumC.any(cArray, NumC.Axis.NONE).astype(np.bool).item() == np.any(data).item():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing any: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.any(cArray, NumC.Axis.ROW).flatten().astype(np.bool), np.any(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing any: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.any(cArray, NumC.Axis.COL).flatten().astype(np.bool), np.any(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing append: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.append(cArray1, cArray2, NumC.Axis.NONE).getNumpyArray().flatten(), np.append(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing append: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    numRows = np.random.randint(1, 100, [1, ]).item()
    shape1 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumC.Shape(shapeInput[0].item() + numRows, shapeInput[1].item())
    cArray1 = NumC.NdArray(shape1)
    cArray2 = NumC.NdArray(shape2)
    data1 = np.random.randint(0, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(0, 100, [shape2.rows, shape2.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.append(cArray1, cArray2, NumC.Axis.ROW).getNumpyArray(), np.append(data1, data2, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing append: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    numCols = np.random.randint(1, 100, [1, ]).item()
    shape1 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item() + numCols)
    cArray1 = NumC.NdArray(shape1)
    cArray2 = NumC.NdArray(shape2)
    data1 = np.random.randint(0, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(0, 100, [shape2.rows, shape2.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.append(cArray1, cArray2, NumC.Axis.COL).getNumpyArray(), np.append(data1, data2, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing arange', 'cyan'))
    start = np.random.randn(1).item()
    stop = np.random.randn(1).item() * 100
    step = np.abs(np.random.randn(1).item())
    if stop < start:
        step *= -1
    data = np.arange(start, stop, step)
    if np.array_equal(np.round(NumC.arange(start, stop, step).flatten(), 10), np.round(data, 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing arccos scalar', 'cyan'))
    value = np.abs(np.random.rand(1).item())
    if np.round(NumC.arccos(value), 10) == np.round(np.arccos(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing arccos array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.arccos(cArray), 10), np.round(np.arccos(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing arccosh scalar', 'cyan'))
    value = np.abs(np.random.rand(1).item()) + 1
    if np.round(NumC.arccosh(value), 10) == np.round(np.arccosh(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing arccosh array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) + 1
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.arccosh(cArray), 10), np.round(np.arccosh(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing arcsin scalar', 'cyan'))
    value = np.abs(np.random.rand(1).item())
    if np.round(NumC.arcsin(value), 10) == np.round(np.arcsin(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing arcsin array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.arcsin(cArray), 10), np.round(np.arcsin(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing arcsinh scalar', 'cyan'))
    value = np.abs(np.random.rand(1).item())
    if np.round(NumC.arcsinh(value), 10) == np.round(np.arcsinh(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing arcsinh array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.arcsinh(cArray), 10), np.round(np.arcsinh(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing arctan scalar', 'cyan'))
    value = np.abs(np.random.rand(1).item())
    if np.round(NumC.arctan(value), 10) == np.round(np.arctan(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing arctan array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.arctan(cArray), 10), np.round(np.arctan(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing arctan2 scalar', 'cyan'))
    xy = NumC.uniformOnSphere(1, 2).getNumpyArray().flatten()
    if np.round(NumC.arctan2(xy[1], xy[0]), 10) == np.round(np.arctan2(xy[1], xy[0]), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing arctan2 array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArrayX = NumC.NdArray(shape)
    cArrayY = NumC.NdArray(shape)
    xy = NumC.uniformOnSphere(np.prod(shapeInput).item(), 2).getNumpyArray()
    xData = xy[:, 0].reshape(shapeInput)
    yData = xy[:, 1].reshape(shapeInput)
    cArrayX.setArray(xData)
    cArrayY.setArray(yData)
    if np.array_equal(np.round(NumC.arctan2(cArrayY, cArrayX), 10), np.round(np.arctan2(yData, xData), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing arctanh scalar', 'cyan'))
    value = np.abs(np.random.rand(1).item())
    if np.round(NumC.arctanh(value), 10) == np.round(np.arctanh(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing arctanh array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.arctanh(cArray), 10), np.round(np.arctanh(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argmax: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.argmax(cArray, NumC.Axis.NONE).item(), np.argmax(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argmax: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.argmax(cArray, NumC.Axis.ROW).flatten(), np.argmax(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argmax: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.argmax(cArray, NumC.Axis.COL).flatten(), np.argmax(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argmin: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.argmin(cArray, NumC.Axis.NONE).item(), np.argmin(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argmin: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.argmin(cArray, NumC.Axis.ROW).flatten(), np.argmin(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argmin: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.argmin(cArray, NumC.Axis.COL).flatten(), np.argmin(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argsort: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    dataFlat = data.flatten()
    if np.array_equal(dataFlat[NumC.argsort(cArray, NumC.Axis.NONE).flatten().astype(np.uint32)], dataFlat[np.argsort(data, axis=None)]):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argsort: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    pIdx = np.argsort(data, axis=0)
    cIdx = NumC.argsort(cArray, NumC.Axis.ROW).astype(np.uint16)
    allPass = True
    for idx, row in enumerate(data.T):
        if not np.array_equal(row[cIdx[:, idx]], row[pIdx[:, idx]]):
            allPass = False
            break
    if allPass:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argsort: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    pIdx = np.argsort(data, axis=1)
    cIdx = NumC.argsort(cArray, NumC.Axis.COL).astype(np.uint16)
    allPass = True
    for idx, row in enumerate(data):
        if not np.array_equal(row[cIdx[idx, :]], row[pIdx[idx, :]]):
            allPass = False
            break
    if allPass:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing around scalar', 'cyan'))
    value = np.abs(np.random.rand(1).item()) * np.random.randint(1, 10, [1,]).item()
    numDecimalsRound = np.random.randint(0, 10, [1,]).astype(np.uint8).item()
    if NumC.around(value, numDecimalsRound) == np.round(value, numDecimalsRound):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing around array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * np.random.randint(1, 10, [1,]).item()
    cArray.setArray(data)
    numDecimalsRound = np.random.randint(0, 10, [1,]).astype(np.uint8).item()
    if np.array_equal(NumC.around(cArray, numDecimalsRound), np.round(data, numDecimalsRound)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing array_equal', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    cArray3 = NumC.NdArray(shape)
    data1 = np.random.randint(1, 100, shapeInput)
    data2 = np.random.randint(1, 100, shapeInput)
    cArray1.setArray(data1)
    cArray2.setArray(data1)
    cArray3.setArray(data2)
    if NumC.array_equal(cArray1, cArray2) and not NumC.array_equal(cArray1, cArray3):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing array_equiv', 'cyan'))
    shapeInput1 = np.random.randint(1, 100, [2, ])
    shapeInput3 = np.random.randint(1, 100, [2, ])
    shape1 = NumC.Shape(shapeInput1[0].item(), shapeInput1[1].item())
    shape2 = NumC.Shape(shapeInput1[1].item(), shapeInput1[0].item())
    shape3 = NumC.Shape(shapeInput3[0].item(), shapeInput3[1].item())
    cArray1 = NumC.NdArray(shape1)
    cArray2 = NumC.NdArray(shape2)
    cArray3 = NumC.NdArray(shape3)
    data1 = np.random.randint(1, 100, shapeInput1)
    data3 = np.random.randint(1, 100, shapeInput3)
    cArray1.setArray(data1)
    cArray2.setArray(data1.reshape([shapeInput1[1].item(), shapeInput1[0].item()]))
    cArray3.setArray(data3)
    if NumC.array_equiv(cArray1, cArray2) and not NumC.array_equiv(cArray1, cArray3):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing average: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if NumC.average(cArray, NumC.Axis.NONE).item() == np.average(data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing average: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.average(cArray, NumC.Axis.ROW).flatten(), np.average(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing average: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.average(cArray, NumC.Axis.COL).flatten(), np.average(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing average weighted: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    cWeights = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    weights = np.random.randint(1, 5, [shape.rows, shape.cols])
    cArray.setArray(data)
    cWeights.setArray(weights)
    if NumC.average(cArray, cWeights, NumC.Axis.NONE).item() == np.average(data, weights=weights):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing average weighted: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    cWeights = NumC.NdArray(1, shape.cols)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    weights = np.random.randint(1, 5, [1, shape.rows])
    cArray.setArray(data)
    cWeights.setArray(weights)
    if np.array_equal(NumC.average(cArray, cWeights, NumC.Axis.ROW).flatten(), np.average(data, weights=weights.flatten(), axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing average weighted: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    cWeights = NumC.NdArray(1, shape.rows)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    weights = np.random.randint(1, 5, [1, shape.cols])
    cWeights.setArray(weights)
    cArray.setArray(data)
    if np.array_equal(NumC.average(cArray, cWeights, NumC.Axis.COL).flatten(), np.average(data, weights=weights.flatten(), axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bincount', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArrayInt(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    cArray.setArray(data)
    if np.array_equal(NumC.bincount(cArray, 0).flatten(), np.bincount(data.flatten(), minlength=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bincount with minLength', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArrayInt(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    cArray.setArray(data)
    minLength = int(data.max() + 10)
    if np.array_equal(NumC.bincount(cArray, minLength).flatten(), np.bincount(data.flatten(), minlength=minLength)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bincount weighted', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArrayInt(shape)
    cWeights = NumC.NdArrayInt(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    weights = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    cArray.setArray(data)
    cWeights.setArray(weights)
    if np.array_equal(NumC.bincount(cArray, cWeights, 0).flatten(), np.bincount(data.flatten(), minlength=0, weights=weights.flatten())):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bincount weighted with minLength', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArrayInt(shape)
    cWeights = NumC.NdArrayInt(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    weights = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint16)
    cArray.setArray(data)
    cWeights.setArray(weights)
    minLength = int(data.max() + 10)
    if np.array_equal(NumC.bincount(cArray, cWeights, minLength).flatten(), np.bincount(data.flatten(), minlength=minLength, weights=weights.flatten())):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bitwise_and', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArrayInt64(shape)
    cArray2 = NumC.NdArrayInt64(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.bitwise_and(cArray1, cArray2), np.bitwise_and(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bitwise_not', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArrayInt64(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    cArray.setArray(data)
    if np.array_equal(NumC.bitwise_not(cArray), np.bitwise_not(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bitwise_or', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArrayInt64(shape)
    cArray2 = NumC.NdArrayInt64(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.bitwise_or(cArray1, cArray2), np.bitwise_or(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bitwise_xor', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArrayInt64(shape)
    cArray2 = NumC.NdArrayInt64(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint64)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.bitwise_xor(cArray1, cArray2), np.bitwise_xor(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cbrt', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.cbrt(cArray), 10), np.round(np.cbrt(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing ceil', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols).astype(np.double) * 1000
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.ceil(cArray), 10), np.round(np.ceil(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing column_stack', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape1 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1,]).item())
    shape3 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1,]).item())
    shape4 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1,]).item())
    cArray1 = NumC.NdArray(shape1)
    cArray2 = NumC.NdArray(shape2)
    cArray3 = NumC.NdArray(shape3)
    cArray4 = NumC.NdArray(shape4)
    data1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    data4 = np.random.randint(1, 100, [shape4.rows, shape4.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    if np.array_equal(NumC.column_stack(cArray1, cArray2, cArray3, cArray4),
                      np.column_stack([data1, data2, data3, data4])):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing concatenate: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape1 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1,]).item())
    shape3 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1,]).item())
    shape4 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1,]).item())
    cArray1 = NumC.NdArray(shape1)
    cArray2 = NumC.NdArray(shape2)
    cArray3 = NumC.NdArray(shape3)
    cArray4 = NumC.NdArray(shape4)
    data1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    data4 = np.random.randint(1, 100, [shape4.rows, shape4.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    if np.array_equal(NumC.concatenate(cArray1, cArray2, cArray3, cArray4, NumC.Axis.NONE).flatten(),
                      np.concatenate([data1.flatten(), data2.flatten(), data3.flatten(), data4.flatten()])):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing concatenate: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape1 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumC.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1,]).item(), shapeInput[1].item())
    shape3 = NumC.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1,]).item(), shapeInput[1].item())
    shape4 = NumC.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1,]).item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape1)
    cArray2 = NumC.NdArray(shape2)
    cArray3 = NumC.NdArray(shape3)
    cArray4 = NumC.NdArray(shape4)
    data1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    data4 = np.random.randint(1, 100, [shape4.rows, shape4.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    if np.array_equal(NumC.concatenate(cArray1, cArray2, cArray3, cArray4, NumC.Axis.ROW),
                      np.concatenate([data1, data2, data3, data4], axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing concatenate: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape1 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1,]).item())
    shape3 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1,]).item())
    shape4 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1,]).item())
    cArray1 = NumC.NdArray(shape1)
    cArray2 = NumC.NdArray(shape2)
    cArray3 = NumC.NdArray(shape3)
    cArray4 = NumC.NdArray(shape4)
    data1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    data4 = np.random.randint(1, 100, [shape4.rows, shape4.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    if np.array_equal(NumC.concatenate(cArray1, cArray2, cArray3, cArray4, NumC.Axis.COL),
                      np.concatenate([data1, data2, data3, data4], axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing clip scalar', 'cyan'))
    value = np.random.randint(0, 100, [1, ]).item()
    minValue = np.random.randint(0, 10, [1, ]).item()
    maxValue = np.random.randint(90, 100, [1, ]).item()
    if NumC.clip(value, minValue, maxValue) == np.clip(value, minValue, maxValue):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing clip array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    minValue = np.random.randint(0, 10, [1, ]).item()
    maxValue = np.random.randint(90, 100, [1, ]).item()
    if np.array_equal(NumC.clip(cArray, minValue, maxValue), np.clip(data, minValue, maxValue)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing contains: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    value = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    if NumC.contains(cArray, value, NumC.Axis.NONE).getNumpyArray().item() == (value in data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing contains: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    value = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    truth = list()
    for row in data:
        truth.append(value in row)
    if np.array_equal(NumC.contains(cArray, value, NumC.Axis.COL).getNumpyArray().flatten(), np.asarray(truth)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing contains: Axis = ROW', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    value = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    truth = list()
    for row in data.T:
        truth.append(value in row)
    if np.array_equal(NumC.contains(cArray, value, NumC.Axis.ROW).getNumpyArray().flatten(), np.asarray(truth)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing copy', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.copy(cArray), data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing copysign', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.copysign(cArray1, cArray2), np.copysign(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing copyto', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray()
    data1 = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    if np.array_equal(NumC.copyto(cArray2, cArray1), data1):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cos scalar', 'cyan'))
    value = np.abs(np.random.rand(1).item())
    if np.round(NumC.cos(value), 10) == np.round(np.cos(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cos array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.cos(cArray), 10), np.round(np.cos(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cosh scalar', 'cyan'))
    value = np.abs(np.random.rand(1).item())
    if np.round(NumC.cos(value), 10) == np.round(np.cos(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cosh array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.cosh(cArray), 10), np.round(np.cosh(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing count_nonzero: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 3, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if NumC.count_nonzero(cArray, NumC.Axis.NONE) == np.count_nonzero(data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing count_nonzero: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 3, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.count_nonzero(cArray, NumC.Axis.ROW).flatten(), np.count_nonzero(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing count_nonzero: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 3, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.count_nonzero(cArray, NumC.Axis.COL).flatten(), np.count_nonzero(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cross 2D: Axis = None', 'cyan'))
    shape = NumC.Shape(1, 2)
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if NumC.cross(cArray1, cArray2, NumC.Axis.NONE).item() == np.cross(data1, data2).item():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cross 2D: Axis = ROW', 'cyan'))
    shape = NumC.Shape(2, np.random.randint(1,100, [1, ]).item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.cross(cArray1, cArray2, NumC.Axis.ROW).getNumpyArray().flatten(), np.cross(data1, data2, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cross 2D: Axis = COL', 'cyan'))
    shape = NumC.Shape(np.random.randint(1,100, [1, ]).item(), 2)
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.cross(cArray1, cArray2, NumC.Axis.COL).getNumpyArray().flatten(), np.cross(data1, data2, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cross 3D: Axis = None', 'cyan'))
    shape = NumC.Shape(1, 3)
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.cross(cArray1, cArray2, NumC.Axis.NONE).getNumpyArray().flatten(), np.cross(data1, data2).flatten()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cross 3D: Axis = ROW', 'cyan'))
    shape = NumC.Shape(3, np.random.randint(1,100, [1, ]).item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.cross(cArray1, cArray2, NumC.Axis.ROW).getNumpyArray(), np.cross(data1, data2, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cross 3D: Axis = COL', 'cyan'))
    shape = NumC.Shape(np.random.randint(1,100, [1, ]).item(), 3)
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(1, 10, [shape.rows, shape.cols]).astype(np.double)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.cross(cArray1, cArray2, NumC.Axis.COL).getNumpyArray(), np.cross(data1, data2, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cube array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.cube(cArray), 10), np.round(data * data * data, 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cumprod: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.cumprod(cArray, NumC.Axis.NONE).flatten().astype(np.uint32), data.cumprod()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cumprod: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.cumprod(cArray, NumC.Axis.ROW).astype(np.uint32), data.cumprod(axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cumprod: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.cumprod(cArray, NumC.Axis.COL).astype(np.uint32), data.cumprod(axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cumsum: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.cumsum(cArray, NumC.Axis.NONE).flatten().astype(np.uint32), data.cumsum()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cumsum: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.cumsum(cArray, NumC.Axis.ROW).astype(np.uint32), data.cumsum(axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cumsum: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.cumsum(cArray, NumC.Axis.COL).astype(np.uint32), data.cumsum(axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing deg2rad scalar', 'cyan'))
    value = np.abs(np.random.rand(1).item()) * 360
    if np.round(NumC.deg2rad(value), 10) == np.round(np.deg2rad(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing deg2rad array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 360
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.deg2rad(cArray), 10), np.round(np.deg2rad(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing deleteIndices Slice: Axis = NONE', 'cyan'))
    shapeInput = np.asarray([100,100])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    indices = NumC.Slice(0, 100, 4)
    indicesPy = slice(0, 99, 4)
    cArray.setArray(data)
    if np.array_equal(NumC.delete(cArray, indices, NumC.Axis.NONE).flatten(), np.delete(data, indicesPy, axis=None)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing deleteIndices Slice: Axis = Row', 'cyan'))
    shapeInput = np.asarray([100,100])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    indices = NumC.Slice(0, 100, 4)
    indicesPy = slice(0, 99, 4)
    cArray.setArray(data)
    if np.array_equal(NumC.delete(cArray, indices, NumC.Axis.ROW), np.delete(data, indicesPy, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing deleteIndices Slice: Axis = Col', 'cyan'))
    shapeInput = np.asarray([100,100])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    indices = NumC.Slice(0, 100, 4)
    indicesPy = slice(0, 99, 4)
    cArray.setArray(data)
    if np.array_equal(NumC.delete(cArray, indices, NumC.Axis.COL), np.delete(data, indicesPy, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing deleteIndices Scalar: Axis = NONE', 'cyan'))
    shapeInput = np.asarray([100,100])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    index = np.random.randint(0, shape.size(), [1, ]).item()
    cArray.setArray(data)
    if np.array_equal(NumC.delete(cArray, index, NumC.Axis.NONE).flatten(), np.delete(data, index, axis=None)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing deleteIndices Slice: Axis = Row', 'cyan'))
    shapeInput = np.asarray([100,100])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    index = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    if np.array_equal(NumC.delete(cArray, index, NumC.Axis.ROW), np.delete(data, index, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing deleteIndices Slice: Axis = Col', 'cyan'))
    shapeInput = np.asarray([100,100])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    index = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    if np.array_equal(NumC.delete(cArray, index, NumC.Axis.COL), np.delete(data, index, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    return

    print(colored('Testing diagflat array', 'cyan'))
    numElements = np.random.randint(1, 25, [1, ]).item()
    shape = NumC.Shape(1, numElements)
    elements = np.random.randint(1, 100, [numElements,])
    cElements = NumC.NdArray(shape)
    cElements.setArray(elements)
    if np.array_equal(NumC.diagflat(cElements), np.diagflat(elements)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing diagonal: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    offset = np.random.randint(0, min(shape.rows, shape.cols), [1, ]).item()
    if np.array_equal(NumC.diagonal(cArray, offset, NumC.Axis.ROW).astype(np.uint32).flatten(),
                      np.diagonal(data, offset, axis1=1, axis2=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing diagonal: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    offset = np.random.randint(0, min(shape.rows, shape.cols), [1, ]).item()
    if np.array_equal(NumC.diagonal(cArray, offset, NumC.Axis.COL).astype(np.uint32).flatten(),
                      np.diagonal(data, offset, axis1=0, axis2=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing diff: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.diff(cArray, NumC.Axis.NONE).flatten().astype(np.uint32), np.diff(data.flatten())):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing diff: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.diff(cArray, NumC.Axis.ROW).astype(np.uint32), np.diff(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing diff: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.diff(cArray, NumC.Axis.COL).astype(np.uint32), np.diff(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(data)
        print(NumC.diff(cArray, NumC.Axis.COL).astype(np.uint32))
        print(colored('\tFAIL', 'red'))

    print(colored('Testing divide', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(1, 50, [shape.rows, shape.cols]).astype(np.double)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(np.round(NumC.divide(cArray1, cArray2), 10), np.round(np.divide(data1, data2), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing dot vector', 'cyan'))
    size = np.random.randint(1, 100, [1,]).item()
    shape = NumC.Shape(1, size)
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    data2 = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if NumC.dot(cArray1, cArray2).astype(np.uint32).item() == np.dot(data1, data2.T).item():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing dot array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    shape1 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumC.Shape(shapeInput[1].item(), np.random.randint(1, 100, [1,]).item())
    cArray1 = NumC.NdArray(shape1)
    cArray2 = NumC.NdArray(shape2)
    data1 = np.random.randint(1, 50, [shape1.rows, shape1.cols], dtype=np.uint32)
    data2 = np.random.randint(1, 50, [shape2.rows, shape2.cols], dtype=np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.dot(cArray1, cArray2).astype(np.uint32), np.dot(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing dump', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    tempDir = r'C:\Temp'
    if not os.path.exists(tempDir):
        os.mkdir(tempDir)
    tempFile = os.path.join(tempDir, 'NdArrayDump.bin')
    NumC.dump(cArray, tempFile)
    if os.path.exists(tempFile):
        filesize = os.path.getsize(tempFile)
        if filesize == data.size * 8:
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))
    else:
        print(colored('\tFAIL', 'red'))
    os.remove(tempFile)

    print(colored('Testing empty rectangle', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    cArray = NumC.empty(shapeInput[0].item(), shapeInput[1].item())
    if (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing empty Shape', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.empty(shape)
    if (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing empty_like', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.empty_like(cArray1)
    if (cArray2.shape().rows == shape.rows and cArray2.shape().cols == shape.cols and
            cArray2.size() == shapeInput.prod()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing endianess', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    if NumC.endianess(cArray) == NumC.Endian.NATIVE:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing equal', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(0, 10, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 10, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.equal(cArray1, cArray2), np.equal(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing exp scalar', 'cyan'))
    value = np.abs(np.random.rand(1).item())
    if np.round(NumC.exp(value), 10) == np.round(np.exp(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing exp array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.exp(cArray), 10), np.round(np.exp(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing exp2 scalar', 'cyan'))
    value = np.abs(np.random.rand(1).item())
    if np.round(NumC.exp2(value), 10) == np.round(np.exp2(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing exp2 array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.exp2(cArray), 10), np.round(np.exp2(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing expm1 scalar', 'cyan'))
    value = np.abs(np.random.rand(1).item())
    if np.round(NumC.expm1(value), 10) == np.round(np.expm1(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing expm1 array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.expm1(cArray), 10), np.round(np.expm1(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing eye 1D', 'cyan'))
    shapeInput = np.random.randint(1, 100, [1, ]).item()
    randK = np.random.randint(0, shapeInput, [1, ]).item()
    if np.array_equal(NumC.eye(shapeInput, randK), np.eye(shapeInput, k=randK)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing eye 2D', 'cyan'))
    shapeInput = np.random.randint(10, 100, [2, ])
    randK = np.random.randint(0, shapeInput.min(), [1, ]).item()
    if np.array_equal(NumC.eye(shapeInput[0].item(), shapeInput[1].item(), randK), np.eye(shapeInput[0].item(), shapeInput[1].item(), k=randK)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing eye Shape', 'cyan'))
    shapeInput = np.random.randint(10, 100, [2, ])
    cShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    randK = np.random.randint(0, shapeInput.min(), [1, ]).item()
    if np.array_equal(NumC.eye(cShape, randK), np.eye(shapeInput[0].item(), shapeInput[1].item(), k=randK)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing fix scalar', 'cyan'))
    value = np.random.randn(1).item() * 100
    if NumC.fix(value) == np.fix(value):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing fix array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    if np.array_equal(NumC.fix(cArray), np.fix(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing flatten', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.flatten(cArray).getNumpyArray(), np.resize(data, [1, data.size])):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing flatnonzero', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.flatnonzero(cArray).getNumpyArray().flatten(), np.flatnonzero(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing flip', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.flip(cArray, NumC.Axis.NONE).getNumpyArray(), np.flip(data.reshape(1, data.size), axis=1).reshape(shapeInput)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing fliplr', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.fliplr(cArray).getNumpyArray(), np.fliplr(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing flipud', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.flipud(cArray).getNumpyArray(), np.flipud(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing floor scalar', 'cyan'))
    value = np.random.randn(1).item() * 100
    if NumC.floor(value) == np.floor(value):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing floor array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    if np.array_equal(NumC.floor(cArray), np.floor(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing floor_divide scalar', 'cyan'))
    value1 = np.random.randn(1).item() * 100 + 1000
    value2 = np.random.randn(1).item() * 100 + 1000
    if NumC.floor_divide(value1, value2) == np.floor_divide(value1, value2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing floor_divide array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data2 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.floor_divide(cArray1, cArray2), np.floor_divide(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing fmax scalar', 'cyan'))
    value1 = np.random.randn(1).item() * 100 + 1000
    value2 = np.random.randn(1).item() * 100 + 1000
    if NumC.fmax(value1, value2) == np.fmax(value1, value2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing fmax array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data2 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.fmax(cArray1, cArray2), np.fmax(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing fmin scalar', 'cyan'))
    value1 = np.random.randn(1).item() * 100 + 1000
    value2 = np.random.randn(1).item() * 100 + 1000
    if NumC.fmin(value1, value2) == np.fmin(value1, value2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing fmin array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data2 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.fmin(cArray1, cArray2), np.fmin(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing fmod scalar', 'cyan'))
    value1 = np.random.randint(1, 100, [1, ]).item() * 100 + 1000
    value2 = np.random.randint(1, 100, [1, ]).item() * 100 + 1000
    if NumC.fmod(value1, value2) == np.fmod(value1, value2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing fmod array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArrayInt(shape)
    cArray2 = NumC.NdArrayInt(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]) * 100 + 1000
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]) * 100 + 1000
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.fmod(cArray1, cArray2), np.fmod(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing full square', 'cyan'))
    shapeInput = np.random.randint(1, 100, [1,]).item()
    value = np.random.randint(1, 100, [1, ]).item()
    cArray = NumC.full(shapeInput, value)
    if (cArray.shape[0] == shapeInput and cArray.shape[1] == shapeInput and
            cArray.size == shapeInput**2 and np.all(cArray == value)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing full rectangle', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    value = np.random.randint(1, 100, [1, ]).item()
    cArray = NumC.full(shapeInput[0].item(), shapeInput[1].item(), value)
    if (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod() and np.all(cArray == value)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing full Shape', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    value = np.random.randint(1, 100, [1, ]).item()
    cArray = NumC.full(shape, value)
    if (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod() and np.all(cArray == value)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing full_like', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    value = np.random.randint(1, 100, [1, ]).item()
    cArray2 = NumC.full_like(cArray1, value)
    if (cArray2.shape().rows == shape.rows and cArray2.shape().cols == shape.cols and
            cArray2.size() == shapeInput.prod() and np.all(cArray2.getNumpyArray() == value)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing greater', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.greater(cArray1, cArray2).getNumpyArray(), np.greater(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing greater_equal array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.greater_equal(cArray1, cArray2).getNumpyArray(), np.greater_equal(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing hstack', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape1 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1,]).item())
    shape3 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1,]).item())
    shape4 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item() + np.random.randint(1, 10, [1,]).item())
    cArray1 = NumC.NdArray(shape1)
    cArray2 = NumC.NdArray(shape2)
    cArray3 = NumC.NdArray(shape3)
    cArray4 = NumC.NdArray(shape4)
    data1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    data4 = np.random.randint(1, 100, [shape4.rows, shape4.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    if np.array_equal(NumC.hstack(cArray1, cArray2, cArray3, cArray4),
                      np.hstack([data1, data2, data3, data4])):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing hypot scalar', 'cyan'))
    value1 = np.random.randn(1).item() * 100 + 1000
    value2 = np.random.randn(1).item() * 100 + 1000
    if NumC.hypot(value1, value2) == np.hypot(value1, value2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing hypot array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    data2 = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.hypot(cArray1, cArray2), np.hypot(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing identity', 'cyan'))
    squareSize = np.random.randint(10, 100, [1, ]).item()
    if np.array_equal(NumC.identity(squareSize).getNumpyArray(), np.identity(squareSize)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing intersect1d', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArrayInt(shape)
    cArray2 = NumC.NdArrayInt(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.intersect1d(cArray1, cArray2).getNumpyArray().flatten(), np.intersect1d(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing invert', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArrayInt(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.invert(cArray).getNumpyArray(), np.invert(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing isclose', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.rand(shape.rows, shape.cols)
    data2 = data1 + np.random.randn(shape.rows, shape.cols) * 1e-5
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    rtol = 1e-5
    atol = 1e-8
    if np.array_equal(NumC.isclose(cArray1, cArray2, rtol, atol).getNumpyArray(), np.isclose(data1, data2, rtol=rtol, atol=atol)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing isnan scalar', 'cyan'))
    value = np.random.randn(1).item() * 100 + 1000
    if NumC.isnan(value) == np.isnan(value):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing isnan array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray.setArray(data)
    if np.array_equal(NumC.isnan(cArray), np.isnan(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing ldexp scalar', 'cyan'))
    value1 = np.random.randn(1).item() * 100
    value2 = np.random.randint(1, 20, [1,]).item()
    if np.round(NumC.ldexp(value1, value2), 10) == np.round(np.ldexp(value1, value2), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing ldexp array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArrayInt8(shape)
    data1 = np.random.randn(shape.rows, shape.cols) * 100
    data2 = np.random.randint(1, 20, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(np.round(NumC.ldexp(cArray1, cArray2), 10), np.round(np.ldexp(data1, data2), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing left_shift', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    bitsToShift = np.random.randint(1, 32, [1, ]).item()
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArrayInt(shape)
    data = np.random.randint(1, np.iinfo(np.uint32).max, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.left_shift(cArray, bitsToShift).getNumpyArray(), np.left_shift(data, bitsToShift)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing less', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.less(cArray1, cArray2).getNumpyArray(), np.less(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing less_equal array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.less_equal(cArray1, cArray2).getNumpyArray(), np.less_equal(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing linspace: include endPoint True', 'cyan'))
    start = np.random.randint(1, 10, [1, ]).item()
    end = np.random.randint(start + 10, 100, [1, ]).item()
    numPoints = np.random.randint(1, 100, [1, ]).item()
    if np.array_equal(np.round(NumC.linspace(start, end, numPoints, True).getNumpyArray().flatten(), 10),
                      np.round(np.linspace(start, end, numPoints, endpoint=True), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing linspace: include endPoint False', 'cyan'))
    start = np.random.randint(1, 10, [1, ]).item()
    end = np.random.randint(start + 10, 100, [1, ]).item()
    numPoints = np.random.randint(1, 100, [1, ]).item()
    if np.array_equal(np.round(NumC.linspace(start, end, numPoints, False).getNumpyArray().flatten(), 10),
                      np.round(np.linspace(start, end, numPoints, endpoint=False), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing log scalar', 'cyan'))
    value = np.random.randn(1).item() * 100 + 1000
    if np.round(NumC.log(value), 10) == np.round(np.log(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing log array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.log(cArray), 10), np.round(np.log(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing log10 scalar', 'cyan'))
    value = np.random.randn(1).item() * 100 + 1000
    if np.round(NumC.log10(value), 10) == np.round(np.log10(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing log10 array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.log10(cArray), 10), np.round(np.log10(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing log1p scalar', 'cyan'))
    value = np.random.randn(1).item() * 100 + 1000
    if np.round(NumC.log1p(value), 10) == np.round(np.log1p(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing log1p array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.log1p(cArray), 10), np.round(np.log1p(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing log2 scalar', 'cyan'))
    value = np.random.randn(1).item() * 100 + 1000
    if np.round(NumC.log2(value), 10) == np.round(np.log2(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing log2 array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100 + 1000
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.log2(cArray), 10), np.round(np.log2(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing logical_and', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(0, 20, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 20, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.logical_and(cArray1, cArray2).getNumpyArray(), np.logical_and(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing logical_not', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 20, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.logical_not(cArray).getNumpyArray(), np.logical_not(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing logical_or', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(0, 20, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 20, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.logical_or(cArray1, cArray2).getNumpyArray(), np.logical_or(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing logical_xor', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(0, 20, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 20, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.logical_xor(cArray1, cArray2).getNumpyArray(), np.logical_xor(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing matmult', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape1 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumC.Shape(shapeInput[1].item(), shapeInput[0].item())
    cArray1 = NumC.NdArray(shape1)
    cArray2 = NumC.NdArray(shape2)
    data1 = np.random.randint(0, 20, [shape1.rows, shape1.cols])
    data2 = np.random.randint(0, 20, [shape2.rows, shape2.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.matmul(cArray1, cArray2).getNumpyArray(), np.matmul(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing max: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if NumC.max(cArray, NumC.Axis.NONE).item() == np.max(data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing max: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.max(cArray, NumC.Axis.ROW).getNumpyArray().astype(np.uint32).flatten(), np.max(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing max: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.max(cArray, NumC.Axis.COL).getNumpyArray().astype(np.uint32).flatten(), np.max(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing maximum', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.maximum(cArray1, cArray2).getNumpyArray(), np.maximum(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing mean', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.mean(cArray, NumC.Axis.COL).getNumpyArray().flatten(), np.mean(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing median', 'cyan'))
    isEven = True
    while isEven:
        shapeInput = np.random.randint(1, 100, [2, ])
        isEven = shapeInput.prod().item() % 2 == 0
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.median(cArray, NumC.Axis.COL).getNumpyArray().flatten(), np.median(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing min: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if NumC.min(cArray, NumC.Axis.NONE).item() == np.min(data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing min: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.min(cArray, NumC.Axis.ROW).getNumpyArray().astype(np.uint32).flatten(), np.min(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing min: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.min(cArray, NumC.Axis.COL).getNumpyArray().astype(np.uint32).flatten(), np.min(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing minimum', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.minimum(cArray1, cArray2).getNumpyArray(), np.minimum(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing mod', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArrayInt(shape)
    cArray2 = NumC.NdArrayInt(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.mod(cArray1, cArray2).getNumpyArray(), np.mod(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing multiply', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.multiply(cArray1, cArray2).getNumpyArray(), np.multiply(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing nbytes', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if NumC.nbytes(cArray) == data.size * 8:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing newbyteorder scalar', 'cyan'))
    value = np.random.randint(1, 100, [1,]).item()
    if NumC.newbyteorder(value, NumC.Endian.BIG) == np.asarray([value]).newbyteorder().item():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing newbyteorder array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArrayInt(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.newbyteorder(cArray, NumC.Endian.BIG), data.newbyteorder()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing negative array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(-100, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.negative(cArray).getNumpyArray(), 10), np.round(np.negative(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing nonzero', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.nonzero(cArray).getNumpyArray().flatten(), data.flatten().nonzero()[0]):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing norm: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if NumC.norm(cArray, NumC.Axis.NONE).flatten() == np.linalg.norm(data.flatten()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing norm: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    norms = NumC.norm(cArray, NumC.Axis.ROW).getNumpyArray().flatten()
    allPass = True
    for idx, row in enumerate(data.transpose()):
        if norms[idx] != np.linalg.norm(row):
            allPass = False
            break
    if allPass:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing norm: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    norms = NumC.norm(cArray, NumC.Axis.COL).getNumpyArray().flatten()
    allPass = True
    for idx, row in enumerate(data):
        if norms[idx] != np.linalg.norm(row):
            allPass = False
            break
    if allPass:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing not_equal', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.not_equal(cArray1, cArray2).getNumpyArray(), np.not_equal(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing ones square', 'cyan'))
    shapeInput = np.random.randint(1, 100, [1, ]).item()
    cArray = NumC.ones(shapeInput)
    if (cArray.shape[0] == shapeInput and cArray.shape[1] == shapeInput and
            cArray.size == shapeInput ** 2 and np.all(cArray == 1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing ones rectangle', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    cArray = NumC.ones(shapeInput[0].item(), shapeInput[1].item())
    if (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod() and np.all(cArray == 1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing ones Shape', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.ones(shape)
    if (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod() and np.all(cArray == 1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing ones_like', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.ones_like(cArray1)
    if (cArray2.shape().rows == shape.rows and cArray2.shape().cols == shape.cols and
            cArray2.size() == shapeInput.prod() and np.all(cArray2.getNumpyArray() == 1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing pad', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    padWidth = np.random.randint(1, 10, [1, ]).item()
    padValue = np.random.randint(1, 100, [1, ]).item()
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray = NumC.NdArray(shape)
    cArray.setArray(data)
    if np.array_equal(NumC.pad(cArray, padWidth, padValue).getNumpyArray(), np.pad(data, padWidth, mode='constant', constant_values=padValue)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing partition: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    kthElement = np.random.randint(0, shapeInput.prod(), [1,], dtype=np.uint32).item()
    partitionedArray = NumC.partition(cArray, kthElement, NumC.Axis.NONE).getNumpyArray().flatten()
    if (np.all(partitionedArray[:kthElement] <= partitionedArray[kthElement]) and
        np.all(partitionedArray[kthElement:] >= partitionedArray[kthElement])):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing partition: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    kthElement = np.random.randint(0, shapeInput[0], [1,], dtype=np.uint32).item()
    partitionedArray = NumC.partition(cArray, kthElement, NumC.Axis.ROW).getNumpyArray().transpose()
    allPass = True
    for row in partitionedArray:
        if not (np.all(row[:kthElement] <= row[kthElement]) and
                np.all(row[kthElement:] >= row[kthElement])):
            allPass = False
            break
    if allPass:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing partition: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    kthElement = np.random.randint(0, shapeInput[1], [1,], dtype=np.uint32).item()
    partitionedArray = NumC.partition(cArray, kthElement, NumC.Axis.COL).getNumpyArray()
    allPass = True
    for row in partitionedArray:
        if not (np.all(row[:kthElement] <= row[kthElement]) and
                np.all(row[kthElement:] >= row[kthElement])):
            allPass = False
            break
    if allPass:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing power array scalar', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    exponent = np.random.randint(0, 5, [1, ]).item()
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.power(cArray, exponent), 10), np.round(np.power(data, exponent), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing power array array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    cExponents = NumC.NdArrayInt8(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    exponents = np.random.randint(0, 5, [shape.rows, shape.cols]).astype(np.uint8)
    cArray.setArray(data)
    cExponents.setArray(exponents)
    if np.array_equal(np.round(NumC.power(cArray, cExponents), 10), np.round(np.power(data, exponents), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing prod: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32).astype(np.double)
    cArray.setArray(data)
    if NumC.prod(cArray, NumC.Axis.NONE).item() == data.prod():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing prod: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(NumC.prod(cArray, NumC.Axis.ROW).getNumpyArray().flatten(), data.prod(axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing prod: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(NumC.prod(cArray, NumC.Axis.COL).getNumpyArray().flatten(), data.prod(axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing ptp: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if NumC.ptp(cArray, NumC.Axis.NONE).getNumpyArray().astype(np.uint32).item() == data.ptp():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing ptp: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.ptp(cArray, NumC.Axis.ROW).getNumpyArray().flatten().astype(np.uint32), data.ptp(axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing ptp: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.ptp(cArray, NumC.Axis.COL).getNumpyArray().flatten().astype(np.uint32), data.ptp(axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing put', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    numIndices = np.random.randint(0, shape.size())
    indices = np.asarray(range(numIndices))
    values = np.random.randint(1, 500, [numIndices, ])
    cIndices = NumC.NdArrayInt(1, numIndices)
    cValues = NumC.NdArray(1, numIndices)
    cIndices.setArray(indices)
    cValues.setArray(values)
    NumC.put(cArray, cIndices, cValues)
    data.put(indices, values)
    if np.array_equal(cArray.getNumpyArray(), data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing rad2deg scalar', 'cyan'))
    value = np.abs(np.random.rand(1).item()) * 2 * np.pi
    if np.round(NumC.rad2deg(value), 10) == np.round(np.rad2deg(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing rad2deg array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 2 * np.pi
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.rad2deg(cArray), 9), np.round(np.rad2deg(data), 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing reciprocal array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.reciprocal(cArray).getNumpyArray(), 10), np.round(np.reciprocal(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    # numpy and cmath remainders are calculated differently, so convert for testing purposes
    print(colored('Testing remainder scalar', 'cyan'))
    values = np.random.rand(2) * 100
    values = np.sort(values)
    res = NumC.remainder(values[1].item(), values[0].item())
    if res < 0:
        res += values[0].item()
    if np.round(res, 10) == np.round(np.remainder(values[1], values[0]), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    # numpy and cmath remainders are calculated differently, so convert for testing purposes
    print(colored('Testing remainder array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape)
    cArray2 = NumC.NdArray(shape)
    data1 = np.random.rand(shape.rows, shape.cols) * 100 + 10
    data2 = data1 - np.random.rand(shape.rows, shape.cols) * 10
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    res = NumC.remainder(cArray1, cArray2)
    res[res < 0] = res[res < 0] + data2[res < 0]
    if np.array_equal(np.round(res, 10), np.round(np.remainder(data1, data2), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing reshape', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    newShape = NumC.Shape(shapeInput[1].item(), shapeInput[0].item())
    NumC.reshape(cArray, newShape)
    if np.array_equal(cArray.getNumpyArray(), data.reshape(shapeInput[::-1])):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing resizeFast', 'cyan'))
    shapeInput1 = np.random.randint(1, 100, [2, ])
    shapeInput2 = np.random.randint(1, 100, [2, ])
    shape1 = NumC.Shape(shapeInput1[0].item(), shapeInput1[1].item())
    shape2 = NumC.Shape(shapeInput2[0].item(), shapeInput2[1].item())
    cArray = NumC.NdArray(shape1)
    data = np.random.randint(1, 100, [shape1.rows, shape1.cols], dtype=np.uint32)
    cArray.setArray(data)
    NumC.resizeFast(cArray, shape2)
    if np.all(cArray.getNumpyArray() == 0) and cArray.shape().rows == shape2.rows and cArray.shape().cols == shape2.cols:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing resizeSlow', 'cyan'))
    shapeInput1 = np.random.randint(1, 100, [2, ])
    shapeInput2 = np.random.randint(1, 100, [2, ])
    shape1 = NumC.Shape(shapeInput1[0].item(), shapeInput1[1].item())
    shape2 = NumC.Shape(shapeInput2[0].item(), shapeInput2[1].item())
    cArray = NumC.NdArray(shape1)
    data = np.random.randint(1, 100, [shape1.rows, shape1.cols], dtype=np.uint32)
    cArray.setArray(data)
    NumC.resizeSlow(cArray, shape2)
    if cArray.shape().rows == shape2.rows and cArray.shape().cols == shape2.cols and not np.all(cArray.getNumpyArray() == 0):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing right_shift', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    bitsToShift = np.random.randint(1, 32, [1, ]).item()
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArrayInt(shape)
    data = np.random.randint(1, np.iinfo(np.uint32).max, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(NumC.right_shift(cArray, bitsToShift).getNumpyArray(), np.right_shift(data, bitsToShift)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing rint scalar', 'cyan'))
    value = np.abs(np.random.rand(1).item()) * 2 * np.pi
    if NumC.rint(value) == np.rint(value):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing rint array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 2 * np.pi
    cArray.setArray(data)
    if np.array_equal(NumC.rint(cArray), np.rint(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing round scalar', 'cyan'))
    value = np.abs(np.random.rand(1).item()) * 2 * np.pi
    if NumC.round(value, 10) == np.round(value, 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing round array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 2 * np.pi
    cArray.setArray(data)
    if np.array_equal(NumC.round(cArray, 10), np.round(data, 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing row_stack', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape1 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumC.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1,]).item(), shapeInput[1].item())
    shape3 = NumC.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1,]).item(), shapeInput[1].item())
    shape4 = NumC.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1,]).item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape1)
    cArray2 = NumC.NdArray(shape2)
    cArray3 = NumC.NdArray(shape3)
    cArray4 = NumC.NdArray(shape4)
    data1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    data4 = np.random.randint(1, 100, [shape4.rows, shape4.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    if np.array_equal(NumC.row_stack(cArray1, cArray2, cArray3, cArray4),
                      np.row_stack([data1, data2, data3, data4])):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing hypot scalar', 'cyan'))
    value1 = np.random.randn(1).item() * 100 + 1000
    value2 = np.random.randn(1).item() * 100 + 1000
    if NumC.hypot(value1, value2) == np.hypot(value1, value2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing setdiff1d', 'cyan'))
    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArrayInt(shape)
    cArray2 = NumC.NdArrayInt(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.setdiff1d(cArray1, cArray2).getNumpyArray().flatten(), np.setdiff1d(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing shape array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    if cArray.shape().rows == shape.rows and cArray.shape().cols == shape.cols:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sign scalar', 'cyan'))
    value = np.random.randn(1).item() * 100
    if NumC.sign(value) == np.sign(value):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sign array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    if np.array_equal(NumC.sign(cArray), np.sign(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing signbit scalar', 'cyan'))
    value = np.random.randn(1).item() * 100
    if NumC.signbit(value) == np.signbit(value):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing signbit array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols) * 100
    cArray.setArray(data)
    if np.array_equal(NumC.signbit(cArray), np.signbit(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sin scalar', 'cyan'))
    value = np.random.randn(1).item()
    if np.round(NumC.sin(value), 10) == np.round(np.sin(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sin array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.sin(cArray), 10), np.round(np.sin(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sinc scalar', 'cyan'))
    value = np.random.randn(1).item()
    if np.round(NumC.sinc(value), 10) == np.round(np.sinc(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sinc array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.sinc(cArray), 10), np.round(np.sinc(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sinh scalar', 'cyan'))
    value = np.random.randn(1).item()
    if np.round(NumC.sinh(value), 10) == np.round(np.sinh(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sinh array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.sinh(cArray), 10), np.round(np.sinh(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing size', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    if cArray.size() == shapeInput.prod().item():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sort: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    d = data.flatten()
    d.sort()
    if np.array_equal(NumC.sort(cArray, NumC.Axis.NONE).getNumpyArray().flatten(), d):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sort: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    pSorted = np.sort(data, axis=0)
    cSorted = NumC.sort(cArray, NumC.Axis.ROW).getNumpyArray().astype(np.uint32)
    if np.array_equal(cSorted, pSorted):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sort: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    pSorted = np.sort(data, axis=1)
    cSorted = NumC.sort(cArray, NumC.Axis.COL).getNumpyArray().astype(np.uint32)
    if np.array_equal(cSorted, pSorted):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sqrt scalar', 'cyan'))
    value = np.random.randint(1, 100, [1,]).item()
    if np.round(NumC.sqrt(value), 10) == np.round(np.sqrt(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sqrt array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.sqrt(cArray), 10), np.round(np.sqrt(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing square scalar', 'cyan'))
    value = np.random.randint(1, 100, [1, ]).item()
    if np.round(NumC.square(value), 10) == np.round(np.square(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing square array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.square(cArray), 10), np.round(np.square(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing std: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.round(NumC.std(cArray, NumC.Axis.NONE).item(), 10) == np.round(np.std(data), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing std: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.std(cArray, NumC.Axis.ROW).getNumpyArray().flatten(), 10), np.round(np.std(data, axis=0), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing std: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.std(cArray, NumC.Axis.COL).getNumpyArray().flatten(), 10), np.round(np.std(data, axis=1), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sum: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if NumC.sum(cArray, NumC.Axis.NONE).item() == np.sum(data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sum: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(NumC.sum(cArray, NumC.Axis.ROW).getNumpyArray().flatten(), np.sum(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sum: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(NumC.sum(cArray, NumC.Axis.COL).getNumpyArray().flatten(), np.sum(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing swapaxes', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(NumC.swapaxes(cArray).getNumpyArray(), data.T):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing tan scalar', 'cyan'))
    value = np.random.rand(1).item() * np.pi
    if np.round(NumC.tan(value), 10) == np.round(np.tan(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing tan array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.tan(cArray), 10), np.round(np.tan(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing tanh scalar', 'cyan'))
    value = np.random.rand(1).item() * np.pi
    if np.round(NumC.tanh(value), 10) == np.round(np.tanh(value), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing tanh array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.tanh(cArray), 10), np.round(np.tanh(data), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing tile rectangle', 'cyan'))
    shapeInput = np.random.randint(1, 10, [2, ])
    shapeRepeat = np.random.randint(1, 10, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    shapeR = NumC.Shape(shapeRepeat[0].item(), shapeRepeat[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.tile(cArray, shapeR.rows, shapeR.cols), np.tile(data, shapeRepeat)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing tile shape', 'cyan'))
    shapeInput = np.random.randint(1, 10, [2, ])
    shapeRepeat = np.random.randint(1, 10, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    shapeR = NumC.Shape(shapeRepeat[0].item(), shapeRepeat[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumC.tile(cArray, shapeR), np.tile(data, shapeRepeat)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing tofile bin', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    filename = r'C:\Temp\temp'
    NumC.tofile(cArray, filename, '')
    if os.path.exists(filename + '.bin'):
        print(colored('\tPASS', 'green'))
        os.remove(filename + '.bin')
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing tofile txt', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    filename = r'C:\Temp\temp'
    NumC.tofile(cArray, filename, '\n')
    if os.path.exists(filename + '.txt'):
        print(colored('\tPASS', 'green'))
        os.remove(filename + '.txt')
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing toStlVector', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    out = np.asarray(NumC.toStlVector(cArray))
    if np.array_equal(out, data.flatten()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing trace: Offset=Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows, [1, ]).item()
    if np.array_equal(NumC.trace(cArray, offset, NumC.Axis.ROW), data.trace(offset, axis1=1, axis2=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing trace: Offset=Col', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows, [1, ]).item()
    if np.array_equal(NumC.trace(cArray, offset, NumC.Axis.COL), data.trace(offset, axis1=0, axis2=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing transpose', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(NumC.transpose(cArray).getNumpyArray(), np.transpose(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing tri: square', 'cyan'))
    squareSize = np.random.randint(10, 100, [1, ]).item()
    offset = np.random.randint(0, squareSize, [1, ]).item()
    if np.array_equal(NumC.tri(squareSize, offset), np.tri(squareSize, k=offset)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing tri: rectangle', 'cyan'))
    shapeInput = np.random.randint(10, 100, [2, ])
    offset = np.random.randint(0, squareSize, [1, ]).item()
    if np.array_equal(NumC.tri(shapeInput[0].item(), shapeInput[1].item(), offset),
                      np.tri(shapeInput[0].item(), shapeInput[1].item(), k=offset)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing trim_zeros: "f"', 'cyan'))
    numElements = np.random.randint(50, 100, [1, ]).item()
    offsetBeg = np.random.randint(0, 10, [1, ]).item()
    offsetEnd = np.random.randint(10, numElements, [1, ]).item()
    shape = NumC.Shape(1, numElements)
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    data[0, :offsetBeg] = 0
    data[0, -offsetEnd:] = 0
    cArray.setArray(data)
    if np.array_equal(NumC.trim_zeros(cArray, 'f').getNumpyArray().flatten(),
                      np.trim_zeros(data.flatten(), 'f')):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing trim_zeros: "b"', 'cyan'))
    numElements = np.random.randint(50, 100, [1, ]).item()
    offsetBeg = np.random.randint(0, 10, [1, ]).item()
    offsetEnd = np.random.randint(10, numElements, [1, ]).item()
    shape = NumC.Shape(1, numElements)
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    data[0, :offsetBeg] = 0
    data[0, -offsetEnd:] = 0
    cArray.setArray(data)
    if np.array_equal(NumC.trim_zeros(cArray, 'b').getNumpyArray().flatten(),
                      np.trim_zeros(data.flatten(), 'b')):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing trim_zeros: "fb"', 'cyan'))
    numElements = np.random.randint(50, 100, [1, ]).item()
    offsetBeg = np.random.randint(0, 10, [1, ]).item()
    offsetEnd = np.random.randint(10, numElements, [1, ]).item()
    shape = NumC.Shape(1, numElements)
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    data[0, :offsetBeg] = 0
    data[0, -offsetEnd:] = 0
    cArray.setArray(data)
    if np.array_equal(NumC.trim_zeros(cArray, 'fb').getNumpyArray().flatten(),
                      np.trim_zeros(data.flatten(), 'fb')):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing trunc scalar', 'cyan'))
    value = np.random.rand(1).item() * np.pi
    if NumC.trunc(value) == np.trunc(value):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing trunc array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(NumC.trunc(cArray), np.trunc(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing union1d', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumC.NdArrayInt(shape)
    cArray2 = NumC.NdArrayInt(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(NumC.union1d(cArray1, cArray2).getNumpyArray().flatten(), np.union1d(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing unique array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(NumC.unique(cArray).getNumpyArray().flatten(), np.unique(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing unwrap scalar', 'cyan'))
    value = np.random.randn(1).item() * 3 * np.pi
    if value < 0:
        pValue = value + 2 * np.pi
    elif value >= 2 * np.pi:
        pValue = value - 2 * np.pi
    else:
        pValue = value
    if np.round(NumC.unwrap(value), 10) == np.round(pValue, 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing unwrap array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    data[data < 0] = data[data < 0] + 2 * np.pi
    data[data > 2 * np.pi] = data[data > 2 * np.pi] - 2 * np.pi
    if np.array_equal(np.round(NumC.unwrap(cArray), 10), np.round(data, 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing var: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.round(NumC.var(cArray, NumC.Axis.NONE).item(), 9) == np.round(np.var(data), 9):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing var: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.var(cArray, NumC.Axis.ROW).getNumpyArray().flatten(), 9), np.round(np.var(data, axis=0), 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing var: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.var(cArray, NumC.Axis.COL).getNumpyArray().flatten(), 9), np.round(np.var(data, axis=1), 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing vstack', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape1 = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumC.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1,]).item(), shapeInput[1].item())
    shape3 = NumC.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1,]).item(), shapeInput[1].item())
    shape4 = NumC.Shape(shapeInput[0].item() + np.random.randint(1, 10, [1,]).item(), shapeInput[1].item())
    cArray1 = NumC.NdArray(shape1)
    cArray2 = NumC.NdArray(shape2)
    cArray3 = NumC.NdArray(shape3)
    cArray4 = NumC.NdArray(shape4)
    data1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    data4 = np.random.randint(1, 100, [shape4.rows, shape4.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    if np.array_equal(NumC.vstack(cArray1, cArray2, cArray3, cArray4),
                      np.vstack([data1, data2, data3, data4])):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing zeros square', 'cyan'))
    shapeInput = np.random.randint(1, 100, [1, ]).item()
    cArray = NumC.zeros(shapeInput)
    if (cArray.shape[0] == shapeInput and cArray.shape[1] == shapeInput and
            cArray.size == shapeInput ** 2 and np.all(cArray == 0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing zeros rectangle', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    cArray = NumC.zeros(shapeInput[0].item(), shapeInput[1].item())
    if (cArray.shape[0] == shapeInput[0] and cArray.shape[1] == shapeInput[1] and
            cArray.size == shapeInput.prod() and np.all(cArray == 0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing zeros Shape', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.zeros(shape)
    if (cArray.shape[0] == shape.rows and cArray.shape[1] == shape.cols and
            cArray.size == shapeInput.prod() and np.all(cArray == 0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

####################################################################################
if __name__ == '__main__':
    doTest()