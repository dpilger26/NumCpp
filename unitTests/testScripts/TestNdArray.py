import numpy as np
from termcolor import colored
import os
import getpass
import sys
if sys.platform == 'linux':
    sys.path.append(r'../lib')
else:
    sys.path.append(r'../build/x64/Release')
import NumCpp

####################################################################################
def doTest():
    print(colored('Testing NdArray Class', 'magenta'))

    print(colored('Testing Default Constructor', 'cyan'))
    cArray = NumCpp.NdArray()
    if cArray.shape().rows == 0 and cArray.shape().cols == 0 and cArray.size() == 0:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Square Constructor', 'cyan'))
    numRowsCols = np.random.randint(1, 100, [1,]).item()
    cArray = NumCpp.NdArray(numRowsCols)
    if (cArray.shape().rows == numRowsCols and cArray.shape().cols == numRowsCols and
            cArray.size() == numRowsCols**2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Rectangle Constructor', 'cyan'))
    numRowsCols = np.random.randint(1, 100, [2,])
    cArray = NumCpp.NdArray(numRowsCols[0].item(), numRowsCols[1].item())
    if (cArray.shape().rows == numRowsCols[0] and cArray.shape().cols == numRowsCols[1] and
            cArray.size() == numRowsCols.prod()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Shape Constructor', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    if (cArray.shape().rows == shape.rows and cArray.shape().cols == shape.cols and
            cArray.size() == shape.rows * shape.cols):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing 1D List Constructor', 'cyan'))
    if NumCpp.NdArray.test1DListContructor():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing 2D List Constructor', 'cyan'))
    if NumCpp.NdArray.test2DListContructor():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Copy Constructor', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(np.random.randint(1, 100, [shape.rows, shape.cols]))
    cArray2 = NumCpp.NdArray(cArray)
    if np.array_equal(np.round(cArray2.getNumpyArray(), 10), np.round(cArray.getNumpyArray(), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing at flat', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    randomIdx = np.random.randint(0, shapeInput.prod(), [1,]).item()
    if cArray.get(randomIdx) == data.flatten()[randomIdx]:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing at row/col', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    randomRowIdx = np.random.randint(0, shapeInput[0], [1,]).item()
    randomColIdx = np.random.randint(0, shapeInput[1], [1,]).item()
    if cArray.get(randomRowIdx, randomColIdx) == data[randomRowIdx, randomColIdx]:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing at slice 1D', 'cyan'))
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    start = np.random.randint(0, shapeInput.prod() // 10, [1,]).item()
    stop = np.random.randint(start + 1, shapeInput.prod(), [1,]).item()
    step = np.random.randint(1, shapeInput.prod() // 10, [1,]).item()
    if np.array_equal(cArray.get(NumCpp.Slice(start, stop, step)).flatten().astype(np.uint32), data.flatten()[start:stop:step]):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing at slice 2D', 'cyan'))
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1,]).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1,]).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1,]).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1,]).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1,]).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1,]).item()
    if np.array_equal(cArray.get(NumCpp.Slice(startRow, stopRow, stepRow), NumCpp.Slice(startCol, stopCol, stepCol)).astype(np.uint32),
                      data[startRow:stopRow:stepRow, startCol:stopCol:stepCol]):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing all: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if cArray.all(NumCpp.Axis.NONE).astype(np.bool).item() == np.all(data).item():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing all: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.all(NumCpp.Axis.ROW).flatten().astype(np.bool), np.all(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing all: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.all(NumCpp.Axis.COL).flatten().astype(np.bool), np.all(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing any: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if cArray.any(NumCpp.Axis.NONE).astype(np.bool).item() == np.any(data).item():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing any: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.any(NumCpp.Axis.ROW).flatten().astype(np.bool), np.any(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing any: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.any(NumCpp.Axis.COL).flatten().astype(np.bool), np.any(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argmax: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.argmax(NumCpp.Axis.NONE).item(), np.argmax(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argmax: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.argmax(NumCpp.Axis.ROW).flatten(), np.argmax(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argmax: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.argmax(NumCpp.Axis.COL).flatten(), np.argmax(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argmin: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.argmin(NumCpp.Axis.NONE).item(), np.argmin(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argmin: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.argmin(NumCpp.Axis.ROW).flatten(), np.argmin(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argmin: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.argmin(NumCpp.Axis.COL).flatten(), np.argmin(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argsort: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    dataFlat = data.flatten()
    if np.array_equal(dataFlat[cArray.argsort(NumCpp.Axis.NONE).flatten().astype(np.uint32)], dataFlat[np.argsort(data, axis=None)]):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing argsort: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    pIdx = np.argsort(data, axis=0)
    cIdx = cArray.argsort(NumCpp.Axis.ROW).astype(np.uint16)
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
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    pIdx = np.argsort(data, axis=1)
    cIdx = cArray.argsort(NumCpp.Axis.COL).astype(np.uint16)
    allPass = True
    for idx, row in enumerate(data):
        if not np.array_equal(row[cIdx[idx, :]], row[pIdx[idx, :]]):
            allPass = False
            break
    if allPass:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing back', 'cyan'))
    if cArray.back() == data.flatten()[-1]:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing byteswap', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayInt(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    oldEndianess = cArray.endianess()
    cArray.byteswap()
    if np.array_equal(cArray.getNumpyArray().astype(np.uint32), data.byteswap()) and cArray.endianess() != oldEndianess:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing clip', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.clip(5, 90).astype(np.ushort), data.clip(5, 90)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    colIdx = np.random.randint(0, shape.cols, [1,]).item()
    if np.array_equal(cArray.column(colIdx).getNumpyArray().flatten(), data[:, colIdx].flatten()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing contains: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    value = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    if cArray.contains(value, NumCpp.Axis.NONE) == (value in data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing contains: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    value = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    truth = list()
    for row in data:
        truth.append(value in row)
    if np.array_equal(cArray.contains(value, NumCpp.Axis.COL).flatten(), np.asarray(truth)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing contains: Axis = ROW', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    value = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    truth = list()
    for row in data.T:
        truth.append(value in row)
    if np.array_equal(cArray.contains(value, NumCpp.Axis.ROW).flatten(), np.asarray(truth)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing copy', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.copy(), data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cumprod: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(cArray.cumprod(NumCpp.Axis.NONE).flatten().astype(np.uint32), data.cumprod()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cumprod: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(cArray.cumprod(NumCpp.Axis.ROW).astype(np.uint32), data.cumprod(axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cumprod: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(cArray.cumprod(NumCpp.Axis.COL).astype(np.uint32), data.cumprod(axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cumsum: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(cArray.cumsum(NumCpp.Axis.NONE).flatten().astype(np.uint32), data.cumsum()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cumsum: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(cArray.cumsum(NumCpp.Axis.ROW).astype(np.uint32), data.cumsum(axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing cumsum: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(cArray.cumsum(NumCpp.Axis.COL).astype(np.uint32), data.cumsum(axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing diagonal: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    offset = np.random.randint(-min(shape.rows, shape.cols), min(shape.rows, shape.cols), [1,]).item()
    if np.array_equal(cArray.diagonal(offset, NumCpp.Axis.ROW).astype(np.uint32).flatten(), data.diagonal(offset, axis1=1, axis2=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing diagonal: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    offset = np.random.randint(-min(shape.rows, shape.cols), min(shape.rows, shape.cols), [1,]).item()
    if np.array_equal(cArray.diagonal(offset, NumCpp.Axis.COL).astype(np.uint32).flatten(), data.diagonal(offset, axis1=0, axis2=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing dot vector', 'cyan'))
    size = np.random.randint(1, 100, [1,]).item()
    shape = NumCpp.Shape(1, size)
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    data2 = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if cArray1.dot(cArray2).item() == np.dot(data1, data2.T).item():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing dot array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[1].item(), np.random.randint(1, 100, [1,]).item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    data1 = np.random.randint(1, 50, [shape1.rows, shape1.cols], dtype=np.uint32)
    data2 = np.random.randint(1, 50, [shape2.rows, shape2.cols], dtype=np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(cArray1.dot(cArray2), np.dot(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing dump', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    tempDir = r'C:\Temp'
    if not os.path.exists(tempDir):
        os.mkdir(tempDir)
    tempFile = os.path.join(tempDir, 'NdArrayDump.bin')
    cArray.dump(tempFile)
    if os.path.exists(tempFile):
        filesize = os.path.getsize(tempFile)
        if filesize == data.size * 8:
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))
    else:
        print(colored('\tFAIL', 'red'))
    os.remove(tempFile)

    print(colored('Testing fill', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    fillValue = np.random.randint(1,100, [1,]).item()
    ret = cArray.fill(fillValue)
    if np.all(ret == fillValue):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing flatten', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(cArray.flatten().astype(np.uint32), data.reshape([1, data.size])):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing front', 'cyan'))
    if cArray.front() == data.flatten()[0]:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing getByIndices', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, shapeInput)
    cArray.setArray(data)
    numIndices = np.random.randint(0, shape.size(), [1, ]).item()
    indices = np.random.randint(0, shape.size(), [numIndices, ])
    cIndices = NumCpp.NdArrayInt(1, numIndices)
    cIndices.setArray(indices)
    if np.array_equal(cArray.getByIndices(cIndices).flatten(), data.flatten()[indices]):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing getByMask', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, shapeInput)
    cArray.setArray(data)
    mask = data > np.random.randint(1, data.max(), [1, ]).item()
    cMask = NumCpp.NdArrayBool(shape)
    cMask.setArray(mask)
    if np.array_equal(cArray.getByMask(cMask).flatten(), data[mask].flatten()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing item', 'cyan'))
    shape = NumCpp.Shape(1, 1)
    cArray = NumCpp.NdArray(shape)
    fillValue = np.random.randint(1, 100, [1, ]).item()
    cArray.fill(fillValue)
    if cArray.item() == fillValue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing max: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if cArray.max(NumCpp.Axis.NONE).item() == np.max(data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing max: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.max(NumCpp.Axis.ROW).flatten(), np.max(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing max: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.max(NumCpp.Axis.COL).flatten(), np.max(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing min: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if cArray.min(NumCpp.Axis.NONE).item() == np.min(data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing min: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.min(NumCpp.Axis.ROW).flatten(), np.min(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing min: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.min(NumCpp.Axis.COL).flatten(), np.min(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing mean: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if cArray.mean(NumCpp.Axis.NONE).item() == np.mean(data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing mean: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.mean(NumCpp.Axis.ROW).flatten(), np.mean(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing mean: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.mean(NumCpp.Axis.COL).flatten(), np.mean(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing median: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    while np.prod(shapeInput) % 2 == 0:
        shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if cArray.median(NumCpp.Axis.NONE).item() == np.median(data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing median: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    while shapeInput[0] % 2 == 0:
        shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.median(NumCpp.Axis.ROW).flatten(), np.median(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing median: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    while shapeInput[1] % 2 == 0:
        shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.median(NumCpp.Axis.COL).flatten(), np.median(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing nans', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    cArray.nans()
    if np.all(np.isnan(cArray.getNumpyArray())):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing nbytes', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if cArray.nbytes() == 8 * data.size:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing newbyteorder', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayInt(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(cArray.newbyteorder(NumCpp.Endian.BIG).astype(np.uint32), data.newbyteorder()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing nonzero', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 10, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.nonzero(), data.flatten().nonzero()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing norm: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if cArray.norm(NumCpp.Axis.NONE) == np.linalg.norm(data.flatten()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing norm: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    norms = cArray.norm(NumCpp.Axis.ROW).flatten()
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
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    norms = cArray.norm(NumCpp.Axis.COL).flatten()
    allPass = True
    for idx, row in enumerate(data):
        if norms[idx] != np.linalg.norm(row):
            allPass = False
            break
    if allPass:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing ones', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    ret = cArray.ones()
    if np.all(ret == 1):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing partition: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    kthElement = np.random.randint(0, shapeInput.prod(), [1,], dtype=np.uint32).item()
    partitionedArray = cArray.partition(kthElement, NumCpp.Axis.NONE).flatten()
    if (np.all(partitionedArray[:kthElement] <= partitionedArray[kthElement]) and
        np.all(partitionedArray[kthElement:] >= partitionedArray[kthElement])):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing partition: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    kthElement = np.random.randint(0, shapeInput[0], [1,], dtype=np.uint32).item()
    partitionedArray = cArray.partition(kthElement, NumCpp.Axis.ROW).transpose()
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
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    kthElement = np.random.randint(0, shapeInput[1], [1,], dtype=np.uint32).item()
    partitionedArray = cArray.partition(kthElement, NumCpp.Axis.COL)
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

    print(colored('Testing prod: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32).astype(np.double)
    cArray.setArray(data)
    if cArray.prod(NumCpp.Axis.NONE).item() == data.prod():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing prod: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(cArray.prod(NumCpp.Axis.ROW).flatten(), data.prod(axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing prod: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(cArray.prod(NumCpp.Axis.COL).flatten(), data.prod(axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing ptp: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if cArray.ptp(NumCpp.Axis.NONE).astype(np.uint32).item() == data.ptp():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing ptp: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(cArray.ptp(NumCpp.Axis.ROW).flatten().astype(np.uint32), data.ptp(axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing ptp: Axis = Col', 'cyan'))
    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    if np.array_equal(cArray.ptp(NumCpp.Axis.COL).flatten().astype(np.uint32), data.ptp(axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing put flat', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    randomIdx = np.random.randint(0, shapeInput.prod(), [1,]).item()
    randomValue = np.random.randint(1, 500, [1, ]).item()
    cArray.put(randomIdx, randomValue)
    if cArray.get(randomIdx) == randomValue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing put row/col', 'cyan'))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    randomRowIdx = np.random.randint(0, shapeInput[0], [1,]).item()
    randomColIdx = np.random.randint(0, shapeInput[1], [1,]).item()
    randomValue = np.random.randint(1, 500, [1, ]).item()
    cArray.put(randomRowIdx, randomColIdx, randomValue)
    if cArray.get(randomRowIdx, randomColIdx) == randomValue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing put slice 1D value', 'cyan'))
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    start = np.random.randint(0, shapeInput.prod() // 4, [1,]).item()
    stop = np.random.randint(start + 1, shapeInput.prod(), [1,]).item()
    step = np.random.randint(1, shapeInput.prod() // 10, [1,]).item()
    randomValue = np.random.randint(1, 500, [1, ]).item()
    inputSlice = NumCpp.Slice(start, stop, step)
    cArray.put(inputSlice, randomValue)
    if np.all(cArray.get(inputSlice).astype(np.uint32) == randomValue):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing put slice 1D values', 'cyan'))
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    start = np.random.randint(0, shapeInput.prod() // 4, [1,]).item()
    stop = np.random.randint(start + 1, shapeInput.prod(), [1,]).item()
    step = np.random.randint(1, shapeInput.prod() // 10, [1,]).item()
    inputSlice = NumCpp.Slice(start, stop, step)
    randomValues = np.random.randint(1, 500, [inputSlice.numElements(cArray.size()), ])
    cArray.put(inputSlice, randomValues)
    if np.array_equal(cArray.get(inputSlice).flatten().astype(np.uint32), randomValues):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing put slice 2D value', 'cyan'))
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1,]).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1,]).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1,]).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1,]).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1,]).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1,]).item()
    inputRowSlice = NumCpp.Slice(startRow, stopRow, stepRow)
    inputColSlice = NumCpp.Slice(startCol, stopCol, stepCol)
    randomValue = np.random.randint(1, 500, [1, ]).item()
    cArray.put(inputRowSlice, inputColSlice, randomValue)
    if np.all(cArray.get(inputRowSlice, inputColSlice).astype(np.uint32) == randomValue):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing put slice 2D values', 'cyan'))
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    startRow = np.random.randint(0, shapeInput[0] // 10, [1,]).item()
    stopRow = np.random.randint(startRow + 1, shapeInput[0], [1,]).item()
    stepRow = np.random.randint(1, shapeInput[0] // 10, [1,]).item()
    startCol = np.random.randint(0, shapeInput[1] // 10, [1,]).item()
    stopCol = np.random.randint(startCol + 1, shapeInput[1], [1,]).item()
    stepCol = np.random.randint(1, shapeInput[1] // 10, [1,]).item()
    inputRowSlice = NumCpp.Slice(startRow, stopRow, stepRow)
    inputColSlice = NumCpp.Slice(startCol, stopCol, stepCol)
    randomValues = np.random.randint(1, 500, [inputRowSlice.numElements(shape.rows), inputColSlice.numElements(shape.cols)])
    cArray.put(inputRowSlice, inputColSlice, randomValues)
    if np.array_equal(cArray.get(inputRowSlice, inputColSlice).astype(np.uint32), randomValues):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing putMask: Single Value', 'cyan'))
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    mask = data > np.random.randint(0, data.max(), [1, ]).item()
    inputValue = np.random.randint(0, 666, [1, ]).item()
    cArray.putMask(mask, inputValue)
    data[mask] = inputValue
    if np.array_equal(cArray.getNumpyArray().astype(np.uint32), data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing putMask: Multiple Values', 'cyan'))
    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    mask = data > np.random.randint(0, data.max(), [1, ]).item()
    inputValues = np.random.randint(0, 666, [np.count_nonzero(mask), ])
    cArray.putMask(mask, inputValues)
    data[mask] = inputValues
    if np.array_equal(cArray.getNumpyArray().astype(np.uint32), data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing reshape', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    newShape = NumCpp.Shape(shapeInput[1].item(), shapeInput[0].item())
    if np.array_equal(cArray.reshape(newShape), data.reshape(shapeInput[::-1])):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing resizeFast', 'cyan'))
    shapeInput1 = np.random.randint(1, 100, [2, ])
    shapeInput2 = np.random.randint(1, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput1[0].item(), shapeInput1[1].item())
    shape2 = NumCpp.Shape(shapeInput2[0].item(), shapeInput2[1].item())
    cArray = NumCpp.NdArray(shape1)
    data = np.random.randint(1, 100, [shape1.rows, shape1.cols], dtype=np.uint32)
    cArray.setArray(data)
    res = cArray.resizeFast(shape2)
    if np.all(res == 0) and cArray.shape().rows == shape2.rows and cArray.shape().cols == shape2.cols:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing resizeSlow', 'cyan'))
    shapeInput1 = np.random.randint(1, 100, [2, ])
    shapeInput2 = np.random.randint(1, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput1[0].item(), shapeInput1[1].item())
    shape2 = NumCpp.Shape(shapeInput2[0].item(), shapeInput2[1].item())
    cArray = NumCpp.NdArray(shape1)
    data = np.random.randint(1, 100, [shape1.rows, shape1.cols], dtype=np.uint32)
    cArray.setArray(data)
    res = cArray.resizeSlow(shape2)
    if cArray.shape().rows == shape2.rows and cArray.shape().cols == shape2.cols and not np.all(res == 0):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing rms: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if cArray.rms(NumCpp.Axis.NONE).item() == np.sqrt(np.mean(np.square(data))):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing rms: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.rms(NumCpp.Axis.ROW).flatten(), np.sqrt(np.mean(np.square(data), axis=0))):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing rms: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(cArray.rms(NumCpp.Axis.COL).flatten(), np.sqrt(np.mean(np.square(data), axis=1))):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    rowIdx = np.random.randint(0, shape.rows, [1,]).item()
    if np.array_equal(cArray.row(rowIdx).getNumpyArray().flatten(), data[rowIdx, :].flatten()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing round', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols)
    cArray.setArray(data)
    numRoundDecimals = np.random.randint(0, 10, [1,]).item()
    if np.array_equal(cArray.round(numRoundDecimals), np.round(data, numRoundDecimals)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing shape', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    if cArray.shape().rows == shape.rows and cArray.shape().cols == shape.cols:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing size', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    if cArray.size() == shapeInput.cumprod()[-1].item():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sort: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    d = data.flatten()
    d.sort()
    if np.array_equal(cArray.sort(NumCpp.Axis.NONE).flatten(), d):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sort: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    pSorted = np.sort(data, axis=0)
    cSorted = cArray.sort(NumCpp.Axis.ROW).astype(np.uint32)
    if np.array_equal(cSorted, pSorted):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sort: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    pSorted = np.sort(data, axis=1)
    cSorted = cArray.sort(NumCpp.Axis.COL).astype(np.uint32)
    if np.array_equal(cSorted, pSorted):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing stddev: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.round(cArray.stdev(NumCpp.Axis.NONE).item(), 10) == np.round(np.std(data), 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing stddev: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(np.round(cArray.stdev(NumCpp.Axis.ROW).flatten(), 10), np.round(np.std(data, axis=0), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing stddev: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(np.round(cArray.stdev(NumCpp.Axis.COL).flatten(), 10), np.round(np.std(data, axis=1), 10)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sum: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if cArray.sum(NumCpp.Axis.NONE).item() == np.sum(data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sum: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(cArray.sum(NumCpp.Axis.ROW).flatten(), np.sum(data, axis=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing sum: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(cArray.sum(NumCpp.Axis.COL).flatten(), np.sum(data, axis=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing swapaxes', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(cArray.swapaxes(), data.T):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing tofile bin', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if sys.platform == 'linux':
        tempDir = r'/home/' + getpass.getuser() + r'/Desktop/'
        filename = os.path.join(tempDir, 'temp.bin')
    else:
        filename = r'C:\Temp\temp.bin'
    cArray.tofile(filename, '')
    if os.path.exists(filename):
        print(colored('\tPASS', 'green'))
        os.remove(filename)
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing tofile txt', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if sys.platform == 'linux':
        tempDir = r'/home/' + getpass.getuser() + r'/Desktop/'
        filename = os.path.join(tempDir, 'temp.txt')
    else:
        filename = r'C:\Temp\temp.txt'
    cArray.tofile(filename, '\n')
    if os.path.exists(filename):
        print(colored('\tPASS', 'green'))
        os.remove(filename)
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing toStlVector', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    out = np.asarray(cArray.toStlVector())
    if np.array_equal(out, data.flatten()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing trace: Offset=Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows, [1,]).item()
    if np.array_equal(cArray.trace(offset, NumCpp.Axis.ROW), data.trace(offset, axis1=1, axis2=0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing trace: Offset=Col', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows, [1,]).item()
    if np.array_equal(cArray.trace(offset, NumCpp.Axis.COL), data.trace(offset, axis1=0, axis2=1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing transpose', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(cArray.transpose(), data.T):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing var: Axis = None', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.round(cArray.var(NumCpp.Axis.NONE).item(), 9) == np.round(np.var(data), 9):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing var: Axis = Row', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(np.round(cArray.var(NumCpp.Axis.ROW).flatten(), 9), np.round(np.var(data, axis=0), 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing var: Axis = Column', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if np.array_equal(np.round(cArray.var(NumCpp.Axis.COL).flatten(), 9), np.round(np.var(data, axis=1), 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing zeros', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    cArray.zeros()
    if np.all(cArray.getNumpyArray() == 0):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator + scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 100, [1,]).item()
    if np.array_equal(cArray.operatorPlusScaler(randScaler), data + randScaler):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator + array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(cArray1.operatorPlusArray(cArray2), data1 + data2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator - scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 100, [1,]).item()
    if np.array_equal(cArray.operatorMinusScaler(randScaler), data - randScaler):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator - array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(cArray1.operatorMinusArray(cArray2), data1 - data2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator * scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 100, [1,]).item()
    if np.array_equal(cArray.operatorMultiplyScaler(randScaler), data * randScaler):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator * array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(cArray1.operatorMultiplyArray(cArray2), data1 * data2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator / scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 100, [1,]).item()
    if np.array_equal(cArray.operatorDivideScaler(randScaler), data / randScaler):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator / array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.double)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(cArray1.operatorDivideArray(cArray2), data1 / data2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator % scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayInt(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    randScaler = np.random.randint(1, 100, [1,]).item()
    if np.array_equal(cArray.operatorModulusScaler(randScaler), data % randScaler):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator % array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayInt(shape)
    cArray2 = NumCpp.NdArrayInt(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols])
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(cArray1.operatorModulusArray(cArray2), data1 % data2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator | scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayInt(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 100, [1,]).item()
    if np.array_equal(cArray.operatorBitwiseOrScaler(randScaler), np.bitwise_or(data, randScaler)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator | array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayInt(shape)
    cArray2 = NumCpp.NdArrayInt(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(cArray1.operatorBitwiseOrArray(cArray2), np.bitwise_or(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator & scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayInt(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 100, [1,]).item()
    if np.array_equal(cArray.operatorBitwiseAndScaler(randScaler), np.bitwise_and(data, randScaler)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator & array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayInt(shape)
    cArray2 = NumCpp.NdArrayInt(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(cArray1.operatorBitwiseAndArray(cArray2), np.bitwise_and(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator ^ scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayInt(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 100, [1,]).item()
    if np.array_equal(cArray.operatorBitwiseXorScaler(randScaler), np.bitwise_xor(data, randScaler)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator ^ array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayInt(shape)
    cArray2 = NumCpp.NdArrayInt(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(cArray1.operatorBitwiseXorArray(cArray2), np.bitwise_xor(data1, data2)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator ~', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayInt(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    if np.array_equal(cArray.operatorBitwiseNot(), ~data):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    # print(colored('Testing operator !', 'cyan'))
    # shapeInput = np.random.randint(1, 100, [2, ])
    # shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    # cArray = NumCpp.NdArrayInt(shape)
    # data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    # cArray.setArray(data)
    # if np.array_equal(cArray.operatorNot(), ~data):
    #     print(colored('\tPASS', 'green'))
    # else:
    #     print(colored('\tFAIL', 'red'))

    print(colored('Testing operator == scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    if np.array_equal(cArray.operatorEquality(value), data == value):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator == array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(cArray1.operatorEquality(cArray2), data1 == data2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator != scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    if np.array_equal(cArray.operatorNotEquality(value), data != value):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator != array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(cArray1.operatorNotEquality(cArray2), data1 != data2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator < scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    if np.array_equal(cArray.operatorLess(value), data < value):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator < array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(cArray1.operatorLess(cArray2), data1 < data2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator > scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    if np.array_equal(cArray.operatorGreater(value), data > value):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator > array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(cArray1.operatorGreater(cArray2), data1 > data2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator <= scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    if np.array_equal(cArray.operatorLessEqual(value), data <= value):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator <= array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(cArray1.operatorLessEqual(cArray2), data1 <= data2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator >= scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    if np.array_equal(cArray.operatorGreaterEqual(value), data >= value):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator >= array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    if np.array_equal(cArray1.operatorGreaterEqual(cArray2), data1 >= data2):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator << scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayInt(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 10, [1,]).item()
    if np.array_equal(cArray.operatorBitshiftLeft(randScaler), data << randScaler):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator >> scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayInt(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 10, [1,]).item()
    if np.array_equal(cArray.operatorBitshiftRight(randScaler), data >> randScaler):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator post++ scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    if np.array_equal(cArray.operatorPostPlusPlus(), data) and np.array_equal(cArray.getNumpyArray(), data + 1):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator post-- scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    if np.array_equal(cArray.operatorPostMinusMinus(), data) and np.array_equal(cArray.getNumpyArray(), data - 1):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator pre++ scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    if np.array_equal(cArray.operatorPrePlusPlus(), data + 1):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator pre-- scaler', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    if np.array_equal(cArray.operatorPreMinusMinus(), data - 1):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

####################################################################################
if __name__ == '__main__':
    doTest()
