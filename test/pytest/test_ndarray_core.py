import numpy as np
from functools import reduce
import os
import getpass
import sys
sys.path.append(os.path.abspath(r'../lib'))
import NumCpp  # noqa E402


####################################################################################
def factors(n):
    return set(reduce(list.__add__,
                      ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


####################################################################################
def test_ndarray():
    np.random.seed(666)

    cArray = NumCpp.NdArray()
    assert cArray.shape().rows == 0
    assert cArray.shape().cols == 0
    assert cArray.size() == 0

    numRowsCols = np.random.randint(1, 100, [1, ]).item()
    cArray = NumCpp.NdArray(numRowsCols)
    assert cArray.shape().rows == numRowsCols
    assert cArray.shape().cols == numRowsCols
    assert cArray.size() == numRowsCols**2

    numRowsCols = np.random.randint(1, 100, [2, ])
    cArray = NumCpp.NdArray(numRowsCols[0].item(), numRowsCols[1].item())
    assert cArray.shape().rows == numRowsCols[0]
    assert cArray.shape().cols == numRowsCols[1]
    assert cArray.size() == numRowsCols.prod()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    assert cArray.shape().rows == shape.rows
    assert cArray.shape().cols == shape.cols
    assert cArray.size() == shape.rows * shape.cols

    assert NumCpp.NdArray.test1DListContructor()
    assert NumCpp.NdArray.test2DListContructor()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(np.random.randint(1, 100, [shape.rows, shape.cols]))
    cArray2 = NumCpp.NdArray(cArray)
    assert np.array_equal(np.round(cArray2.getNumpyArray(), 10), np.round(cArray.getNumpyArray(), 10))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    randomIdx = np.random.randint(0, shapeInput.prod(), [1, ]).item()
    assert cArray.get(randomIdx) == data.flatten()[randomIdx]

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    randomRowIdx = np.random.randint(0, shapeInput[0], [1, ]).item()
    randomColIdx = np.random.randint(0, shapeInput[1], [1, ]).item()
    assert cArray.get(randomRowIdx, randomColIdx) == data[randomRowIdx, randomColIdx]

    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    start = np.random.randint(0, shapeInput.prod() // 10, [1, ]).item()
    stop = np.random.randint(start + 1, shapeInput.prod(), [1, ]).item()
    step = np.random.randint(1, shapeInput.prod() // 10, [1, ]).item()
    assert np.array_equal(cArray.get(NumCpp.Slice(start, stop, step)).flatten().astype(np.uint32),
                          data.flatten()[start:stop:step])

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
    assert np.array_equal(cArray.get(NumCpp.Slice(startRow, stopRow, stepRow),
                                     NumCpp.Slice(startCol, stopCol, stepCol)).astype(np.uint32),
                          data[startRow:stopRow:stepRow, startCol:stopCol:stepCol])

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert cArray.all(NumCpp.Axis.NONE).astype(np.bool).item() == np.all(data).item()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.all(NumCpp.Axis.ROW).flatten().astype(np.bool), np.all(data, axis=0))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.all(NumCpp.Axis.COL).flatten().astype(np.bool), np.all(data, axis=1))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert cArray.any(NumCpp.Axis.NONE).astype(np.bool).item() == np.any(data).item()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.any(NumCpp.Axis.ROW).flatten().astype(np.bool), np.any(data, axis=0))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.any(NumCpp.Axis.COL).flatten().astype(np.bool), np.any(data, axis=1))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.argmax(NumCpp.Axis.NONE).item(), np.argmax(data))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.argmax(NumCpp.Axis.ROW).flatten(), np.argmax(data, axis=0))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.argmax(NumCpp.Axis.COL).flatten(), np.argmax(data, axis=1))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.argmin(NumCpp.Axis.NONE).item(), np.argmin(data))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.argmin(NumCpp.Axis.ROW).flatten(), np.argmin(data, axis=0))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.argmin(NumCpp.Axis.COL).flatten(), np.argmin(data, axis=1))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    dataFlat = data.flatten()
    assert np.array_equal(dataFlat[cArray.argsort(NumCpp.Axis.NONE).flatten().astype(np.uint32)],
                          dataFlat[np.argsort(data, axis=None)])

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
    assert allPass

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
    assert allPass
    
    assert cArray.back() == data.flatten()[-1]
    assert cArray.backReference() == data.flatten()[-1]

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    oldEndianess = cArray.endianess()
    cArray.byteswap()
    assert np.array_equal(cArray.getNumpyArray().astype(np.uint32),
                          data.byteswap()) and cArray.endianess() != oldEndianess

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.clip(5, 90).astype(np.ushort), data.clip(5, 90))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    colIdx = np.random.randint(0, shape.cols, [1, ]).item()
    assert np.array_equal(cArray.column(colIdx).getNumpyArray().flatten(), data[:, colIdx].flatten())

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    value = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    assert cArray.contains(value, NumCpp.Axis.NONE) == (value in data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    value = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    truth = list()
    for row in data:
        truth.append(value in row)
    assert np.array_equal(cArray.contains(value, NumCpp.Axis.COL).flatten(), np.asarray(truth))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    value = np.random.randint(0, 100, [1, ]).item()
    cArray.setArray(data)
    truth = list()
    for row in data.T:
        truth.append(value in row)
    assert np.array_equal(cArray.contains(value, NumCpp.Axis.ROW).flatten(), np.asarray(truth))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.copy(), data)

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(cArray.cumprod(NumCpp.Axis.NONE).flatten().astype(np.uint32), data.cumprod())

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(cArray.cumprod(NumCpp.Axis.ROW).astype(np.uint32), data.cumprod(axis=0))

    shapeInput = np.random.randint(1, 5, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 4, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(cArray.cumprod(NumCpp.Axis.COL).astype(np.uint32), data.cumprod(axis=1))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(cArray.cumsum(NumCpp.Axis.NONE).flatten().astype(np.uint32), data.cumsum())

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(cArray.cumsum(NumCpp.Axis.ROW).astype(np.uint32), data.cumsum(axis=0))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(cArray.cumsum(NumCpp.Axis.COL).astype(np.uint32), data.cumsum(axis=1))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    offset = np.random.randint(-min(shape.rows, shape.cols), min(shape.rows, shape.cols), [1, ]).item()
    assert np.array_equal(cArray.diagonal(offset, NumCpp.Axis.ROW).astype(np.uint32).flatten(),
                          data.diagonal(offset, axis1=0, axis2=1))
    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    offset = np.random.randint(-min(shape.rows, shape.cols), min(shape.rows, shape.cols), [1, ]).item()
    assert np.array_equal(cArray.diagonal(offset, NumCpp.Axis.COL).astype(np.uint32).flatten(),
                          data.diagonal(offset, axis1=1, axis2=0))

    size = np.random.randint(1, 100, [1, ]).item()
    shape = NumCpp.Shape(1, size)
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    data2 = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert cArray1.dot(cArray2).item() == np.dot(data1, data2.T).item()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shapeInput[1].item(), np.random.randint(1, 100, [1, ]).item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    data1 = np.random.randint(1, 50, [shape1.rows, shape1.cols], dtype=np.uint32)
    data2 = np.random.randint(1, 50, [shape2.rows, shape2.cols], dtype=np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(cArray1.dot(cArray2), np.dot(data1, data2))

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
        assert filesize == data.size * 8
    else:
        assert False
    os.remove(tempFile)

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    fillValue = np.random.randint(1, 100, [1, ]).item()
    ret = cArray.fill(fillValue)
    assert np.all(ret == fillValue)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 10, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.flatnonzero().flatten().astype(np.uint32), np.flatnonzero(data))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(cArray.flatten().astype(np.uint32), data.reshape([1, data.size]))
    assert cArray.front() == data.flatten()[0]
    assert cArray.frontReference() == data.flatten()[0]

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, shapeInput)
    cArray.setArray(data)
    numIndices = np.random.randint(0, shape.size(), [1, ]).item()
    indices = np.random.randint(0, shape.size(), [numIndices, ])
    cIndices = NumCpp.NdArrayUInt32(1, numIndices)
    cIndices.setArray(indices)
    assert np.array_equal(cArray.getByIndices(cIndices).flatten(), data.flatten()[indices])

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, shapeInput)
    cArray.setArray(data)
    mask = data > np.random.randint(1, np.max(data), [1, ]).item()
    cMask = NumCpp.NdArrayBool(shape)
    cMask.setArray(mask)
    assert np.array_equal(cArray.getByMask(cMask).flatten(), data[mask].flatten())

    shapeInput = np.random.randint(2, 100, [2, ])
    sizeInput = np.random.randint(2, 100, [1, ]).item()
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(1, sizeInput)
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    assert not cArray1.isflat() and cArray2.isflat()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    dataSorted = np.sort(data.flatten()).reshape(data.shape)
    cArray.setArray(data)
    if not cArray.issorted(NumCpp.Axis.NONE).item():
        cArray.setArray(dataSorted)
        assert cArray.issorted(NumCpp.Axis.NONE).item()
    else:
        assert False

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    dataSorted = np.sort(data, axis=0).reshape(data.shape)
    cArray.setArray(data)
    if not np.all(cArray.issorted(NumCpp.Axis.ROW)):
        cArray.setArray(dataSorted)
        assert np.all(cArray.issorted(NumCpp.Axis.ROW))
    else:
        assert False

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    dataSorted = np.sort(data, axis=1).reshape(data.shape)
    cArray.setArray(data)
    if not np.all(cArray.issorted(NumCpp.Axis.COL)):
        cArray.setArray(dataSorted)
        assert np.all(cArray.issorted(NumCpp.Axis.COL))
    else:
        assert False

    while True:
        shapeInput = np.random.randint(2, 100, [2, ])
        if np.prod(shapeInput) != np.square(shapeInput[0]):
            break
    sizeInput = np.random.randint(2, 100, [1, ]).item()
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(sizeInput, sizeInput)
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    assert not cArray1.issquare()
    assert cArray2.issquare()

    shape = NumCpp.Shape(1, 1)
    cArray = NumCpp.NdArray(shape)
    fillValue = np.random.randint(1, 100, [1, ]).item()
    cArray.fill(fillValue)
    assert cArray.item() == fillValue

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert cArray.max(NumCpp.Axis.NONE).item() == np.max(data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.max(NumCpp.Axis.ROW).flatten(), np.max(data, axis=0))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.max(NumCpp.Axis.COL).flatten(), np.max(data, axis=1))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert cArray.min(NumCpp.Axis.NONE).item() == np.min(data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.min(NumCpp.Axis.ROW).flatten(), np.min(data, axis=0))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.min(NumCpp.Axis.COL).flatten(), np.min(data, axis=1))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert cArray.mean(NumCpp.Axis.NONE).item() == np.mean(data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.mean(NumCpp.Axis.ROW).flatten(), np.mean(data, axis=0))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.mean(NumCpp.Axis.COL).flatten(), np.mean(data, axis=1))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert cArray.median(NumCpp.Axis.NONE).item() == np.median(data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.median(NumCpp.Axis.ROW).flatten(), np.median(data, axis=0))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.median(NumCpp.Axis.COL).flatten(), np.median(data, axis=1))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    cArray.nans()
    assert np.all(np.isnan(cArray.getNumpyArray()))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert cArray.nbytes() == 8 * data.size

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(cArray.newbyteorder(NumCpp.Endian.BIG).astype(np.uint32), data.newbyteorder())

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert cArray.none(NumCpp.Axis.NONE).astype(np.bool).item() == np.logical_not(np.any(data).item())

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.none(NumCpp.Axis.ROW).flatten().astype(np.bool), np.logical_not(np.any(data, axis=0)))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.none(NumCpp.Axis.COL).flatten().astype(np.bool), np.logical_not(np.any(data, axis=1)))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 10, [shape.rows, shape.cols])
    cArray.setArray(data)
    rowsC, colsC = cArray.nonzero()
    rows, cols = data.nonzero()
    assert np.array_equal(rowsC.flatten(), rows)
    assert np.array_equal(colsC.flatten(), cols)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert cArray.norm(NumCpp.Axis.NONE).item() == np.linalg.norm(data.flatten())

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
    assert allPass

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
    assert allPass

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    ret = cArray.ones()
    assert np.all(ret == 1)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    kthElement = np.random.randint(0, shapeInput.prod(), [1, ], dtype=np.uint32).item()
    partitionedArray = cArray.partition(kthElement, NumCpp.Axis.NONE).flatten()
    assert (np.all(partitionedArray[:kthElement] <= partitionedArray[kthElement]) and
            np.all(partitionedArray[kthElement:] >= partitionedArray[kthElement]))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    kthElement = np.random.randint(0, shapeInput[0], [1, ], dtype=np.uint32).item()
    partitionedArray = cArray.partition(kthElement, NumCpp.Axis.ROW).transpose()
    allPass = True
    for row in partitionedArray:
        if not (np.all(row[:kthElement] <= row[kthElement]) and
                np.all(row[kthElement:] >= row[kthElement])):
            allPass = False
            break
    assert allPass

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    kthElement = np.random.randint(0, shapeInput[1], [1, ], dtype=np.uint32).item()
    partitionedArray = cArray.partition(kthElement, NumCpp.Axis.COL)
    allPass = True
    for row in partitionedArray:
        if not (np.all(row[:kthElement] <= row[kthElement]) and
                np.all(row[kthElement:] >= row[kthElement])):
            allPass = False
            break
    assert allPass

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32).astype(np.double)
    cArray.setArray(data)
    assert cArray.prod(NumCpp.Axis.NONE).item() == data.prod()

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(cArray.prod(NumCpp.Axis.ROW).flatten(), data.prod(axis=0))

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(cArray.prod(NumCpp.Axis.COL).flatten(), data.prod(axis=1))

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert cArray.ptp(NumCpp.Axis.NONE).astype(np.uint32).item() == data.ptp()

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(cArray.ptp(NumCpp.Axis.ROW).flatten().astype(np.uint32), data.ptp(axis=0))

    shapeInput = np.random.randint(1, 10, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 15, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(cArray.ptp(NumCpp.Axis.COL).flatten().astype(np.uint32), data.ptp(axis=1))

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    randomIdx = np.random.randint(0, shapeInput.prod(), [1, ]).item()
    randomValue = np.random.randint(1, 500, [1, ]).item()
    cArray.put(randomIdx, randomValue)
    assert cArray.get(randomIdx) == randomValue

    shapeInput = np.random.randint(1, 50, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    randomRowIdx = np.random.randint(0, shapeInput[0], [1, ]).item()
    randomColIdx = np.random.randint(0, shapeInput[1], [1, ]).item()
    randomValue = np.random.randint(1, 500, [1, ]).item()
    cArray.put(randomRowIdx, randomColIdx, randomValue)
    assert cArray.get(randomRowIdx, randomColIdx) == randomValue

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
    cArray.put(inputSlice, randomValue)
    assert np.all(cArray.get(inputSlice).astype(np.uint32) == randomValue)

    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    start = np.random.randint(0, shapeInput.prod() // 4, [1, ]).item()
    stop = np.random.randint(start + 1, shapeInput.prod(), [1, ]).item()
    step = np.random.randint(1, shapeInput.prod() // 10, [1, ]).item()
    inputSlice = NumCpp.Slice(start, stop, step)
    randomValues = np.random.randint(1, 500, [inputSlice.numElements(cArray.size()), ])
    cArray.put(inputSlice, randomValues)
    assert np.array_equal(cArray.get(inputSlice).flatten().astype(np.uint32), randomValues)

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
    cArray.put(inputRowSlice, inputColSlice, randomValue)
    assert np.all(cArray.get(inputRowSlice, inputColSlice).astype(np.uint32) == randomValue)

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
    cArray.put(inputRowSlice, inputColSlice, randomValues)
    assert np.array_equal(cArray.get(inputRowSlice, inputColSlice).astype(np.uint32), randomValues)

    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    mask = data > np.random.randint(0, np.max(data), [1, ]).item()
    inputValue = np.random.randint(0, 666, [1, ]).item()
    cArray.putMask(mask, inputValue)
    data[mask] = inputValue
    assert np.array_equal(cArray.getNumpyArray().astype(np.uint32), data)

    shapeInput = np.random.randint(100, 500, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 50, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    mask = data > np.random.randint(0, np.max(data), [1, ]).item()
    inputValues = np.random.randint(0, 666, [np.count_nonzero(mask), ])
    cArray.putMask(mask, inputValues)
    data[mask] = inputValues
    assert np.array_equal(cArray.getNumpyArray().astype(np.uint32), data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    assert np.array_equal(cArray.ravel().flatten(), data.ravel())

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayInt32(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    oldValue = np.random.randint(1, 100, 1).item()
    newValue = np.random.randint(1, 100, 1).item()
    dataCopy = data.copy()
    dataCopy[dataCopy == oldValue] = newValue
    assert np.array_equal(cArray.replace(oldValue, newValue), dataCopy)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    newShape = data.size
    assert np.array_equal(cArray.reshape(newShape), data.reshape(1, newShape))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    newShape = NumCpp.Shape(shapeInput[1].item(), shapeInput[0].item())
    assert np.array_equal(cArray.reshape(newShape), data.reshape(shapeInput[::-1]))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    newShape = NumCpp.Shape(shapeInput[1].item(), shapeInput[0].item())
    assert np.array_equal(cArray.reshapeList(newShape), data.reshape(shapeInput[::-1]))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    newNumCols = np.random.choice(np.array(list(factors(data.size))), 1).item()
    assert np.array_equal(cArray.reshape(-1, newNumCols), data.reshape(-1, newNumCols))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    newNumRows = np.random.choice(np.array(list(factors(data.size))), 1).item()
    assert np.array_equal(cArray.reshape(newNumRows, -1), data.reshape(newNumRows, -1))

    shapeInput1 = np.random.randint(1, 100, [2, ])
    shapeInput2 = np.random.randint(1, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput1[0].item(), shapeInput1[1].item())
    shape2 = NumCpp.Shape(shapeInput2[0].item(), shapeInput2[1].item())
    cArray = NumCpp.NdArray(shape1)
    data = np.random.randint(1, 100, [shape1.rows, shape1.cols], dtype=np.uint32)
    cArray.setArray(data)
    res = cArray.resizeFast(shape2)
    assert np.all(res == 0)
    assert cArray.shape().rows == shape2.rows
    assert cArray.shape().cols == shape2.cols

    shapeInput1 = np.random.randint(1, 100, [2, ])
    shapeInput2 = np.random.randint(1, 100, [2, ])
    shape1 = NumCpp.Shape(shapeInput1[0].item(), shapeInput1[1].item())
    shape2 = NumCpp.Shape(shapeInput2[0].item(), shapeInput2[1].item())
    cArray = NumCpp.NdArray(shape1)
    data = np.random.randint(1, 100, [shape1.rows, shape1.cols], dtype=np.uint32)
    cArray.setArray(data)
    res = cArray.resizeSlow(shape2)
    assert cArray.shape().rows == shape2.rows
    assert cArray.shape().cols == shape2.cols
    assert not np.all(res == 0)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert cArray.rms(NumCpp.Axis.NONE).item() == np.sqrt(np.mean(np.square(data)))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.rms(NumCpp.Axis.ROW).flatten(), np.sqrt(np.mean(np.square(data), axis=0)))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(cArray.rms(NumCpp.Axis.COL).flatten(), np.sqrt(np.mean(np.square(data), axis=1)))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    rowIdx = np.random.randint(0, shape.rows, [1, ]).item()
    assert np.array_equal(cArray.row(rowIdx).getNumpyArray().flatten(), data[rowIdx, :].flatten())

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols)
    cArray.setArray(data)
    numRoundDecimals = np.random.randint(0, 10, [1, ]).item()
    assert np.array_equal(cArray.round(numRoundDecimals), np.round(data, numRoundDecimals))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    assert cArray.shape().rows == shape.rows and cArray.shape().cols == shape.cols

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    assert cArray.size() == shapeInput.cumprod()[-1].item()

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    d = data.flatten()
    d.sort()
    assert np.array_equal(cArray.sort(NumCpp.Axis.NONE).flatten(), d)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    pSorted = np.sort(data, axis=0)
    cSorted = cArray.sort(NumCpp.Axis.ROW).astype(np.uint32)
    assert np.array_equal(cSorted, pSorted)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    pSorted = np.sort(data, axis=1)
    cSorted = cArray.sort(NumCpp.Axis.COL).astype(np.uint32)
    assert np.array_equal(cSorted, pSorted)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.round(cArray.stdev(NumCpp.Axis.NONE).item(), 10) == np.round(np.std(data), 10)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(np.round(cArray.stdev(NumCpp.Axis.ROW).flatten(), 10), np.round(np.std(data, axis=0), 10))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(np.round(cArray.stdev(NumCpp.Axis.COL).flatten(), 10), np.round(np.std(data, axis=1), 10))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert cArray.sum(NumCpp.Axis.NONE).item() == np.sum(data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(cArray.sum(NumCpp.Axis.ROW).flatten(), np.sum(data, axis=0))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(cArray.sum(NumCpp.Axis.COL).flatten(), np.sum(data, axis=1))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(cArray.swapaxes(), data.T)

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
    assert os.path.exists(filename)
    os.remove(filename)

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
    assert os.path.exists(filename)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    out = np.asarray(cArray.toStlVector())
    assert np.array_equal(out, data.flatten())

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows, [1, ]).item()
    assert np.array_equal(cArray.trace(offset, NumCpp.Axis.ROW), data.trace(offset, axis1=1, axis2=0))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    offset = np.random.randint(0, shape.rows, [1, ]).item()
    assert np.array_equal(cArray.trace(offset, NumCpp.Axis.COL), data.trace(offset, axis1=0, axis2=1))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(cArray.transpose(), data.T)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.round(cArray.var(NumCpp.Axis.NONE).item(), 9) == np.round(np.var(data), 9)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(np.round(cArray.var(NumCpp.Axis.ROW).flatten(), 9), np.round(np.var(data, axis=0), 9))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    assert np.array_equal(np.round(cArray.var(NumCpp.Axis.COL).flatten(), 9), np.round(np.var(data, axis=1), 9))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    cArray.zeros()
    assert np.all(cArray.getNumpyArray() == 0)
