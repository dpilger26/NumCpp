import numpy as np
import os
import sys

import NumCppPy as NumCpp  # noqa E402

np.random.seed(666)


####################################################################################
def test_default_constructor():
    dataCube = NumCpp.DataCube()
    assert dataCube.isempty()


####################################################################################
def test_size_constructor():
    size = np.random.randint(10, 20, [1, ]).item()
    dataCube = NumCpp.DataCube(size)
    assert dataCube.sizeZ() == 0


####################################################################################
def test_methods():
    shape = np.random.randint(10, 100, [3, ])
    cShape = NumCpp.Shape(shape[0].item(), shape[1].item())
    data = np.random.randint(0, 100, shape)
    dataCube = NumCpp.DataCube()
    for frame in range(shape[-1]):
        cArray = NumCpp.NdArray(cShape)
        cArray.setArray(data[:, :, frame])
        dataCube.push_back(cArray)

    assert not dataCube.isempty()
    assert dataCube.sizeZ() == shape[-1]
    assert dataCube.shape() == cShape
    assert np.array_equal(dataCube.back().getNumpyArray(), data[:, :, -1])
    assert np.array_equal(dataCube.front().getNumpyArray(), data[:, :, 0])

    allPass = True
    for frame in range(shape[-1]):
        if not np.array_equal(dataCube[frame].getNumpyArray(), data[:, :, frame]):
            allPass = False
            break
    assert allPass

    for frame in range(shape[-1]):
        if not np.array_equal(dataCube.at(frame).getNumpyArray(), data[:, :, frame]):
            allPass = False
            break
    assert allPass

    tempDir = r'C:\Temp'
    if not os.path.exists(tempDir):
        os.mkdir(tempDir)
    tempFile = os.path.join(tempDir, 'DataCube.bin')
    dataCube.dump(tempFile)
    if os.path.exists(tempFile):
        filesize = os.path.getsize(tempFile)
        assert filesize == data.size * 8
    else:
        assert False
    os.remove(tempFile)

    sizeInitial = dataCube.sizeZ()
    sizeNow = sizeInitial
    allPass = True
    for idx in range(sizeInitial):
        dataCube.pop_back()
        sizeNow -= 1

        if dataCube.sizeZ() != sizeNow:
            allPass = False

    assert dataCube.isempty()
    assert allPass


def test_z_slices():
    shape = np.random.randint(30, 100, [3, ], dtype=np.uint32)
    data = np.random.randint(0, 100, shape)
    dataCube = NumCpp.DataCube(shape[-1])
    cArray = NumCpp.NdArray(dataCube.shape())
    for i in range(shape[-1]):
        cArray.setArray(data[:, :, i])
        dataCube.push_back(cArray)

    rowIdx = np.random.randint(0, shape[0])
    colIdx = np.random.randint(0, shape[1])
    idx = int(rowIdx * shape[1] + colIdx)

    assert np.array_equal(dataCube.sliceZAllat(
        idx).flatten(), data[rowIdx, colIdx, :])

    shape = np.random.randint(30, 100, [3, ], dtype=np.uint32)
    data = np.random.randint(0, 100, shape)
    dataCube = NumCpp.DataCube(shape[-1])
    cArray = NumCpp.NdArray(dataCube.shape())
    for i in range(shape[-1]):
        cArray.setArray(data[:, :, i])
        dataCube.push_back(cArray)

    rowIdx = np.random.randint(0, shape[0])
    colIdx = np.random.randint(0, shape[1])
    idx = int(rowIdx * shape[1] + colIdx)

    zStart = 2
    zEnd = int(shape[-1] - 2)
    zStep = 3
    zSlice = NumCpp.Slice(zStart, zEnd, zStep)

    assert np.array_equal(dataCube.sliceZat(idx, zSlice).flatten(),
                          data[rowIdx, colIdx, zStart:zEnd:zStep])

    shape = np.random.randint(30, 100, [3, ], dtype=np.uint32)
    data = np.random.randint(0, 100, shape)
    dataCube = NumCpp.DataCube(shape[-1])
    cArray = NumCpp.NdArray(dataCube.shape())
    for i in range(shape[-1]):
        cArray.setArray(data[:, :, i])
        dataCube.push_back(cArray)

    rowIdx = np.random.randint(0, shape[0])
    colIdx = np.random.randint(0, shape[1])

    assert np.array_equal(dataCube.sliceZAllat(
        rowIdx, colIdx).flatten(), data[rowIdx, colIdx, :])

    shape = np.random.randint(30, 100, [3, ], dtype=np.uint32)
    data = np.random.randint(0, 100, shape)
    dataCube = NumCpp.DataCube(shape[-1])
    cArray = NumCpp.NdArray(dataCube.shape())
    for i in range(shape[-1]):
        cArray.setArray(data[:, :, i])
        dataCube.push_back(cArray)

    rowIdx = np.random.randint(0, shape[0])
    colIdx = np.random.randint(0, shape[1])

    zStart = 2
    zEnd = int(shape[-1] - 2)
    zStep = 3
    zSlice = NumCpp.Slice(zStart, zEnd, zStep)

    assert np.array_equal(dataCube.sliceZat(rowIdx, colIdx, zSlice).flatten(),
                          data[rowIdx, colIdx, zStart:zEnd:zStep])

    shape = np.random.randint(30, 100, [3, ], dtype=np.uint32)
    data = np.random.randint(0, 100, shape)
    dataCube = NumCpp.DataCube(shape[-1])
    cArray = NumCpp.NdArray(dataCube.shape())
    for i in range(shape[-1]):
        cArray.setArray(data[:, :, i])
        dataCube.push_back(cArray)

    rowStart = 2
    rowEnd = int(shape[0] - 2)
    rowStep = 3
    rowSlice = NumCpp.Slice(rowStart, rowEnd, rowStep)
    colIdx = np.random.randint(0, shape[1])

    assert np.array_equal(dataCube.sliceZAllat(
        rowSlice, colIdx), data[rowStart:rowEnd:rowStep, colIdx, :])

    shape = np.random.randint(30, 100, [3, ], dtype=np.uint32)
    data = np.random.randint(0, 100, shape)
    dataCube = NumCpp.DataCube(shape[-1])
    cArray = NumCpp.NdArray(dataCube.shape())
    for i in range(shape[-1]):
        cArray.setArray(data[:, :, i])
        dataCube.push_back(cArray)

    rowStart = 2
    rowEnd = int(shape[0] - 2)
    rowStep = 3
    rowSlice = NumCpp.Slice(rowStart, rowEnd, rowStep)
    colIdx = np.random.randint(0, shape[1])

    zStart = 2
    zEnd = int(shape[-1] - 2)
    zStep = 3
    zSlice = NumCpp.Slice(zStart, zEnd, zStep)

    assert np.array_equal(dataCube.sliceZat(rowSlice, colIdx, zSlice),
                          data[rowStart:rowEnd:rowStep, colIdx, zStart:zEnd:zStep])

    shape = np.random.randint(30, 100, [3, ], dtype=np.uint32)
    data = np.random.randint(0, 100, shape)
    dataCube = NumCpp.DataCube(shape[-1])
    cArray = NumCpp.NdArray(dataCube.shape())
    for i in range(shape[-1]):
        cArray.setArray(data[:, :, i])
        dataCube.push_back(cArray)

    rowIdx = np.random.randint(0, shape[0])
    colStart = 2
    colEnd = int(shape[1] - 2)
    colStep = 3
    colSlice = NumCpp.Slice(colStart, colEnd, colStep)

    assert np.array_equal(dataCube.sliceZAllat(
        rowIdx, colSlice), data[rowIdx, colStart:colEnd:colStep, :])

    shape = np.random.randint(30, 100, [3, ], dtype=np.uint32)
    data = np.random.randint(0, 100, shape)
    dataCube = NumCpp.DataCube(shape[-1])
    cArray = NumCpp.NdArray(dataCube.shape())
    for i in range(shape[-1]):
        cArray.setArray(data[:, :, i])
        dataCube.push_back(cArray)

    rowIdx = np.random.randint(0, shape[0])
    colStart = 2
    colEnd = int(shape[1] - 2)
    colStep = 3
    colSlice = NumCpp.Slice(colStart, colEnd, colStep)

    zStart = 2
    zEnd = int(shape[-1] - 2)
    zStep = 3
    zSlice = NumCpp.Slice(zStart, zEnd, zStep)

    assert np.array_equal(dataCube.sliceZat(rowIdx, colSlice, zSlice),
                          data[rowIdx, colStart:colEnd:colStep, zStart:zEnd:zStep])

    shape = np.random.randint(30, 100, [3, ], dtype=np.uint32)
    data = np.random.randint(0, 100, shape)
    dataCube = NumCpp.DataCube(shape[-1])
    cArray = NumCpp.NdArray(dataCube.shape())
    for i in range(shape[-1]):
        cArray.setArray(data[:, :, i])
        dataCube.push_back(cArray)

    rowStart = 1
    rowEnd = int(shape[0] - 1)
    rowStep = 2
    rowSlice = NumCpp.Slice(rowStart, rowEnd, rowStep)
    colStart = 2
    colEnd = int(shape[1] - 2)
    colStep = 3
    colSlice = NumCpp.Slice(colStart, colEnd, colStep)

    zSlice = dataCube.sliceZAllat(rowSlice, colSlice)
    for z in range(shape[-1]):
        assert np.array_equal(zSlice[z].getNumpyArray(
        ), data[rowStart:rowEnd:rowStep, colStart:colEnd:colStep, z])

    shape = np.random.randint(30, 100, [3, ], dtype=np.uint32)
    data = np.random.randint(0, 100, shape)
    dataCube = NumCpp.DataCube(shape[-1])
    cArray = NumCpp.NdArray(dataCube.shape())
    for i in range(shape[-1]):
        cArray.setArray(data[:, :, i])
        dataCube.push_back(cArray)

    rowStart = 1
    rowEnd = int(shape[0] - 1)
    rowStep = 2
    rowSlice = NumCpp.Slice(rowStart, rowEnd, rowStep)
    colStart = 2
    colEnd = int(shape[1] - 2)
    colStep = 3
    colSlice = NumCpp.Slice(colStart, colEnd, colStep)

    zStart = 2
    zEnd = int(shape[-1] - 2)
    zStep = 3
    zSlice = NumCpp.Slice(zStart, zEnd, zStep)

    zSliceDataNC = dataCube.sliceZat(rowSlice, colSlice, zSlice)
    zSliceData = data[rowStart:rowEnd:rowStep,
                      colStart:colEnd:colStep, zStart:zEnd:zStep]

    assert zSliceDataNC.sizeZ() == zSliceData.shape[-1]
    for z in range(zSliceDataNC.sizeZ()):
        assert np.array_equal(
            zSliceDataNC[z].getNumpyArray(), zSliceData[:, :, z])
