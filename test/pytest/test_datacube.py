import numpy as np
import os
import sys
sys.path.append(os.path.abspath(r'../lib'))
import NumCpp  # noqa E402

np.random.seed(666)


####################################################################################
def test_default_constructor():
    dataCube = NumCpp.DataCube()
    assert dataCube.isempty()


####################################################################################
def test_size_constructor():
    size = np.random.randint(10, 20, [1, ]).item()
    dataCube = NumCpp.DataCube(size)
    assert dataCube.size() == size


####################################################################################
def test_methods():
    shape = np.random.randint(10, 100, [3, ])
    cShape = NumCpp.Shape(shape[0].item(), shape[1].item())
    data = np.random.randint(0, 100, shape)
    dataCube = NumCpp.DataCube()
    frameOrder = list()
    for frame in range(shape[-1]):
        cArray = NumCpp.NdArray(cShape)
        cArray.setArray(data[:, :, frame])
        if frame % 2 == 0:
            dataCube.push_back(cArray)
            frameOrder.append(frame)
        else:
            dataCube.push_front(cArray)
            frameOrder = [frame] + frameOrder

    assert not dataCube.isempty()
    assert dataCube.size() == shape[-1]
    assert dataCube.shape() == cShape
    assert np.array_equal(dataCube.back().getNumpyArray(), data[:, :, frameOrder[-1]])
    assert np.array_equal(dataCube.front().getNumpyArray(), data[:, :, frameOrder[0]])

    allPass = True
    for idx, frame in enumerate(frameOrder):
        if not np.array_equal(dataCube[idx].getNumpyArray(), data[:, :, frame]):
            allPass = False
            break
    assert allPass

    for idx, frame in enumerate(frameOrder):
        if not np.array_equal(dataCube.at(idx).getNumpyArray(), data[:, :, frame]):
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

    sizeInitial = dataCube.size()
    sizeNow = sizeInitial
    allPass = True
    for idx in range(sizeInitial):
        if idx % 2 == 0:
            dataCube.pop_front()
        else:
            dataCube.pop_back()

        sizeNow -= 1

        if dataCube.size() != sizeNow:
            allPass = False

    assert dataCube.isempty()
    assert allPass
