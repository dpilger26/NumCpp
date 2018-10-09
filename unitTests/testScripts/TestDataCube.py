import numpy as np
from termcolor import colored
import os
import sys
if sys.platform == 'linux':
    sys.path.append(r'../src/cmake-build-release')
    import libNumCpp as NumCpp
else:
    sys.path.append(r'../build/x64/Release')
    import NumCpp

####################################################################################
def doTest():
    print(colored('Testing DataCube Module', 'magenta'))

    print(colored('Testing Default Constructor', 'cyan'))
    dataCube = NumCpp.DataCube()
    if dataCube.isempty():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Size Constructor', 'cyan'))
    size = np.random.randint(10, 20, [1,]).item()
    dataCube = NumCpp.DataCube(size)
    if dataCube.size() == size:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing push_front/push_back', 'cyan'))
    shape = np.random.randint(10, 100, [3,])
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
    if not dataCube.isempty() and dataCube.size() == shape[-1]:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing shape', 'cyan'))
    if dataCube.shape() == cShape:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing back', 'cyan'))
    if np.array_equal(dataCube.back().getNumpyArray(), data[:, :, frameOrder[-1]]):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing front', 'cyan'))
    if np.array_equal(dataCube.front().getNumpyArray(), data[:, :, frameOrder[0]]):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing [] operator', 'cyan'))
    allPass = True
    for idx, frame in enumerate(frameOrder):
        if not np.array_equal(dataCube[idx].getNumpyArray(), data[:, :, frame]):
            allPass = False
            break
    if allPass:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing at', 'cyan'))
    for idx, frame in enumerate(frameOrder):
        if not np.array_equal(dataCube.at(idx).getNumpyArray(), data[:, :, frame]):
            allPass = False
            break
    if allPass:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing dump', 'cyan'))
    tempDir = r'C:\Temp'
    if not os.path.exists(tempDir):
        os.mkdir(tempDir)
    tempFile = os.path.join(tempDir, 'DataCube.bin')
    dataCube.dump(tempFile)
    if os.path.exists(tempFile):
        filesize = os.path.getsize(tempFile)
        if filesize == data.size * 8:
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))
    else:
        print(colored('\tFAIL', 'red'))
    os.remove(tempFile)

    print(colored('Testing pop_front/pop_back', 'cyan'))
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

    if dataCube.isempty() and allPass:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))


####################################################################################
if __name__ == '__main__':
    doTest()
