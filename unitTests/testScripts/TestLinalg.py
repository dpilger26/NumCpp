import numpy as np
from termcolor import colored
import sys
sys.path.append(r'../build/x64/Release')
import NumC

####################################################################################
def doTest():
    print(colored('Testing Linalg Module', 'magenta'))

    print(colored('Testing det: 2x2', 'cyan'))
    order = 2
    shape = NumC.Shape(order)
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if round(NumC.det(cArray)) == round(np.linalg.det(data).item()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing det: 3x3', 'cyan'))
    order = 3
    shape = NumC.Shape(order)
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if round(NumC.det(cArray)) == round(np.linalg.det(data).item()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing det: NxN', 'cyan'))
    order = np.random.randint(4, 9, [1,]).item()
    shape = NumC.Shape(order)
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if round(NumC.det(cArray)) == round(np.linalg.det(data).item()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing hat', 'cyan'))
    shape = NumC.Shape(1, 3)
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).flatten()
    cArray.setArray(data)
    if np.array_equal(NumC.hat(cArray), hat(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing inv', 'cyan'))
    order = np.random.randint(5, 50, [1,]).item()
    shape = NumC.Shape(order)
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(np.round(NumC.inv(cArray).getNumpyArray(), 9), np.round(np.linalg.inv(data), 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing lstsq', 'cyan'))
    shapeInput = np.random.randint(5, 50, [2,])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    aArray = NumC.NdArray(shape)
    bArray = NumC.NdArray(1, shape.rows)
    aData = np.random.randint(1, 100, [shape.rows, shape.cols])
    bData = np.random.randint(1, 100, [shape.rows,])
    aArray.setArray(aData)
    bArray.setArray(bData)
    x = NumC.lstsq(aArray, bArray, 1e-12).getNumpyArray().flatten()
    if np.array_equal(np.round(x, 9), np.round(np.linalg.lstsq(aData, bData)[0], 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing svd', 'cyan'))
    shapeInput = np.random.randint(5, 50, [2,])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    uArray = NumC.NdArray()
    sArray = NumC.NdArray()
    vArray = NumC.NdArray()
    NumC.svd(cArray, uArray, sArray, vArray)
    data2 = np.dot(uArray.getNumpyArray(), np.dot(sArray.getNumpyArray(), vArray.getNumpyArray().T))
    if np.array_equal(np.round(data, 9), np.round(data2, 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

####################################################################################
def hat(inVec):
    mat = np.zeros([3,3])
    mat[0, 1] = -inVec[2]
    mat[0, 2] = inVec[1]
    mat[1, 0] = inVec[2]
    mat[1, 2] = -inVec[0]
    mat[2, 0] = -inVec[1]
    mat[2, 1] = inVec[0]

    return mat

####################################################################################
if __name__ == '__main__':
    doTest()