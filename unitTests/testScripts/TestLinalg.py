import numpy as np
from termcolor import colored
import sys
if sys.platform == 'linux':
    sys.path.append(r'../src/cmake-build-release')
    import libNumCpp as NumCpp
else:
    sys.path.append(r'../build/x64/Release')
    import NumCpp

####################################################################################
def doTest():
    print(colored('Testing Linalg Module', 'magenta'))

    print(colored('Testing det: 2x2', 'cyan'))
    order = 2
    shape = NumCpp.Shape(order)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if round(NumCpp.det(cArray)) == round(np.linalg.det(data).item()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing det: 3x3', 'cyan'))
    order = 3
    shape = NumCpp.Shape(order)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if round(NumCpp.det(cArray)) == round(np.linalg.det(data).item()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing det: NxN', 'cyan'))
    order = np.random.randint(4, 8, [1,]).item()
    shape = NumCpp.Shape(order)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    if round(NumCpp.det(cArray)) == round(np.linalg.det(data).item()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing hat', 'cyan'))
    shape = NumCpp.Shape(1, 3)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).flatten()
    cArray.setArray(data)
    if np.array_equal(NumCpp.hat(cArray), hat(data)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing inv', 'cyan'))
    order = np.random.randint(5, 50, [1,]).item()
    shape = NumCpp.Shape(order)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(np.round(NumCpp.inv(cArray).getNumpyArray(), 9), np.round(np.linalg.inv(data), 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing matrix_power: power = 0', 'cyan'))
    order = np.random.randint(5, 50, [1, ]).item()
    shape = NumCpp.Shape(order)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumCpp.matrix_power(cArray, 0).getNumpyArray(), np.linalg.matrix_power(data, 0)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing matrix_power: power = 1', 'cyan'))
    order = np.random.randint(5, 50, [1, ]).item()
    shape = NumCpp.Shape(order)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(NumCpp.matrix_power(cArray, 1).getNumpyArray(), np.linalg.matrix_power(data, 1)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing matrix_power: power = -1', 'cyan'))
    order = np.random.randint(5, 50, [1, ]).item()
    shape = NumCpp.Shape(order)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if np.array_equal(np.round(NumCpp.matrix_power(cArray, -1).getNumpyArray(), 8), np.round(np.linalg.matrix_power(data, -1), 8)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing matrix_power: power > 1', 'cyan'))
    order = np.random.randint(5, 50, [1, ]).item()
    shape = NumCpp.Shape(order)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 5, [shape.rows, shape.cols]).astype(np.double)
    cArray.setArray(data)
    power = np.random.randint(2, 9, [1,]).item()
    if np.array_equal(NumCpp.matrix_power(cArray, power).getNumpyArray(), np.linalg.matrix_power(data, power)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing matrix_power: power < -1', 'cyan'))
    order = np.random.randint(5, 50, [1,]).item()
    shape = NumCpp.Shape(order)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    power = np.random.randint(2, 9, [1, ]).item() * -1
    if np.array_equal(np.round(NumCpp.matrix_power(cArray, power).getNumpyArray(), 9), np.round(np.linalg.matrix_power(data, power), 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing multi_dot', 'cyan'))
    shapeInput = np.random.randint(5, 50, [2, ])
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(shape1.cols, np.random.randint(5, 50, [1, ]).item())
    shape3 = NumCpp.Shape(shape2.cols, np.random.randint(5, 50, [1, ]).item())
    shape4 = NumCpp.Shape(shape3.cols, np.random.randint(5, 50, [1, ]).item())
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    cArray3 = NumCpp.NdArray(shape3)
    cArray4 = NumCpp.NdArray(shape4)
    data1 = np.random.randint(1, 10, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 10, [shape2.rows, shape2.cols])
    data3 = np.random.randint(1, 10, [shape3.rows, shape3.cols])
    data4 = np.random.randint(1, 10, [shape4.rows, shape4.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    if np.array_equal(np.round(NumCpp.multi_dot(cArray1, cArray2, cArray3, cArray4), 9), np.round(np.linalg.multi_dot([data1, data2, data3, data4]), 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing lstsq', 'cyan'))
    shapeInput = np.random.randint(5, 50, [2,])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    aArray = NumCpp.NdArray(shape)
    bArray = NumCpp.NdArray(1, shape.rows)
    aData = np.random.randint(1, 100, [shape.rows, shape.cols])
    bData = np.random.randint(1, 100, [shape.rows,])
    aArray.setArray(aData)
    bArray.setArray(bData)
    x = NumCpp.lstsq(aArray, bArray, 1e-12).getNumpyArray().flatten()
    if np.array_equal(np.round(x, 9), np.round(np.linalg.lstsq(aData, bData, rcond=None)[0], 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing svd', 'cyan'))
    shapeInput = np.random.randint(5, 50, [2,])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    uArray = NumCpp.NdArray()
    sArray = NumCpp.NdArray()
    vArray = NumCpp.NdArray()
    NumCpp.svd(cArray, uArray, sArray, vArray)
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
