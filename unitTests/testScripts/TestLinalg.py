import numpy as np
from termcolor import colored
import sys
sys.path.append(r'../build/x64/Release')
import NumC

####################################################################################
def doTest():
    print(colored('Testing Linalg Module', 'magenta'))

    print(colored('Testing det: 2x2', 'cyan'))
    shape = NumC.Shape(2, 2)
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    if round(NumC.det(cArray)) == round(np.linalg.det(data).item()):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing det: 3x3', 'cyan'))
    shape = NumC.Shape(3, 3)
    cArray = NumC.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
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