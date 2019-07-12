import numpy as np
import scipy.special as special
from termcolor import colored
import sys
if sys.platform == 'linux':
    sys.path.append(r'../lib')
else:
    sys.path.append(r'../build/x64/Release')
import NumCpp


####################################################################################
def doTest():
    print(colored('Testing Special Module', 'magenta'))

    print(colored('Testing erf scaler', 'cyan'))
    value = np.random.randn(1).item()
    if np.round(NumCpp.erfScaler(value), 9) == np.round(special.erf(value), 9):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing erf array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumCpp.erfArray(cArray), 9), np.round(special.erf(data), 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing erfc scaler', 'cyan'))
    value = np.random.randn(1).item()
    if np.round(NumCpp.erfcScaler(value), 9) == np.round(special.erfc(value), 9):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing erfc array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randn(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(np.round(NumCpp.erfcArray(cArray), 9), np.round(special.erfc(data), 9)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))


####################################################################################
if __name__ == '__main__':
    doTest()
