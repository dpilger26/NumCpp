import numpy as np
from termcolor import colored
import sys
if sys.platform == 'linux':
    sys.path.append(r'../src/build')
    import libNumCpp as NumCpp
else:
    sys.path.append(r'../build/x64/Release')
    import NumCpp

####################################################################################
def doTest():
    print(colored('Testing Shape Class', 'magenta'))

    print(colored('Testing Default Constructor', 'cyan'))
    shape = NumCpp.Shape()
    if shape.rows == 0 and shape.cols == 0:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Square Constructor', 'cyan'))
    shapeInput = np.random.randint(0, 100, [1,]).item()
    shape = NumCpp.Shape(shapeInput)
    if shape.rows == shapeInput and shape.cols == shapeInput:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Rectangle Constructor', 'cyan'))
    shapeInput = np.random.randint(0, 100, [2,])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    if shape.rows == shapeInput[0] and shape.cols == shapeInput[1]:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Copy Constructor', 'cyan'))
    shape2 = NumCpp.Shape(shape)
    if shape2.rows == shape.rows and shape2.cols == shape.cols:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Member Setting', 'cyan'))
    shape = NumCpp.Shape()
    shapeInput = np.random.randint(0, 100, [2, ])
    shape.rows = shapeInput[0].item()
    shape.cols = shapeInput[1].item()
    if shape.rows == shapeInput[0] and shape.cols == shapeInput[1]:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Print', 'cyan'))
    shape.print()
    print(colored('\tPASS', 'green'))

####################################################################################
if __name__ == '__main__':
    doTest()
