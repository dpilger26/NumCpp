import numpy as np
from termcolor import colored
import sys
sys.path.append(r'../build/x64/Release')
import NumC

####################################################################################
def doTest():
    print(colored('Testing Shape Class', 'magenta'))

    print(colored('Testing Default Constructor', 'cyan'))
    shape = NumC.Shape()
    if shape.rows == 0 and shape.cols == 0:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Square Constructor', 'cyan'))
    shapeInput = np.random.randint(0, 100, [1,]).item()
    shape = NumC.Shape(shapeInput)
    if shape.rows == shapeInput and shape.cols == shapeInput:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Rectangle Constructor', 'cyan'))
    shapeInput = np.random.randint(0, 100, [2,])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    if shape.rows == shapeInput[0] and shape.cols == shapeInput[1]:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Copy Constructor', 'cyan'))
    shape2 = NumC.Shape(shape)
    if shape2.rows == shape.rows and shape2.cols == shape.cols:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Member Setting', 'cyan'))
    shape = NumC.Shape()
    shapeInput = np.random.randint(0, 100, [2, ])
    shape.rows = shapeInput[0].item()
    shape.cols = shapeInput[1].item()
    if shape.rows == shapeInput[0] and shape.cols == shapeInput[1]:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Print', 'cyan'))
    shape.print()

####################################################################################
if __name__ == '__main__':
    doTest()