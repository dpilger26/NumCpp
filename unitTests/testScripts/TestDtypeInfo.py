import numpy as np
from termcolor import colored
import sys
sys.path.append(r'../build/x64/Release')
import NumCpp

####################################################################################
def doTest():
    print(colored('Testing DtypeInfo Class', 'magenta'))

    print(colored('Testing bits', 'cyan'))
    if NumCpp.DtypeIntoUint32.bits() == 32:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing epsilon', 'cyan'))
    if NumCpp.DtypeIntoUint32.epsilon() == 0:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing isInteger', 'cyan'))
    if NumCpp.DtypeIntoUint32.isInteger():
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing max', 'cyan'))
    if NumCpp.DtypeIntoUint32.max() == np.iinfo(np.uint32).max:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing min', 'cyan'))
    if NumCpp.DtypeIntoUint32.min() == np.iinfo(np.uint32).min:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

####################################################################################
if __name__ == '__main__':
    doTest()
