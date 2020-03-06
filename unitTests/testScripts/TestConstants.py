import numpy as np
from termcolor import colored
import sys
if sys.platform == 'linux':
    sys.path.append(r'../lib')
else:
    sys.path.append(r'../build/x64/Release')
import NumCpp

####################################################################################
def doTest():
    print(colored('Testing Constants Module', 'magenta'))

    NUM_DECIMALS_ROUND = 10

    print(colored('Testing c', 'cyan'))
    if round(NumCpp.c, NUM_DECIMALS_ROUND) == round(3e8, 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing e', 'cyan'))
    if round(NumCpp.e, NUM_DECIMALS_ROUND) == round(np.e, 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing inf', 'cyan'))
    if np.isinf(NumCpp.inf):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing pi', 'cyan'))
    if round(NumCpp.pi, NUM_DECIMALS_ROUND) == round(np.pi, 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing nan', 'cyan'))
    if np.isnan(NumCpp.nan):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing VERSION', 'cyan'))
    if NumCpp.VERSION == '1.3':
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

####################################################################################
if __name__ == '__main__':
    doTest()
