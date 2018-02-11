import numpy as np
from termcolor import colored
import sys
sys.path.append(r'../build/x64/Release')
import NumC

####################################################################################
def doTest():
    print(colored('Testing Constants Module', 'magenta'))

    NUM_DECIMALS_ROUND = 10

    print(colored('Testing e', 'cyan'))
    if round(NumC.e, NUM_DECIMALS_ROUND) == round(np.e, 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing pi', 'cyan'))
    if round(NumC.pi, NUM_DECIMALS_ROUND) == round(np.pi, 10):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing nan', 'cyan'))
    if np.isnan(NumC.nan):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing version', 'cyan'))
    if NumC.version == '0.1':
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

####################################################################################
if __name__ == '__main__':
    doTest()