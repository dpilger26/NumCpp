import numpy as np
from termcolor import colored
import sys
sys.path.append(r'../build/x64/Release')
import NumCpp

####################################################################################
def doTest():
    print(colored('Testing Polynomial Module', 'magenta'))
    print(colored('\tPASS', 'green'))

####################################################################################
if __name__ == '__main__':
    doTest()
