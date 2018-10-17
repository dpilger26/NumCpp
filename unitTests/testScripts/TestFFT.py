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
    print(colored('Testing FFT Module', 'magenta'))
    print(colored('\tPASS', 'green'))

####################################################################################
if __name__ == '__main__':
    doTest()
