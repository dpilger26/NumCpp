import numpy as np
import scipy.special as sp
from termcolor import colored
import os
import sys
if sys.platform == 'linux':
    sys.path.append(r'../lib')
else:
    sys.path.append(os.path.abspath('../build/x64/Release'))
import NumCpp


####################################################################################
def doTest():
    print(colored('Testing Roots Module', 'magenta'))

    print(colored('Testing bisection', 'cyan'))
    root = np.random.randint(-50, 50, [1,]).item()
    roots = np.array([root, root + np.random.randint(5, 50, [1,]).item()])
    largestRoot = roots.max().item()
    rootsC = NumCpp.NdArray(1, roots.size)
    rootsC.setArray(roots)
    polyC = NumCpp.Poly1d(rootsC, True)
    rootC = int(np.round(NumCpp.bisection_roots(polyC, largestRoot - 1, largestRoot + 1)))
    if rootC == largestRoot:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing brent', 'cyan'))
    root = np.random.randint(-50, 50, [1,]).item()
    roots = np.array([root, root + np.random.randint(5, 50, [1,]).item()])
    largestRoot = roots.max().item()
    rootsC = NumCpp.NdArray(1, roots.size)
    rootsC.setArray(roots)
    polyC = NumCpp.Poly1d(rootsC, True)
    rootC = int(np.round(NumCpp.brent_roots(polyC, largestRoot - 1, largestRoot + 1)))
    if rootC == largestRoot:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing dekker', 'cyan'))
    root = np.random.randint(-50, 50, [1,]).item()
    roots = np.array([root, root + np.random.randint(5, 50, [1,]).item()])
    largestRoot = roots.max().item()
    rootsC = NumCpp.NdArray(1, roots.size)
    rootsC.setArray(roots)
    polyC = NumCpp.Poly1d(rootsC, True)
    rootC = int(np.round(NumCpp.dekker_roots(polyC, largestRoot - 1, largestRoot + 1)))
    if rootC == largestRoot:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing newton', 'cyan'))
    root = np.random.randint(-50, 50, [1,]).item()
    roots = np.array([root, root + np.random.randint(5, 50, [1,]).item()])
    largestRoot = roots.max().item()
    rootsC = NumCpp.NdArray(1, roots.size)
    rootsC.setArray(roots)
    polyC = NumCpp.Poly1d(rootsC, True)
    rootC = int(np.round(NumCpp.newton_roots(polyC, largestRoot)))
    if rootC == largestRoot:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing secant', 'cyan'))
    root = np.random.randint(-50, 50, [1,]).item()
    roots = np.array([root, root + np.random.randint(5, 50, [1,]).item()])
    largestRoot = roots.max().item()
    rootsC = NumCpp.NdArray(1, roots.size)
    rootsC.setArray(roots)
    polyC = NumCpp.Poly1d(rootsC, True)
    rootC = int(np.round(NumCpp.secant_roots(polyC, largestRoot - 1, largestRoot + 1)))
    if rootC == largestRoot:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))


####################################################################################
if __name__ == '__main__':
    doTest()
