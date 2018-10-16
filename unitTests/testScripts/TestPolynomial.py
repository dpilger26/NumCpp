import numpy as np
from termcolor import colored
import sys
if sys.platform == 'linux':
    sys.path.append(r'../src/cmake-build-release')
    import libNumCpp as NumCpp
else:
    sys.path.append(r'../build/x64/Release')
    import NumCpp

####################################################################################
def doTest():
    print(colored('Testing Polynomial Module', 'magenta'))
    print(colored('Testing Constructor', 'cyan'))
    numCoefficients = np.random.randint(3, 10, [1,]).item()
    coefficients = np.random.randint(-20, 20, [numCoefficients,])
    coefficientsC = NumCpp.NdArray(1, numCoefficients)
    coefficientsC.setArray(coefficients)
    polyC = NumCpp.Poly1d(coefficientsC, False)
    if np.array_equal(polyC.coefficients().getNumpyArray().flatten(), coefficients):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Constructor Roots', 'cyan'))
    numRoots = np.random.randint(3, 10, [1,]).item()
    roots = np.random.randint(-20, 20, [numRoots,])
    rootsC = NumCpp.NdArray(1, numRoots)
    rootsC.setArray(roots)
    poly = np.poly1d(roots, True)
    polyC = NumCpp.Poly1d(rootsC, True)
    if np.array_equal(np.fliplr(polyC.coefficients().getNumpyArray()).flatten(), poly.coefficients):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing order', 'cyan'))
    if polyC.order() == roots.size:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator()', 'cyan'))
    value = np.random.randint(-20, 20, [1,]).item()
    if polyC[value] == poly(value):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing addition', 'cyan'))
    numCoefficients = np.random.randint(3, 10, [1,]).item()
    coefficients = np.random.randint(-20, 20, [numCoefficients,])
    coefficientsC = NumCpp.NdArray(1, numCoefficients)
    coefficientsC.setArray(coefficients)
    polyC2 = NumCpp.Poly1d(coefficientsC, False)
    poly2 = np.poly1d(coefficients)
    if np.array_equal(np.fliplr((polyC + polyC2).coefficients().getNumpyArray()).flatten(), (poly + poly2).coefficients):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing subtraction', 'cyan'))
    if np.array_equal((polyC - polyC2).coefficients().getNumpyArray().flatten(), (poly - poly2).coefficients):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing multiplication', 'cyan'))
    if np.array_equal(np.fliplr((polyC * polyC2).coefficients().getNumpyArray()).flatten(), (poly * poly2).coefficients):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing power', 'cyan'))
    exponent = np.random.randint(1, 10, [1,]).item()
    if np.array_equal(np.fliplr((polyC ** exponent).coefficients().getNumpyArray()).flatten(), (poly ** exponent).coefficients):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing print', 'cyan'))
    polyC.print()

####################################################################################
if __name__ == '__main__':
    doTest()
