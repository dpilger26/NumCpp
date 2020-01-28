import numpy as np
import scipy.special as sp
from termcolor import colored
import sys
if sys.platform == 'linux':
    sys.path.append(r'../lib')
else:
    sys.path.append(r'../build/x64/Release')
import NumCpp


####################################################################################
NUM_DECIMALS_ROUND = 1


####################################################################################
def doTest():
    print(colored('Testing Integration Module', 'magenta'))

    print(colored('Testing gauss_legendre', 'cyan'))
    numCoefficients = np.random.randint(2, 5, [1, ]).item()
    coefficients = np.random.randint(-20, 20, [numCoefficients, ])
    coefficientsC = NumCpp.NdArray(1, numCoefficients)
    coefficientsC.setArray(coefficients)
    poly = np.poly1d(np.flipud(coefficients), False)
    polyIntegral = poly.integ()
    polyC = NumCpp.Poly1d(coefficientsC, False)
    a, b = np.sort(np.random.rand(2) * 100 - 50)
    area = np.round(polyIntegral(b) - polyIntegral(a), NUM_DECIMALS_ROUND)
    areaC = np.round(NumCpp.integrate_gauss_legendre(polyC, a, b), NUM_DECIMALS_ROUND)
    if area == areaC:
        print(colored('\tPASS', 'green'))
    else:
        print(area)
        print(areaC)
        print(colored('\tFAIL', 'red'))

    print(colored('Testing romberg', 'cyan'))
    PERCENT_LEEWAY = 0.1
    numCoefficients = np.random.randint(2, 5, [1, ]).item()
    coefficients = np.random.randint(-20, 20, [numCoefficients, ])
    coefficientsC = NumCpp.NdArray(1, numCoefficients)
    coefficientsC.setArray(coefficients)
    poly = np.poly1d(np.flipud(coefficients), False)
    polyIntegral = poly.integ()
    polyC = NumCpp.Poly1d(coefficientsC, False)
    a, b = np.sort(np.random.rand(2) * 100 - 50)
    area = np.round(polyIntegral(b) - polyIntegral(a), NUM_DECIMALS_ROUND)
    areaC = np.round(NumCpp.integrate_romberg(polyC, a, b), NUM_DECIMALS_ROUND)
    # romberg is much less acurate so let's give it some leeway
    areaLow, areaHigh = np.sort([area * (1 - PERCENT_LEEWAY), area * (1 + PERCENT_LEEWAY)])
    if areaLow < areaC < areaHigh:
        print(colored('\tPASS', 'green'))
    else:
        print(area)
        print(areaC)
        print(colored('\tFAIL', 'red'))

    print(colored('Testing simpson', 'cyan'))
    numCoefficients = np.random.randint(2, 5, [1, ]).item()
    coefficients = np.random.randint(-20, 20, [numCoefficients, ])
    coefficientsC = NumCpp.NdArray(1, numCoefficients)
    coefficientsC.setArray(coefficients)
    poly = np.poly1d(np.flipud(coefficients), False)
    polyIntegral = poly.integ()
    polyC = NumCpp.Poly1d(coefficientsC, False)
    a, b = np.sort(np.random.rand(2) * 100 - 50)
    area = np.round(polyIntegral(b) - polyIntegral(a), NUM_DECIMALS_ROUND)
    areaC = np.round(NumCpp.integrate_simpson(polyC, a, b), NUM_DECIMALS_ROUND)
    if area == areaC:
        print(colored('\tPASS', 'green'))
    else:
        print(area)
        print(areaC)
        print(colored('\tFAIL', 'red'))

    print(colored('Testing trapazoidal', 'cyan'))
    numCoefficients = np.random.randint(2, 5, [1, ]).item()
    coefficients = np.random.randint(-20, 20, [numCoefficients, ])
    coefficientsC = NumCpp.NdArray(1, numCoefficients)
    coefficientsC.setArray(coefficients)
    poly = np.poly1d(np.flipud(coefficients), False)
    polyIntegral = poly.integ()
    polyC = NumCpp.Poly1d(coefficientsC, False)
    a, b = np.sort(np.random.rand(2) * 100 - 50)
    area = np.round(polyIntegral(b) - polyIntegral(a), NUM_DECIMALS_ROUND)
    areaC = np.round(NumCpp.integrate_trapazoidal(polyC, a, b), NUM_DECIMALS_ROUND)
    if area == areaC:
        print(colored('\tPASS', 'green'))
    else:
        print(area)
        print(areaC)
        print(colored('\tFAIL', 'red'))


####################################################################################
if __name__ == '__main__':
    doTest()
