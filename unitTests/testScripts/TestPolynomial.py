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
    testPoly1D()
    testFunctions()


####################################################################################
def testPoly1D():
    print(colored('Testing Polynomial Module', 'magenta'))

    print(colored('Testing Poly1d class', 'magenta'))

    print(colored('Testing Constructor', 'cyan'))
    numCoefficients = np.random.randint(3, 10, [1, ]).item()
    coefficients = np.random.randint(-20, 20, [numCoefficients, ])
    coefficientsC = NumCpp.NdArray(1, numCoefficients)
    coefficientsC.setArray(coefficients)
    polyC = NumCpp.Poly1d(coefficientsC, False)
    if np.array_equal(polyC.coefficients().getNumpyArray().flatten(), coefficients):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing Constructor Roots', 'cyan'))
    numRoots = np.random.randint(3, 10, [1, ]).item()
    roots = np.random.randint(-20, 20, [numRoots, ])
    rootsC = NumCpp.NdArray(1, numRoots)
    rootsC.setArray(roots)
    poly = np.poly1d(roots, True)
    polyC = NumCpp.Poly1d(rootsC, True)
    if np.array_equal(np.fliplr(polyC.coefficients().getNumpyArray()).flatten().astype(np.int), poly.coefficients):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing area', 'cyan'))
    bounds = np.random.rand(2) * 100 - 50
    bounds = np.sort(bounds)
    polyIntegral = poly.integ()
    if np.round(polyC.area(*bounds), 3) == np.round(polyIntegral(bounds[1]) - polyIntegral(bounds[0]), 3):
        print(colored('\tPASS', 'green'))
    else:
        print(np.round(polyC.area(*bounds), 3))
        print(np.round(polyIntegral(bounds[1]) - polyIntegral(bounds[0]), 3))
        print(colored('\tFAIL', 'red'))

    print(colored('Testing deriv', 'cyan'))
    if np.array_equal(polyC.deriv().coefficients().getNumpyArray().flatten(), np.flipud(poly.deriv().coefficients)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing integ', 'cyan'))
    if np.array_equal(polyC.integ().coefficients().getNumpyArray().flatten(), np.flipud(poly.integ().coefficients)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing order', 'cyan'))
    if polyC.order() == roots.size:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing operator()', 'cyan'))
    value = np.random.randint(-20, 20, [1, ]).item()
    if polyC[value] == poly(value):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing addition', 'cyan'))
    numCoefficients = np.random.randint(3, 10, [1, ]).item()
    coefficients = np.random.randint(-20, 20, [numCoefficients, ])
    coefficientsC = NumCpp.NdArray(1, numCoefficients)
    coefficientsC.setArray(coefficients)
    polyC2 = NumCpp.Poly1d(coefficientsC, False)
    poly2 = np.poly1d(np.flip(coefficients))
    if np.array_equal(np.fliplr((polyC + polyC2).coefficients().getNumpyArray()).flatten(),
                      (poly + poly2).coefficients):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing subtraction', 'cyan'))
    if np.array_equal(np.fliplr((polyC - polyC2).coefficients().getNumpyArray()).flatten(),
                      (poly - poly2).coefficients):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing multiplication', 'cyan'))
    if np.array_equal(np.fliplr((polyC * polyC2).coefficients().getNumpyArray()).flatten(),
                      (poly * poly2).coefficients):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing power', 'cyan'))
    exponent = np.random.randint(0, 5, [1, ]).item()
    if np.array_equal(np.fliplr((polyC2 ** exponent).coefficients().getNumpyArray()).flatten(),
                      (poly2 ** exponent).coefficients):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing print', 'cyan'))
    polyC.print()


####################################################################################
def testFunctions():
    print(colored('Testing Polynomial functions', 'magenta'))
    ORDER_MAX = 5
    DECIMALS_ROUND = 7

    print(colored('Testing chebyshev_t_Scaler', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        x = np.random.rand(1).item()
        valuePy = sp.eval_chebyt(order, x)
        valueCpp = NumCpp.chebyshev_t_Scaler(order, x)
        if np.round(valuePy, DECIMALS_ROUND) != np.round(valueCpp, DECIMALS_ROUND):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))
        print(f'valuePy = {valuePy}, valueCpp = {valueCpp}')

    print(colored('Testing chebyshev_t_Array', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        shapeInput = np.random.randint(10, 100, [2, ], dtype=np.uint32)
        shape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(shape)
        x = np.random.rand(*shapeInput)
        cArray.setArray(x)
        valuePy = sp.eval_chebyt(order, x)
        valueCpp = NumCpp.chebyshev_t_Array(order, cArray)
        if not np.array_equal(np.round(valuePy, DECIMALS_ROUND), np.round(valueCpp, DECIMALS_ROUND)):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing chebyshev_u_Scaler', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        x = np.random.rand(1).item()
        valuePy = sp.eval_chebyu(order, x)
        valueCpp = NumCpp.chebyshev_u_Scaler(order, x)
        if np.round(valuePy, DECIMALS_ROUND) != np.round(valueCpp, DECIMALS_ROUND):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))
        print(f'valuePy = {valuePy}, valueCpp = {valueCpp}')

    print(colored('Testing chebyshev_u_Array', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        shapeInput = np.random.randint(10, 100, [2, ], dtype=np.uint32)
        shape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(shape)
        x = np.random.rand(*shapeInput)
        cArray.setArray(x)
        valuePy = sp.eval_chebyu(order, x)
        valueCpp = NumCpp.chebyshev_u_Array(order, cArray)
        if not np.array_equal(np.round(valuePy, DECIMALS_ROUND), np.round(valueCpp, DECIMALS_ROUND)):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing hermite_Scaler', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        x = np.random.rand(1).item()
        valuePy = sp.eval_hermite(order, x)
        valueCpp = NumCpp.hermite_Scaler(order, x)
        if np.round(valuePy, DECIMALS_ROUND) != np.round(valueCpp, DECIMALS_ROUND):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))
        print(f'valuePy = {valuePy}, valueCpp = {valueCpp}')

    print(colored('Testing hermite_Array', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        shapeInput = np.random.randint(10, 100, [2, ], dtype=np.uint32)
        shape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(shape)
        x = np.random.rand(*shapeInput)
        cArray.setArray(x)
        valuePy = sp.eval_hermite(order, x)
        valueCpp = NumCpp.hermite_Array(order, cArray)
        if not np.array_equal(np.round(valuePy, DECIMALS_ROUND), np.round(valueCpp, DECIMALS_ROUND)):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing laguerre_Scaler1', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        x = np.random.rand(1).item()
        valuePy = sp.eval_laguerre(order, x)
        valueCpp = NumCpp.laguerre_Scaler1(order, x)
        if np.round(valuePy, DECIMALS_ROUND) != np.round(valueCpp, DECIMALS_ROUND):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))
        print(f'valuePy = {valuePy}, valueCpp = {valueCpp}')

    print(colored('Testing laguerre_Array1', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        shapeInput = np.random.randint(10, 100, [2, ], dtype=np.uint32)
        shape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(shape)
        x = np.random.rand(*shapeInput)
        cArray.setArray(x)
        valuePy = sp.eval_laguerre(order, x)
        valueCpp = NumCpp.laguerre_Array1(order, cArray)
        if not np.array_equal(np.round(valuePy, DECIMALS_ROUND), np.round(valueCpp, DECIMALS_ROUND)):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing laguerre_Scaler2', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        degree = np.random.randint(0, 10, [1, ]).item()
        x = np.random.rand(1).item()
        valuePy = sp.eval_genlaguerre(degree, order, x)
        valueCpp = NumCpp.laguerre_Scaler2(order, degree, x)
        if np.round(valuePy, DECIMALS_ROUND) != np.round(valueCpp, DECIMALS_ROUND):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))
        print(f'valuePy = {valuePy}, valueCpp = {valueCpp}')

    print(colored('Testing laguerre_Array2', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        degree = np.random.randint(0, 10, [1, ]).item()
        shapeInput = np.random.randint(10, 100, [2, ], dtype=np.uint32)
        shape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(shape)
        x = np.random.rand(*shapeInput)
        cArray.setArray(x)
        valuePy = sp.eval_genlaguerre(degree, order, x)
        valueCpp = NumCpp.laguerre_Array2(order, degree, cArray)
        if not np.array_equal(np.round(valuePy, DECIMALS_ROUND), np.round(valueCpp, DECIMALS_ROUND)):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing legendre_p_Scaler1', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        x = np.random.rand(1).item()
        valuePy = sp.eval_legendre(order, x)
        valueCpp = NumCpp.legendre_p_Scaler1(order, x)
        if np.round(valuePy, DECIMALS_ROUND) != np.round(valueCpp, DECIMALS_ROUND):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))
        print(f'valuePy = {valuePy}, valueCpp = {valueCpp}')

    print(colored('Testing legendre_p_Array1', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        shapeInput = np.random.randint(10, 100, [2, ], dtype=np.uint32)
        shape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(shape)
        x = np.random.rand(*shapeInput)
        cArray.setArray(x)
        valuePy = sp.eval_legendre(order, x)
        valueCpp = NumCpp.legendre_p_Array1(order, cArray)
        if not np.array_equal(np.round(valuePy, DECIMALS_ROUND), np.round(valueCpp, DECIMALS_ROUND)):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing legendre_p_Scaler2', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        x = np.random.rand(1).item()
        degree = np.random.randint(order, ORDER_MAX)
        valuePy = sp.lpmn(order, degree, x)[0][order, degree]
        valueCpp = NumCpp.legendre_p_Scaler2(order, degree, x)
        if np.round(valuePy, DECIMALS_ROUND) != np.round(valueCpp, DECIMALS_ROUND):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))
        print(f'valuePy = {valuePy}, valueCpp = {valueCpp}')

    print(colored('Testing legendre_q_Scaler', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        x = np.random.rand(1).item()
        valuePy = sp.lqn(order, x)[0][order]
        valueCpp = NumCpp.legendre_q_Scaler(order, x)
        if np.round(valuePy, DECIMALS_ROUND) != np.round(valueCpp, DECIMALS_ROUND):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))
        print(f'valuePy = {valuePy}, valueCpp = {valueCpp}')

    print(colored('Testing spherical_harmonic', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        degree = np.random.randint(order, ORDER_MAX)
        theta = np.random.rand(1).item() * np.pi * 2
        phi = np.random.rand(1).item() * np.pi
        valuePy = sp.sph_harm(order, degree, theta, phi)
        valueCpp = NumCpp.spherical_harmonic(order, degree, theta, phi)
        if (np.round(valuePy.real, DECIMALS_ROUND) != np.round(valueCpp[0], DECIMALS_ROUND) or
                np.round(valuePy.imag, DECIMALS_ROUND) != np.round(valueCpp[1], DECIMALS_ROUND)):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))
        print(f'valuePy = {valuePy}, valueCpp = {valueCpp}')

    print(colored('Testing spherical_harmonic_r', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        degree = np.random.randint(order, ORDER_MAX)
        theta = np.random.rand(1).item() * np.pi * 2
        phi = np.random.rand(1).item() * np.pi
        valuePy = sp.sph_harm(order, degree, theta, phi)
        valueCpp = NumCpp.spherical_harmonic_r(order, degree, theta, phi)
        if np.round(valuePy.real, DECIMALS_ROUND) != np.round(valueCpp, DECIMALS_ROUND):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))
        print(f'valuePy = {valuePy}, valueCpp = {valueCpp}')

    print(colored('Testing spherical_harmonic_i', 'cyan'))
    allTrue = True
    for order in range(ORDER_MAX):
        degree = np.random.randint(order, ORDER_MAX)
        theta = np.random.rand(1).item() * np.pi * 2
        phi = np.random.rand(1).item() * np.pi
        valuePy = sp.sph_harm(order, degree, theta, phi)
        valueCpp = NumCpp.spherical_harmonic_i(order, degree, theta, phi)
        if np.round(valuePy.imag, DECIMALS_ROUND) != np.round(valueCpp, DECIMALS_ROUND):
            allTrue = False

    if allTrue:
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))
        print(f'valuePy = {valuePy}, valueCpp = {valueCpp}')


####################################################################################
if __name__ == '__main__':
    doTest()
