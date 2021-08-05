import numpy as np
from numpy.polynomial.polynomial import Polynomial
import scipy.special as sp
import os
import sys
sys.path.append(os.path.abspath(r'../lib'))
import NumCppPy as NumCpp  # noqa E402


####################################################################################
ORDER_MAX = 5
DECIMALS_ROUND = 7


####################################################################################
def test_seed():
    np.random.seed(666)


####################################################################################
def test_poly1D_coefficents_constructor():
    numCoefficients = np.random.randint(3, 10, [1, ]).item()
    coefficients = np.random.randint(-20, 20, [numCoefficients, ])
    coefficientsC = NumCpp.NdArray(1, numCoefficients)
    coefficientsC.setArray(coefficients)
    polyC = NumCpp.Poly1d(coefficientsC, False)
    assert np.array_equal(polyC.coefficients().getNumpyArray().flatten(), coefficients)


####################################################################################
def test_poly1D_roots_constructor():
    numRoots = np.random.randint(3, 10, [1, ]).item()
    roots = np.random.randint(-20, 20, [numRoots, ])
    rootsC = NumCpp.NdArray(1, numRoots)
    rootsC.setArray(roots)
    poly = np.poly1d(roots, True)
    polyC = NumCpp.Poly1d(rootsC, True)
    assert np.array_equal(np.fliplr(polyC.coefficients().getNumpyArray()).flatten().astype(int), poly.coefficients)  # noqa


####################################################################################
def test_poly1D_integ_deriv_area_order():
    numRoots = np.random.randint(3, 10, [1, ]).item()
    roots = np.random.randint(-20, 20, [numRoots, ])
    rootsC = NumCpp.NdArray(1, numRoots)
    rootsC.setArray(roots)
    poly = np.poly1d(roots, True)
    polyC = NumCpp.Poly1d(rootsC, True)

    bounds = np.random.rand(2) * 100 - 50
    bounds = np.sort(bounds)
    polyIntegral = poly.integ()
    assert np.round(polyC.area(*bounds), 3) == np.round(polyIntegral(bounds[1]) - polyIntegral(bounds[0]), 3)
    assert np.array_equal(polyC.deriv().coefficients().getNumpyArray().flatten(), np.flipud(poly.deriv().coefficients))
    assert np.array_equal(polyC.integ().coefficients().getNumpyArray().flatten(), np.flipud(poly.integ().coefficients))
    assert polyC.order() == roots.size

    value = np.random.randint(-20, 20, [1, ]).item()
    assert polyC[value] == poly(value)


####################################################################################
def test_poly1D_fit():
    polyOrder = np.random.randint(2, 5)
    numMeasurements = np.random.randint(50, 100)
    xValues = np.random.rand(numMeasurements) * 100 - 50
    coefficients = np.random.rand(polyOrder + 1) * 5 - 10
    yValues = []
    for x in xValues:
        y = 0
        for order in range(polyOrder + 1):
            y += coefficients[order] * x ** order
        yValues.append(y + np.random.randn(1).item())
    yValues = np.array(yValues)
    yValues = yValues.reshape(yValues.size, 1)

    cX = NumCpp.NdArray(1, xValues.size)
    cY = NumCpp.NdArray(yValues.size, 1)
    cX.setArray(xValues)
    cY.setArray(yValues)

    poly = Polynomial.fit(xValues, yValues.flatten(), polyOrder).convert().coef  # noqa
    polyC = NumCpp.Poly1d.fit(cX, cY, polyOrder).coefficients().getNumpyArray().flatten()

    assert np.array_equal(np.round(poly, 5), np.round(polyC, 5))


####################################################################################
def test_poly1D_fit_weighted():
    polyOrder = np.random.randint(2, 5)
    numMeasurements = np.random.randint(50, 100)
    xValues = np.random.rand(numMeasurements) * 100 - 50
    coefficients = np.random.rand(polyOrder + 1) * 5 - 10
    yValues = []
    for x in xValues:
        y = 0
        for order in range(polyOrder + 1):
            y += coefficients[order] * x ** order
        yValues.append(y + np.random.randn(1).item())
    yValues = np.array(yValues)
    yValues = yValues.reshape(yValues.size, 1)
    weights = np.random.rand(numMeasurements)

    cX = NumCpp.NdArray(1, xValues.size)
    cY = NumCpp.NdArray(yValues.size, 1)
    cWeights = NumCpp.NdArray(1, xValues.size)
    cX.setArray(xValues)
    cY.setArray(yValues)
    cWeights.setArray(weights)

    poly = Polynomial.fit(xValues, yValues.flatten(), polyOrder, w=weights).convert().coef  # noqa
    polyC = NumCpp.Poly1d.fitWeighted(cX, cY, cWeights, polyOrder).coefficients().getNumpyArray().flatten()

    assert np.array_equal(np.round(poly, 1), np.round(polyC, 1))


####################################################################################
def test_poly1D_operators():
    numRoots = np.random.randint(3, 10, [1, ]).item()
    roots = np.random.randint(-20, 20, [numRoots, ])
    rootsC = NumCpp.NdArray(1, numRoots)
    rootsC.setArray(roots)
    poly = np.poly1d(roots, True)
    polyC = NumCpp.Poly1d(rootsC, True)

    numCoefficients = np.random.randint(3, 10, [1, ]).item()
    coefficients = np.random.randint(-20, 20, [numCoefficients, ])
    coefficientsC = NumCpp.NdArray(1, numCoefficients)
    coefficientsC.setArray(coefficients)
    polyC2 = NumCpp.Poly1d(coefficientsC, False)
    poly2 = np.poly1d(np.flip(coefficients))
    assert np.array_equal(np.fliplr((polyC + polyC2).coefficients().getNumpyArray()).flatten(),
                          (poly + poly2).coefficients)

    assert np.array_equal(np.fliplr((polyC - polyC2).coefficients().getNumpyArray()).flatten(),
                          (poly - poly2).coefficients)

    assert np.array_equal(np.fliplr((polyC * polyC2).coefficients().getNumpyArray()).flatten(),
                          (poly * poly2).coefficients)

    exponent = np.random.randint(0, 5, [1, ]).item()
    assert np.array_equal(np.fliplr((polyC2 ** exponent).coefficients().getNumpyArray()).flatten(),
                          (poly2 ** exponent).coefficients)

    polyC.print()


####################################################################################
def test_chebyshev():
    if NumCpp.NUMCPP_NO_USE_BOOST:
        return

    for order in range(ORDER_MAX):
        x = np.random.rand(1).item()
        valuePy = sp.eval_chebyt(order, x)
        valueCpp = NumCpp.chebyshev_t_Scaler(order, x)
        assert np.round(valuePy, DECIMALS_ROUND) == np.round(valueCpp, DECIMALS_ROUND)

    for order in range(ORDER_MAX):
        shapeInput = np.random.randint(10, 100, [2, ], dtype=np.uint32)
        shape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(shape)
        x = np.random.rand(*shapeInput)
        cArray.setArray(x)
        valuePy = sp.eval_chebyt(order, x)
        valueCpp = NumCpp.chebyshev_t_Array(order, cArray)
        assert np.array_equal(np.round(valuePy, DECIMALS_ROUND), np.round(valueCpp, DECIMALS_ROUND))

    for order in range(ORDER_MAX):
        x = np.random.rand(1).item()
        valuePy = sp.eval_chebyu(order, x)
        valueCpp = NumCpp.chebyshev_u_Scaler(order, x)
        assert np.round(valuePy, DECIMALS_ROUND) == np.round(valueCpp, DECIMALS_ROUND)

    for order in range(ORDER_MAX):
        shapeInput = np.random.randint(10, 100, [2, ], dtype=np.uint32)
        shape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(shape)
        x = np.random.rand(*shapeInput)
        cArray.setArray(x)
        valuePy = sp.eval_chebyu(order, x)
        valueCpp = NumCpp.chebyshev_u_Array(order, cArray)
        assert np.array_equal(np.round(valuePy, DECIMALS_ROUND), np.round(valueCpp, DECIMALS_ROUND))


####################################################################################
def test_hermite():
    if NumCpp.NUMCPP_NO_USE_BOOST and not NumCpp.STL_SPECIAL_FUNCTIONS:
        return

    for order in range(ORDER_MAX):
        x = np.random.rand(1).item()
        valuePy = sp.eval_hermite(order, x)
        valueCpp = NumCpp.hermite_Scaler(order, x)
        assert np.round(valuePy, DECIMALS_ROUND) == np.round(valueCpp, DECIMALS_ROUND)

    for order in range(ORDER_MAX):
        shapeInput = np.random.randint(10, 100, [2, ], dtype=np.uint32)
        shape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(shape)
        x = np.random.rand(*shapeInput)
        cArray.setArray(x)
        valuePy = sp.eval_hermite(order, x)
        valueCpp = NumCpp.hermite_Array(order, cArray)
        assert np.array_equal(np.round(valuePy, DECIMALS_ROUND), np.round(valueCpp, DECIMALS_ROUND))


####################################################################################
def test_laguerre():
    if NumCpp.NUMCPP_NO_USE_BOOST and not NumCpp.STL_SPECIAL_FUNCTIONS:
        return

    for order in range(ORDER_MAX):
        x = np.random.rand(1).item()
        valuePy = sp.eval_laguerre(order, x)
        valueCpp = NumCpp.laguerre_Scaler1(order, x)
        assert np.round(valuePy, DECIMALS_ROUND) == np.round(valueCpp, DECIMALS_ROUND)

    for order in range(ORDER_MAX):
        shapeInput = np.random.randint(10, 100, [2, ], dtype=np.uint32)
        shape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(shape)
        x = np.random.rand(*shapeInput)
        cArray.setArray(x)
        valuePy = sp.eval_laguerre(order, x)
        valueCpp = NumCpp.laguerre_Array1(order, cArray)
        assert np.array_equal(np.round(valuePy, DECIMALS_ROUND), np.round(valueCpp, DECIMALS_ROUND))

    for order in range(ORDER_MAX):
        degree = np.random.randint(0, 10, [1, ]).item()
        x = np.random.rand(1).item()
        valuePy = sp.eval_genlaguerre(degree, order, x)
        valueCpp = NumCpp.laguerre_Scaler2(order, degree, x)
        assert np.round(valuePy, DECIMALS_ROUND) == np.round(valueCpp, DECIMALS_ROUND)

    for order in range(ORDER_MAX):
        degree = np.random.randint(0, 10, [1, ]).item()
        shapeInput = np.random.randint(10, 100, [2, ], dtype=np.uint32)
        shape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(shape)
        x = np.random.rand(*shapeInput)
        cArray.setArray(x)
        valuePy = sp.eval_genlaguerre(degree, order, x)
        valueCpp = NumCpp.laguerre_Array2(order, degree, cArray)
        assert np.array_equal(np.round(valuePy, DECIMALS_ROUND), np.round(valueCpp, DECIMALS_ROUND))


####################################################################################
def test_legendre():
    if NumCpp.NUMCPP_NO_USE_BOOST and not NumCpp.STL_SPECIAL_FUNCTIONS:
        return

    for order in range(ORDER_MAX):
        x = np.random.rand(1).item()
        valuePy = sp.eval_legendre(order, x)
        valueCpp = NumCpp.legendre_p_Scaler1(order, x)
        assert np.round(valuePy, DECIMALS_ROUND) == np.round(valueCpp, DECIMALS_ROUND)

    for order in range(ORDER_MAX):
        shapeInput = np.random.randint(10, 100, [2, ], dtype=np.uint32)
        shape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(shape)
        x = np.random.rand(*shapeInput)
        cArray.setArray(x)
        valuePy = sp.eval_legendre(order, x)
        valueCpp = NumCpp.legendre_p_Array1(order, cArray)
        assert np.array_equal(np.round(valuePy, DECIMALS_ROUND), np.round(valueCpp, DECIMALS_ROUND))

    if not NumCpp.NUMCPP_NO_USE_BOOST:
        for order in range(ORDER_MAX):
            x = np.random.rand(1).item()
            degree = np.random.randint(order, ORDER_MAX)
            valuePy = sp.lpmn(order, degree, x)[0][order, degree]
            valueCpp = NumCpp.legendre_p_Scaler2(order, degree, x)
            assert np.round(valuePy, DECIMALS_ROUND) == np.round(valueCpp, DECIMALS_ROUND), \
                f'order={order}, degree={degree}, x={x}'

        for order in range(ORDER_MAX):
            x = np.random.rand(1).item()
            valuePy = sp.lqn(order, x)[0][order]
            valueCpp = NumCpp.legendre_q_Scaler(order, x)
            assert np.round(valuePy, DECIMALS_ROUND) == np.round(valueCpp, DECIMALS_ROUND)


####################################################################################
def test_spherical_harmonic():
    if NumCpp.NUMCPP_NO_USE_BOOST:
        return

    for order in range(ORDER_MAX):
        degree = np.random.randint(order, ORDER_MAX)
        theta = np.random.rand(1).item() * np.pi * 2
        phi = np.random.rand(1).item() * np.pi
        valuePy = sp.sph_harm(order, degree, theta, phi)
        valueCpp = NumCpp.spherical_harmonic(order, degree, theta, phi)
        assert (np.round(valuePy.real, DECIMALS_ROUND) == np.round(valueCpp[0], DECIMALS_ROUND) and
                np.round(valuePy.imag, DECIMALS_ROUND) == np.round(valueCpp[1], DECIMALS_ROUND))

    for order in range(ORDER_MAX):
        degree = np.random.randint(order, ORDER_MAX)
        theta = np.random.rand(1).item() * np.pi * 2
        phi = np.random.rand(1).item() * np.pi
        valuePy = sp.sph_harm(order, degree, theta, phi)
        valueCpp = NumCpp.spherical_harmonic_r(order, degree, theta, phi)
        assert np.round(valuePy.real, DECIMALS_ROUND) == np.round(valueCpp, DECIMALS_ROUND)

    for order in range(ORDER_MAX):
        degree = np.random.randint(order, ORDER_MAX)
        theta = np.random.rand(1).item() * np.pi * 2
        phi = np.random.rand(1).item() * np.pi
        valuePy = sp.sph_harm(order, degree, theta, phi)
        valueCpp = NumCpp.spherical_harmonic_i(order, degree, theta, phi)
        assert np.round(valuePy.imag, DECIMALS_ROUND) == np.round(valueCpp, DECIMALS_ROUND)
