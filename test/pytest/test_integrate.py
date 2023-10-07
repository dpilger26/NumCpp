import numpy as np

import NumCppPy as NumCpp  # noqa E402


####################################################################################
NUM_DECIMALS_ROUND = 1


####################################################################################
def test_seed():
    np.random.seed(666)


####################################################################################
def test_gauss_legendre():
    numCoefficients = np.random.randint(
        2,
        5,
        [
            1,
        ],
    ).item()
    coefficients = np.random.randint(
        -20,
        20,
        [
            numCoefficients,
        ],
    )
    coefficientsC = NumCpp.NdArray(1, numCoefficients)
    coefficientsC.setArray(coefficients)
    poly = np.poly1d(np.flipud(coefficients), False)
    polyIntegral = poly.integ()
    polyC = NumCpp.Poly1d(coefficientsC, NumCpp.IsRoots.FALSE)
    a, b = np.sort(np.random.rand(2) * 100 - 50)
    area = np.round(polyIntegral(b) - polyIntegral(a), NUM_DECIMALS_ROUND)
    areaC = np.round(NumCpp.integrate_gauss_legendre(polyC, a, b), NUM_DECIMALS_ROUND)
    assert area == areaC


####################################################################################
def test_romberg():
    PERCENT_LEEWAY = 0.1
    numCoefficients = np.random.randint(
        2,
        5,
        [
            1,
        ],
    ).item()
    coefficients = np.random.randint(
        -20,
        20,
        [
            numCoefficients,
        ],
    )
    coefficientsC = NumCpp.NdArray(1, numCoefficients)
    coefficientsC.setArray(coefficients)
    poly = np.poly1d(np.flipud(coefficients), False)
    polyIntegral = poly.integ()
    polyC = NumCpp.Poly1d(coefficientsC, NumCpp.IsRoots.FALSE)
    a, b = np.sort(np.random.rand(2) * 100 - 50)
    area = np.round(polyIntegral(b) - polyIntegral(a), NUM_DECIMALS_ROUND)
    areaC = np.round(NumCpp.integrate_romberg(polyC, a, b), NUM_DECIMALS_ROUND)
    # romberg is much less acurate so let's give it some leeway
    areaLow, areaHigh = np.sort([area * (1 - PERCENT_LEEWAY), area * (1 + PERCENT_LEEWAY)])
    assert areaLow < areaC < areaHigh


####################################################################################
def test_simpson():
    numCoefficients = np.random.randint(
        2,
        5,
        [
            1,
        ],
    ).item()
    coefficients = np.random.randint(
        -20,
        20,
        [
            numCoefficients,
        ],
    )
    coefficientsC = NumCpp.NdArray(1, numCoefficients)
    coefficientsC.setArray(coefficients)
    poly = np.poly1d(np.flipud(coefficients), False)
    polyIntegral = poly.integ()
    polyC = NumCpp.Poly1d(coefficientsC, NumCpp.IsRoots.FALSE)
    a, b = np.sort(np.random.rand(2) * 100 - 50)
    area = np.round(polyIntegral(b) - polyIntegral(a), NUM_DECIMALS_ROUND)
    areaC = np.round(NumCpp.integrate_simpson(polyC, a, b), NUM_DECIMALS_ROUND)
    assert area == areaC


####################################################################################
def test_trapazoidal():
    numCoefficients = np.random.randint(
        2,
        5,
        [
            1,
        ],
    ).item()
    coefficients = np.random.randint(
        -20,
        20,
        [
            numCoefficients,
        ],
    )
    coefficientsC = NumCpp.NdArray(1, numCoefficients)
    coefficientsC.setArray(coefficients)
    poly = np.poly1d(np.flipud(coefficients), False)
    polyIntegral = poly.integ()
    polyC = NumCpp.Poly1d(coefficientsC, NumCpp.IsRoots.FALSE)
    a, b = np.sort(np.random.rand(2) * 100 - 50)
    area = np.round(polyIntegral(b) - polyIntegral(a), NUM_DECIMALS_ROUND)
    areaC = np.round(NumCpp.integrate_trapazoidal(polyC, a, b), NUM_DECIMALS_ROUND)
    assert area == areaC
