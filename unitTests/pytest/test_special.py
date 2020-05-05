import numpy as np
import scipy.special as sp
import os
import sys
if sys.platform == 'linux':
    sys.path.append(r'../lib')
else:
    sys.path.append(os.path.abspath('../build/x64/Release'))
import NumCpp


NUM_DECIMALS_ROUND = 7


####################################################################################
def test_special():
    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.airy_ai_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.airy(value)[0].item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.airy_ai_Array(cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.airy(data)[0], NUM_DECIMALS_ROUND))

    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.airy_ai_prime_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.airy(value)[1].item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.airy_ai_prime_Array(cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.airy(data)[1], NUM_DECIMALS_ROUND))

    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.airy_bi_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.airy(value)[2].item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.airy_bi_Array(cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.airy(data)[2], NUM_DECIMALS_ROUND))

    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.airy_bi_prime_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.airy(value)[3].item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.airy_bi_prime_Array(cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.airy(data)[3], NUM_DECIMALS_ROUND))

    value = np.random.randint(0, 20)
    assert (roundScaler(NumCpp.bernoulli_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.bernoulli(value)[-1], NUM_DECIMALS_ROUND))

    order = np.random.randint(0, 10)
    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.bessel_in_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.iv(order, value).item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 10)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.bessel_in_Array(order, cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.iv(order, data), NUM_DECIMALS_ROUND))

    order = np.random.randint(0, 10)
    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.bessel_in_prime_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.ivp(order, value).item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 10)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.bessel_in_prime_Array(order, cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.ivp(order, data), NUM_DECIMALS_ROUND))

    order = np.random.randint(0, 10)
    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.bessel_jn_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.jv(order, value).item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 10)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.bessel_jn_Array(order, cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.jv(order, data), NUM_DECIMALS_ROUND))

    order = np.random.randint(0, 10)
    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.bessel_jn_prime_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.jvp(order, value).item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 10)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.bessel_jn_prime_Array(order, cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.jvp(order, data), NUM_DECIMALS_ROUND))

    order = np.random.randint(0, 10)
    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.bessel_kn_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.kn(order, value).item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 5)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.bessel_kn_Array(order, cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.kn(order, data), NUM_DECIMALS_ROUND))

    order = np.random.randint(0, 5)
    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.bessel_kn_prime_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.kvp(order, value).item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 5)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.bessel_kn_prime_Array(order, cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.kvp(order, data), NUM_DECIMALS_ROUND))

    order = np.random.randint(0, 5)
    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.bessel_yn_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.yn(order, value).item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 5)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.bessel_yn_Array(order, cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.yn(order, data), NUM_DECIMALS_ROUND))

    order = np.random.randint(0, 5)
    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.bessel_yn_prime_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.yvp(order, value).item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 5)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.bessel_yn_prime_Array(order, cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.yvp(order, data), NUM_DECIMALS_ROUND))

    a = np.random.rand(1).item() * 10
    b = np.random.rand(1).item() * 10
    assert (roundScaler(NumCpp.beta_Scaler(a, b), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.beta(a, b).item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    aArray = NumCpp.NdArray(shape)
    bArray = NumCpp.NdArray(shape)
    a = np.random.rand(shape.rows, shape.cols) * 10
    b = np.random.rand(shape.rows, shape.cols) * 10
    aArray.setArray(a)
    bArray.setArray(b)
    assert np.array_equal(roundArray(NumCpp.beta_Array(aArray, bArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.beta(a, b), NUM_DECIMALS_ROUND))

    n = np.random.randint(0, 50)
    r = np.random.randint(0, n + 1)
    assert round(NumCpp.cnr(n, r)) == round(sp.comb(n, r))

    order = np.random.randint(0, 6)
    value = np.random.rand(1).item() * 10
    assert (roundComplex(complex(NumCpp.cyclic_hankel_1(order, value)), NUM_DECIMALS_ROUND) ==
            roundComplex(sp.hankel1(order, value), NUM_DECIMALS_ROUND))

    order = np.random.randint(0, 6)
    value = np.random.rand(1).item() * 10
    assert (roundComplex(complex(NumCpp.cyclic_hankel_2(order, value)), NUM_DECIMALS_ROUND) ==
            roundComplex(sp.hankel2(order, value), NUM_DECIMALS_ROUND))

    value = np.random.rand(1).item() * 10
    assert (roundScaler(NumCpp.digamma_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.digamma(value), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 10
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.digamma_Array(cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.digamma(data), NUM_DECIMALS_ROUND))

    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.erf_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.erf(value), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.erf_Array(cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.erf(data), NUM_DECIMALS_ROUND))

    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.erf_inv_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.erfinv(value), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.erf_inv_Array(cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.erfinv(data), NUM_DECIMALS_ROUND))

    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.erfc_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.erfc(value), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.erfc_Array(cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.erfc(data), NUM_DECIMALS_ROUND))

    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.erfc_inv_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.erfcinv(value), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.erfc_inv_Array(cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.erfcinv(data), NUM_DECIMALS_ROUND))

    n = np.random.randint(0, 170)
    assert (roundScaler(NumCpp.factorial_Scaler(n), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.factorial(n).item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(0, 170, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.factorial_Array(cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.factorial(data), NUM_DECIMALS_ROUND))

    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.gamma_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.gamma(value), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.gamma_Array(cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.gamma(data), NUM_DECIMALS_ROUND))

    # There is no scipy equivalent to this function
    value = np.random.rand(1).item()
    assert NumCpp.gamma1pm1_Scaler(value) is not None

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert NumCpp.gamma1pm1_Array(cArray) is not None

    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.log_gamma_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.loggamma(value), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.log_gamma_Array(cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.loggamma(data), NUM_DECIMALS_ROUND))

    n = np.random.randint(0, 10)
    r = np.random.randint(0, n + 1)
    assert round(NumCpp.pnr(n, r)) == round(sp.perm(n, r))

    order = np.random.randint(1, 5)
    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.polygamma_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.polygamma(order, value).item(), NUM_DECIMALS_ROUND))

    order = np.random.randint(1, 5)
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.polygamma_Array(order, cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.polygamma(order, data), NUM_DECIMALS_ROUND))

    # There is no scipy equivalent to this function
    value = np.random.randint(10000)
    assert NumCpp.prime_Scaler(value) is not None

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(0, 10000, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.prime_Array(cArray) is not None

    value = np.random.rand(1).item() * 5 + 1
    assert (roundScaler(NumCpp.riemann_zeta_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.zeta(value, 1).item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols) * 5 + 1
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.riemann_zeta_Array(cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.zeta(data, 1), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.softmax(cArray, NumCpp.Axis.NONE), NUM_DECIMALS_ROUND),
                          roundArray(sp.softmax(data), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.softmax(cArray, NumCpp.Axis.ROW), NUM_DECIMALS_ROUND),
                          roundArray(sp.softmax(data, axis=0), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.softmax(cArray, NumCpp.Axis.COL), NUM_DECIMALS_ROUND),
                          roundArray(sp.softmax(data, axis=1), NUM_DECIMALS_ROUND))

    order = np.random.randint(0, 10)
    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.spherical_bessel_jn_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.spherical_jn(order, value).item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 10)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.spherical_bessel_jn_Array(order, cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.spherical_jn(order, data), NUM_DECIMALS_ROUND))

    order = np.random.randint(0, 10)
    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.spherical_bessel_yn_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.spherical_yn(order, value).item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 10)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.spherical_bessel_yn_Array(order, cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.spherical_yn(order, data), NUM_DECIMALS_ROUND))

    # There is no equivalent scipy functions
    order = np.random.randint(0, 10)
    value = np.random.rand(1).item()
    assert NumCpp.spherical_hankel_1(order, value) is not None

    order = np.random.randint(0, 10)
    value = np.random.rand(1).item()
    assert NumCpp.spherical_hankel_2(order, value) is not None

    value = np.random.rand(1).item()
    assert (roundScaler(NumCpp.trigamma_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.polygamma(1, value).item(), NUM_DECIMALS_ROUND))

    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert np.array_equal(roundArray(NumCpp.trigamma_Array(cArray), NUM_DECIMALS_ROUND),
                          roundArray(sp.polygamma(1, data), NUM_DECIMALS_ROUND))


####################################################################################
def roundScaler(value: float, numDecimals: int) -> float:
    return float(f'{{:.{numDecimals}g}}'.format(value))


####################################################################################
def roundArray(values: np.ndarray, numDecimals: int) -> np.ndarray:
    func = np.vectorize(roundScaler)
    return func(values, numDecimals)


####################################################################################
def roundComplex(value: complex, numDecimals: int) -> complex:
    return complex(roundScaler(value.real, numDecimals),
                   roundScaler(value.imag, numDecimals))
