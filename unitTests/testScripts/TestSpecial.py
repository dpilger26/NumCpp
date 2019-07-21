import numpy as np
import scipy.special as sp
from termcolor import colored
import sys
if sys.platform == 'linux':
    sys.path.append(r'../lib')
else:
    sys.path.append(r'../build/x64/Release')
import NumCpp


NUM_DECIMALS_ROUND = 8


####################################################################################
def doTest():
    print(colored('Testing Special Module', 'magenta'))

    print(colored('Testing airy_ai scaler', 'cyan'))
    value = np.random.rand(1).item()
    if (roundScaler(NumCpp.airy_ai_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.airy(value)[0].item(), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing airy_ai array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(roundArray(NumCpp.airy_ai_Array(cArray), NUM_DECIMALS_ROUND),
                      roundArray(sp.airy(data)[0], NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing airy_ai_prime scaler', 'cyan'))
    value = np.random.rand(1).item()
    if (roundScaler(NumCpp.airy_ai_prime_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.airy(value)[1].item(), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing airy_ai_prime array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(roundArray(NumCpp.airy_ai_prime_Array(cArray), NUM_DECIMALS_ROUND),
                      roundArray(sp.airy(data)[1], NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing airy_bi scaler', 'cyan'))
    value = np.random.rand(1).item()
    if (roundScaler(NumCpp.airy_bi_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.airy(value)[2].item(), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing airy_bi array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(roundArray(NumCpp.airy_bi_Array(cArray), NUM_DECIMALS_ROUND),
                      roundArray(sp.airy(data)[2], NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing airy_bi_prime scaler', 'cyan'))
    value = np.random.rand(1).item()
    if (roundScaler(NumCpp.airy_bi_prime_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.airy(value)[3].item(), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing airy_bi_prime array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(roundArray(NumCpp.airy_bi_prime_Array(cArray), NUM_DECIMALS_ROUND),
                      roundArray(sp.airy(data)[3], NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bernoulli scaler', 'cyan'))
    value = np.random.randint(0, 20)
    if (roundScaler(NumCpp.bernoulli_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.bernoulli(value)[-1], NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bessel_in scaler', 'cyan'))
    order = np.random.randint(0, 10)
    value = np.random.rand(1).item()
    if (roundScaler(NumCpp.bessel_in_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.iv(order, value).item(), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bessel_in array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 10)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(roundArray(NumCpp.bessel_in_Array(order, cArray), NUM_DECIMALS_ROUND),
                      roundArray(sp.iv(order, data), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bessel_in_prime scaler', 'cyan'))
    order = np.random.randint(0, 10)
    value = np.random.rand(1).item()
    if (roundScaler(NumCpp.bessel_in_prime_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.ivp(order, value).item(), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bessel_in_prime array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 10)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(roundArray(NumCpp.bessel_in_prime_Array(order, cArray), NUM_DECIMALS_ROUND),
                      roundArray(sp.ivp(order, data), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bessel_jn scaler', 'cyan'))
    order = np.random.randint(0, 10)
    value = np.random.rand(1).item()
    if (roundScaler(NumCpp.bessel_jn_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.jv(order, value).item(), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bessel_jn array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 10)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(roundArray(NumCpp.bessel_jn_Array(order, cArray), NUM_DECIMALS_ROUND),
                      roundArray(sp.jv(order, data), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bessel_jn_prime scaler', 'cyan'))
    order = np.random.randint(0, 10)
    value = np.random.rand(1).item()
    if (roundScaler(NumCpp.bessel_jn_prime_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.jvp(order, value).item(), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bessel_jn_prime array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 10)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(roundArray(NumCpp.bessel_jn_prime_Array(order, cArray), NUM_DECIMALS_ROUND),
                      roundArray(sp.jvp(order, data), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bessel_kn scaler', 'cyan'))
    order = np.random.randint(0, 10)
    value = np.random.rand(1).item()
    if (roundScaler(NumCpp.bessel_kn_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.kn(order, value).item(), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bessel_kn array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 5)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(roundArray(NumCpp.bessel_kn_Array(order, cArray), NUM_DECIMALS_ROUND),
                      roundArray(sp.kn(order, data), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bessel_kn_prime scaler', 'cyan'))
    order = np.random.randint(0, 5)
    value = np.random.rand(1).item()
    if (roundScaler(NumCpp.bessel_kn_prime_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.kvp(order, value).item(), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bessel_kn_prime array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 5)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(roundArray(NumCpp.bessel_kn_prime_Array(order, cArray), NUM_DECIMALS_ROUND),
                      roundArray(sp.kvp(order, data), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bessel_yn scaler', 'cyan'))
    order = np.random.randint(0, 5)
    value = np.random.rand(1).item()
    if (roundScaler(NumCpp.bessel_yn_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.yn(order, value).item(), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bessel_yn array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 5)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(roundArray(NumCpp.bessel_yn_Array(order, cArray), NUM_DECIMALS_ROUND),
                      roundArray(sp.yn(order, data), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bessel_yn_prime scaler', 'cyan'))
    order = np.random.randint(0, 5)
    value = np.random.rand(1).item()
    if (roundScaler(NumCpp.bessel_yn_prime_Scaler(order, value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.yvp(order, value).item(), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing bessel_yn_prime array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    order = np.random.randint(0, 5)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(roundArray(NumCpp.bessel_yn_prime_Array(order, cArray), NUM_DECIMALS_ROUND),
                      roundArray(sp.yvp(order, data), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing beta scaler', 'cyan'))
    a = np.random.rand(1).item() * 10
    b = np.random.rand(1).item() * 10
    if (roundScaler(NumCpp.beta_Scaler(a, b), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.beta(a, b).item(), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing beta array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    aArray = NumCpp.NdArray(shape)
    bArray = NumCpp.NdArray(shape)
    a = np.random.rand(shape.rows, shape.cols) * 10
    b = np.random.rand(shape.rows, shape.cols) * 10
    aArray.setArray(a)
    bArray.setArray(b)
    if np.array_equal(roundArray(NumCpp.beta_Array(aArray, bArray), NUM_DECIMALS_ROUND),
                      roundArray(sp.beta(a, b), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))










    print(colored('Testing erf scaler', 'cyan'))
    value = np.random.rand(1).item()
    if (roundScaler(NumCpp.erf_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.erf(value), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing erf array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(roundArray(NumCpp.erf_Array(cArray), NUM_DECIMALS_ROUND),
                      roundArray(sp.erf(data), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing erfc scaler', 'cyan'))
    value = np.random.rand(1).item()
    if (roundScaler(NumCpp.erfc_Scaler(value), NUM_DECIMALS_ROUND) ==
            roundScaler(sp.erfc(value), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))

    print(colored('Testing erfc array', 'cyan'))
    shapeInput = np.random.randint(20, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    if np.array_equal(roundArray(NumCpp.erfc_Array(cArray), NUM_DECIMALS_ROUND),
                      roundArray(sp.erfc(data), NUM_DECIMALS_ROUND)):
        print(colored('\tPASS', 'green'))
    else:
        print(colored('\tFAIL', 'red'))


####################################################################################
def roundScaler(value: float, numDecimals: int) -> float:
    return float(f'{{:.{numDecimals}g}}'.format(value))


####################################################################################
def roundArray(values: np.ndarray, numDecimals: int) -> np.ndarray:
    func = np.vectorize(roundScaler)
    return func(values, numDecimals)


####################################################################################
if __name__ == '__main__':
    doTest()
