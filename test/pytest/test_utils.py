import time

import numpy as np

import NumCppPy as NumCpp  # noqa E402

np.random.seed(666)


####################################################################################
def test_num2str():
    """Tests the NumCpp Utils"""
    value = np.random.randint(
        1,
        100,
        [
            1,
        ],
        dtype=np.uint64,
    ).item()
    assert NumCpp.num2str(value) == str(value)


####################################################################################
def test_sqr():
    value = np.random.randint(
        1,
        100,
        [
            1,
        ],
        dtype=np.int64,
    ).item()
    assert NumCpp.sqr(value) == value**2

    value = (
        np.random.randint(
            1,
            100,
            [
                1,
            ],
        )
        .astype(float)
        .item()
    )
    assert NumCpp.sqr(value) == value**2


####################################################################################
def test_cube():
    value = np.random.randint(
        1,
        100,
        [
            1,
        ],
        dtype=np.int64,
    ).item()
    assert NumCpp.cube(value) == value**3

    value = (
        np.random.randint(
            1,
            100,
            [
                1,
            ],
        )
        .astype(float)
        .item()
    )
    assert NumCpp.cube(value) == value**3


####################################################################################
def test_power():
    value = np.random.randint(
        1,
        4,
        [
            1,
        ],
        dtype=np.int8,
    ).item()
    power = np.random.randint(1, 4, dtype=np.uint8).item()
    assert NumCpp.power(value, power) == np.power(value, power)

    value = (
        np.random.randint(
            1,
            5,
            [
                1,
            ],
        )
        .astype(float)
        .item()
    )
    assert NumCpp.power(value, power) == np.power(value, power)

    real = np.random.rand(1).astype(float).item()
    imag = np.random.rand(1).astype(float).item()
    value = np.complex128(complex(real, imag))
    assert np.round(NumCpp.power(value, power), 5) == np.round(np.power(value, power), 5)


####################################################################################
def test_powerf():
    value = np.random.randint(
        1,
        5,
        [
            1,
        ],
        dtype=np.int8,
    ).item()
    power = np.random.rand(1).item() * 5
    assert NumCpp.powerf(value, power) == np.power(value, power)

    value = np.random.rand(1).astype(float).item()
    assert np.round(NumCpp.powerf(value, power), 5) == np.round(np.power(value, power), 5)

    realPower = np.random.rand(1).astype(float).item()
    imagPower = np.random.rand(1).astype(float).item()
    power = np.complex128(complex(realPower, imagPower))
    real = np.random.rand(1).astype(float).item()
    imag = np.random.rand(1).astype(float).item()
    value = np.complex128(complex(real, imag))
    assert np.round(NumCpp.powerf_complex(value, power), 5) == np.round(np.float_power(value, power), 5)


####################################################################################
def test_timeit():
    value1 = 666
    value2 = 357

    def function1(value1_: int, value2_: int) -> None:
        time.sleep(1 / 10000000)

    NumCpp.timeit(1000, True, function1, value1, value2)

    def function2(value1_: int, value2_: int) -> int:
        time.sleep(1 / 10000000)
        return value1 + value2

    NumCpp.timeit(1000, True, function2, value1, value2)
