import numpy as np
import os
import sys
sys.path.append(os.path.abspath(r'../lib'))
import NumCpp  # noqa E402

np.random.seed(666)


####################################################################################
def test_num2str():
    """Tests the NumCpp Utils"""
    value = np.random.randint(1, 100, [1, ], dtype=np.int8).item()
    assert NumCpp.num2str(value) == str(value)

    value = np.random.randint(1, 100, [1, ], dtype=np.int16).item()
    assert NumCpp.num2str(value) == str(value)

    value = np.random.randint(1, 100, [1, ], dtype=np.int32).item()
    assert NumCpp.num2str(value) == str(value)

    value = np.random.randint(1, 100, [1, ], dtype=np.int64).item()
    assert NumCpp.num2str(value) == str(value)

    value = np.random.randint(1, 100, [1, ], dtype=np.uint8).item()
    assert NumCpp.num2str(value) == str(value)

    value = np.random.randint(1, 100, [1, ], dtype=np.uint16).item()
    assert NumCpp.num2str(value) == str(value)

    value = np.random.randint(1, 100, [1, ], dtype=np.uint32).item()
    assert NumCpp.num2str(value) == str(value)

    value = np.random.randint(1, 100, [1, ], dtype=np.uint64).item()
    assert NumCpp.num2str(value) == str(value)


####################################################################################
def test_sqr():
    value = np.random.randint(1, 12, [1, ], dtype=np.int8).item()
    assert NumCpp.sqr(value) == value ** 2

    value = np.random.randint(1, 100, [1, ], dtype=np.int16).item()
    assert NumCpp.sqr(value) == value ** 2

    value = np.random.randint(1, 100, [1, ], dtype=np.int32).item()
    assert NumCpp.sqr(value) == value ** 2

    value = np.random.randint(1, 100, [1, ], dtype=np.int64).item()
    assert NumCpp.sqr(value) == value ** 2

    value = np.random.randint(1, 15, [1, ], dtype=np.uint8).item()
    assert NumCpp.sqr(value) == value ** 2

    value = np.random.randint(1, 100, [1, ], dtype=np.uint16).item()
    assert NumCpp.sqr(value) == value ** 2

    value = np.random.randint(1, 100, [1, ], dtype=np.uint32).item()
    assert NumCpp.sqr(value) == value ** 2

    value = np.random.randint(1, 100, [1, ], dtype=np.uint64).item()
    assert NumCpp.sqr(value) == value ** 2

    value = np.random.randint(1, 100, [1, ]).astype(np.double).item()
    assert NumCpp.sqr(value) == value ** 2

    value = np.random.randint(1, 100, [1, ]).astype(np.float32).item()
    assert NumCpp.sqr(value) == value ** 2


####################################################################################
def test_cube():
    value = np.random.randint(1, 6, [1, ], dtype=np.int8).item()
    assert NumCpp.cube(value) == value ** 3

    value = np.random.randint(1, 32, [1, ], dtype=np.int16).item()
    assert NumCpp.cube(value) == value ** 3

    value = np.random.randint(1, 100, [1, ], dtype=np.int32).item()
    assert NumCpp.cube(value) == value ** 3

    value = np.random.randint(1, 100, [1, ], dtype=np.int64).item()
    assert NumCpp.cube(value) == value ** 3

    value = np.random.randint(1, 7, [1, ], dtype=np.uint8).item()
    assert NumCpp.cube(value) == value ** 3

    value = np.random.randint(1, 41, [1, ], dtype=np.uint16).item()
    assert NumCpp.cube(value) == value ** 3

    value = np.random.randint(1, 100, [1, ], dtype=np.uint32).item()
    assert NumCpp.cube(value) == value ** 3

    value = np.random.randint(1, 100, [1, ], dtype=np.uint64).item()
    assert NumCpp.cube(value) == value ** 3

    value = np.random.randint(1, 100, [1, ]).astype(np.double).item()
    assert NumCpp.cube(value) == value ** 3

    value = np.random.randint(1, 100, [1, ]).astype(np.float32).item()
    assert NumCpp.cube(value) == value ** 3


####################################################################################
def test_power():
    value = np.random.randint(1, 4, [1, ], dtype=np.int8).item()
    power = np.random.randint(1, 4, dtype=np.uint8).item()
    assert NumCpp.power(value, power) == np.power(value, power)

    value = np.random.randint(1, 5, [1, ], dtype=np.int16).item()
    assert NumCpp.power(value, power) == np.power(value, power)

    value = np.random.randint(1, 5, [1, ], dtype=np.int32).item()
    assert NumCpp.power(value, power) == np.power(value, power)

    value = np.random.randint(1, 5, [1, ], dtype=np.int64).item()
    assert NumCpp.power(value, power) == np.power(value, power)

    value = np.random.randint(1, 5, [1, ], dtype=np.uint8).item()
    assert NumCpp.power(value, power) == np.power(value, power)

    value = np.random.randint(1, 5, [1, ], dtype=np.uint16).item()
    assert NumCpp.power(value, power) == np.power(value, power)

    value = np.random.randint(1, 5, [1, ], dtype=np.uint32).item()
    assert NumCpp.power(value, power) == np.power(value, power)

    value = np.random.randint(1, 5, [1, ], dtype=np.uint64).item()
    assert NumCpp.power(value, power) == np.power(value, power)

    value = np.random.randint(1, 5, [1, ]).astype(np.double).item()
    assert NumCpp.power(value, power) == np.power(value, power)

    value = np.random.randint(1, 5, [1, ]).astype(np.float32).item()
    assert NumCpp.power(value, power) == np.power(value, power)

    real = np.random.rand(1).astype(np.double).item()
    imag = np.random.rand(1).astype(np.double).item()
    value = np.complex128(complex(real, imag))
    assert np.round(NumCpp.power(value, power), 5) == np.round(np.power(value, power), 5)

    real = np.random.rand(1).astype(np.double).item()
    imag = np.random.rand(1).astype(np.double).item()
    value = np.complex64(complex(real, imag))
    assert np.round(NumCpp.power(value, power), 5) == np.round(np.power(value, power), 5)


####################################################################################
def test_powerf():
    value = np.random.randint(1, 5, [1, ], dtype=np.int8).item()
    power = np.random.rand(1).item() * 5
    assert NumCpp.powerf(value, power) == np.power(value, power)

    value = np.random.randint(1, 5, [1, ], dtype=np.int16).item()
    assert NumCpp.powerf(value, power) == np.power(value, power)

    value = np.random.randint(1, 5, [1, ], dtype=np.int32).item()
    assert NumCpp.powerf(value, power) == np.power(value, power)

    value = np.random.randint(1, 5, [1, ], dtype=np.int64).item()
    assert NumCpp.powerf(value, power) == np.power(value, power)

    value = np.random.randint(1, 5, [1, ], dtype=np.uint8).item()
    assert NumCpp.powerf(value, power) == np.power(value, power)

    value = np.random.randint(1, 5, [1, ], dtype=np.uint16).item()
    assert NumCpp.powerf(value, power) == np.power(value, power)

    value = np.random.randint(1, 5, [1, ], dtype=np.uint32).item()
    assert NumCpp.powerf(value, power) == np.power(value, power)

    value = np.random.randint(1, 5, [1, ], dtype=np.uint64).item()
    assert NumCpp.powerf(value, power) == np.power(value, power)

    value = np.random.rand(1).astype(np.double).item()
    assert np.round(NumCpp.powerf(value, power), 5) == \
        np.round(np.power(value, power), 5)

    value = np.random.rand(1).astype(np.float32).item()
    assert np.round(NumCpp.powerf(value, np.float32(power)), 5) == \
        np.round(np.power(value, np.float32(power)), 5)

    realPower = np.random.rand(1).astype(np.double).item()
    imagPower = np.random.rand(1).astype(np.double).item()
    power = np.complex128(complex(realPower, imagPower))
    real = np.random.rand(1).astype(np.double).item()
    imag = np.random.rand(1).astype(np.double).item()
    value = np.complex128(complex(real, imag))
    assert np.round(NumCpp.powerf_complex(value, power), 5) == np.round(np.float_power(value, power), 5)

    realPower = np.random.rand(1).astype(np.double).item()
    imagPower = np.random.rand(1).astype(np.double).item()
    power = np.complex64(complex(realPower, imagPower))
    real = np.random.rand(1).astype(np.double).item()
    imag = np.random.rand(1).astype(np.double).item()
    value = np.complex64(complex(real, imag))
    assert np.round(NumCpp.powerf_complex(value, power), 5) == np.round(np.float_power(value, power), 5)
