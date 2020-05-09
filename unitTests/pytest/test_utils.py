import numpy as np
import os
import sys
sys.path.append(os.path.abspath(r'../lib'))
import NumCpp  # noqa E402


####################################################################################
def test_utils():
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

    value = np.random.randint(1, 4, [1, ], dtype=np.int8).item()
    power = np.random.randint(1, 4, dtype=np.uint8).item()
    assert NumCpp.power(value, power) == value ** power

    value = np.random.randint(1, 10, [1, ], dtype=np.int16).item()
    assert NumCpp.power(value, power) == value ** power

    value = np.random.randint(1, 10, [1, ], dtype=np.int32).item()
    assert NumCpp.power(value, power) == value ** power

    value = np.random.randint(1, 10, [1, ], dtype=np.int64).item()
    assert NumCpp.power(value, power) == value ** power

    value = np.random.randint(1, 4, [1, ], dtype=np.uint8).item()
    assert NumCpp.power(value, power) == value ** power

    value = np.random.randint(1, 10, [1, ], dtype=np.uint16).item()
    assert NumCpp.power(value, power) == value ** power

    value = np.random.randint(1, 10, [1, ], dtype=np.uint32).item()
    assert NumCpp.power(value, power) == value ** power

    value = np.random.randint(1, 10, [1, ], dtype=np.uint64).item()
    assert NumCpp.power(value, power) == value ** power

    value = np.random.randint(1, 10, [1, ]).astype(np.double).item()
    assert NumCpp.power(value, power) == value ** power

    value = np.random.randint(1, 10, [1, ]).astype(np.float32).item()
    assert NumCpp.power(value, power) == value ** power

    value = np.random.randint(1, 4, [1, ], dtype=np.int8).item()
    power = np.random.rand(1).item() * 10
    assert NumCpp.powerf(value, power) == value ** power

    value = np.random.randint(1, 10, [1, ], dtype=np.int16).item()
    assert NumCpp.powerf(value, power) == value ** power

    value = np.random.randint(1, 10, [1, ], dtype=np.int32).item()
    assert NumCpp.powerf(value, power) == value ** power

    value = np.random.randint(1, 10, [1, ], dtype=np.int64).item()
    assert NumCpp.powerf(value, power) == value ** power

    value = np.random.randint(1, 4, [1, ], dtype=np.uint8).item()
    assert NumCpp.powerf(value, power) == value ** power

    value = np.random.randint(1, 10, [1, ], dtype=np.uint16).item()
    assert NumCpp.powerf(value, power) == value ** power

    value = np.random.randint(1, 10, [1, ], dtype=np.uint32).item()
    assert NumCpp.powerf(value, power) == value ** power

    value = np.random.randint(1, 10, [1, ], dtype=np.uint64).item()
    assert NumCpp.powerf(value, power) == value ** power

    value = np.random.randint(1, 10, [1, ]).astype(np.double).item()
    assert NumCpp.powerf(value, power) == value ** power

    value = np.random.randint(1, 10, [1, ]).astype(np.float32).item()
    assert NumCpp.powerf(value, power) == value ** power
