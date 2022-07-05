import numpy as np
import os
import sys

import NumCppPy as NumCpp  # noqa E402

np.random.seed(666)


####################################################################################
def test_plus_equal():
    # (1) Arithmetic Arrays
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhsC), lhs + rhs)

    # (1) Complex Arrays
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhsC), lhs + rhs)

    # (2) Complex Array, Arithmetic Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhsC), lhs + rhs)

    # (3) Arithmetic Array, Arithmetic Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhs), lhs + rhs)

    # (3) Complex Array, Complex Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + \
        1j * np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhs), lhs + rhs)

    # (4) Complex Array, Arithmetic Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + \
        1j * np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhs), lhs + rhs)


####################################################################################
def test_plus():
    # (1) Arithmetic Arrays
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # (1) Complex Arrays
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # (2) Arithmetic Array, Complex Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # (3) Complex Array, Arithmetic Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # (4) Arithmetic Array, Arithmetic Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhs), lhs + rhs)

    # (4) Complex Array, Complex Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhs), lhs + rhs)

    # (5) Arithmetic Scaler, Arithmetic Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    rhsC = NumCpp.NdArray(shape)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhs = float(np.random.randint(0, 100))
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhs, rhsC), lhs + rhs)

    # (5) Complex Scaler, Complex Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhs, rhsC), lhs + rhs)

    # (6) Arithmetic Array, Complex Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhs), lhs + rhs)

    # (7) Complex Scaler, Arithmetic Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlus(rhs, lhsC), rhs + lhs)

    # (8) Complex Array, Arithmetic Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhs), lhs + rhs)

    # (9) Arithmetic Scaler, Complex Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlus(rhs, lhsC), rhs + lhs)


####################################################################################
def test_negative():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorNegative(cArray), -data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorNegative(cArray), -data)


####################################################################################
def test_minus_equal():
    # (1) Arithmetic Arrays
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhsC), lhs - rhs)

    # (1) Complex Arrays
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhsC), lhs - rhs)

    # (2) Complex Array, Arithmetic Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhsC), lhs - rhs)

    # (3) Arithmetic Array, Arithmetic Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhs), lhs - rhs)

    # (3) Complex Array, Complex Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + \
        1j * np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhs), lhs - rhs)

    # (4) Complex Array, Arithmetic Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + \
        1j * np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhs), lhs - rhs)


####################################################################################
def test_minus():
    # (1) Arithmetic Arrays
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # (1) Complex Arrays
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # (2) Arithmetic Array, Complex Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # (3) Complex Array, Arithmetic Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # (4) Arithmetic Array, Arithmetic Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhs), lhs - rhs)

    # (4) Complex Array, Complex Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhs), lhs - rhs)

    # (5) Arithmetic Scaler, Arithmetic Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    rhsC = NumCpp.NdArray(shape)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhs = float(np.random.randint(0, 100))
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhs, rhsC), lhs - rhs)

    # (5) Complex Scaler, Complex Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhs, rhsC), lhs - rhs)

    # (6) Arithmetic Array, Complex Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhs), lhs - rhs)

    # (7) Complex Scaler, Arithmetic Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinus(rhs, lhsC), rhs - lhs)

    # (8) Complex Array, Arithmetic Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhs), lhs - rhs)

    # (9) Arithmetic Scaler, Complex Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinus(rhs, lhsC), rhs - lhs)


####################################################################################
def test_multiply_equal():
    # (1) Arithmetic Arrays
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhsC), lhs * rhs)

    # (1) Complex Arrays
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhsC), lhs * rhs)

    # (2) Complex Array, Arithmetic Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhsC), lhs * rhs)

    # (3) Arithmetic Array, Arithmetic Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhs), lhs * rhs)

    # (3) Complex Array, Complex Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + \
        1j * np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhs), lhs * rhs)

    # (4) Complex Array, Arithmetic Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + \
        1j * np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhs), lhs * rhs)


####################################################################################
def test_multiply():
    # (1) Arithmetic Arrays
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # (1) Complex Arrays
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # (2) Arithmetic Array, Complex Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # (3) Complex Array, Arithmetic Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # (4) Arithmetic Array, Arithmetic Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhs), lhs * rhs)

    # (4) Complex Array, Complex Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhs), lhs * rhs)

    # (5) Arithmetic Scaler, Arithmetic Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    rhsC = NumCpp.NdArray(shape)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhs = float(np.random.randint(0, 100))
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhs, rhsC), lhs * rhs)

    # (5) Complex Scaler, Complex Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhs, rhsC), lhs * rhs)

    # (6) Arithmetic Array, Complex Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhs), lhs * rhs)

    # (7) Complex Scaler, Arithmetic Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiply(rhs, lhsC), rhs * lhs)

    # (8) Complex Array, Arithmetic Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhs), lhs * rhs)

    # (9) Arithmetic Scaler, Complex Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiply(rhs, lhsC), rhs * lhs)


####################################################################################
def test_divide_equal():
    # (1) Arithmetic Arrays
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhsC), 9),
                          np.round(lhs / rhs, 9))

    # (1) Complex Arrays
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhsC), 9),
                          np.round(lhs / rhs, 9))

    # (2) Complex Array, Arithmetic Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhsC), 9),
                          np.round(lhs / rhs, 9))

    # (3) Arithmetic Array, Arithmetic Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(1, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhs), 9),
                          np.round(lhs / rhs, 9))

    # (3) Complex Array, Complex Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + \
        1j * np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(1, 100)) + 1j * \
        float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhs), 9),
                          np.round(lhs / rhs, 9))

    # (4) Complex Array, Arithmetic Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + \
        1j * np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(1, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhs), 9),
                          np.round(lhs / rhs, 9))


####################################################################################
def test_divide():
    # (1) Arithmetic Arrays
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 9),
                          np.round(lhs / rhs, 9))

    # (1) Complex Arrays
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 9),
                          np.round(lhs / rhs, 9))

    # (2) Arithmetic Array, Complex Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 9),
                          np.round(lhs / rhs, 9))

    # (3) Complex Array, Arithmetic Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 9),
                          np.round(lhs / rhs, 9))

    # (4) Arithmetic Array, Arithmetic Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(1, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhs), 9),
                          np.round(lhs / rhs, 9))

    # (4) Complex Array, Complex Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(1, 100)) + 1j * \
        float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhs), 9),
                          np.round(lhs / rhs, 9))

    # (5) Arithmetic Scaler, Arithmetic Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    rhsC = NumCpp.NdArray(shape)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    lhs = float(np.random.randint(0, 100))
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhs, rhsC), 9),
                          np.round(lhs / rhs, 9))

    # (5) Complex Scaler, Complex Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhs, rhsC), 9),
                          np.round(lhs / rhs, 9))

    # (6) Arithmetic Array, Complex Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(1, 100)) + 1j * \
        float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhs), 9),
                          np.round(lhs / rhs, 9))

    # (7) Complex Scaler, Arithmetic Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * \
        float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(rhs, lhsC), 9),
                          np.round(rhs / lhs, 9))

    # (8) Complex Array, Arithmetic Scaler
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(1, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhs), 9),
                          np.round(lhs / rhs, 9))

    # (9) Arithmetic Scaler, Complex Array
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * \
        np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(rhs, lhsC), 9),
                          np.round(rhs / lhs, 9))


####################################################################################
def test_equality():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorEquality(
        cArray, value), data == value)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorEquality(
        cArray, value), data == value)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorEquality(
        value, cArray), value == data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorEquality(
        value, cArray), value == data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorEquality(
        cArray1, cArray2), data1 == data2)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    real2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data1 = real1 + 1j * imag1
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorEquality(
        cArray1, cArray2), data1 == data2)


####################################################################################
def test_not_equality():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorNotEquality(
        cArray, value), data != value)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorNotEquality(
        cArray, value), data != value)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorNotEquality(
        value, cArray), value != data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorNotEquality(
        value, cArray), value != data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorNotEquality(
        cArray1, cArray2), data1 != data2)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    real2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data1 = real1 + 1j * imag1
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorNotEquality(
        cArray1, cArray2), data1 != data2)


####################################################################################
def test_less():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorLess(cArray, value), data < value)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, ]).item(
    ) + 1j * np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorLess(cArray, value), data < value)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorLess(value, cArray), value < data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, ]).item(
    ) + 1j * np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorLess(value, cArray), value < data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorLess(cArray1, cArray2), data1 < data2)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    real2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data1 = real1 + 1j * imag1
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorLess(cArray1, cArray2), data1 < data2)


####################################################################################
def test_greater():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorGreater(cArray, value), data > value)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, ]).item(
    ) + 1j * np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorGreater(cArray, value), data > value)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorGreater(value, cArray), value > data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, ]).item(
    ) + 1j * np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorGreater(value, cArray), value > data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorGreater(
        cArray1, cArray2), data1 > data2)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    real2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data1 = real1 + 1j * imag1
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorGreater(
        cArray1, cArray2), data1 > data2)


####################################################################################
def test_less_equal():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorLessEqual(
        cArray, value), data <= value)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, ]).item(
    ) + 1j * np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorLessEqual(
        cArray, value), data <= value)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorLessEqual(
        value, cArray), value <= data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, ]).item(
    ) + 1j * np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorLessEqual(
        value, cArray), value <= data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorLessEqual(
        cArray1, cArray2), data1 <= data2)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    real2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data1 = real1 + 1j * imag1
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorLessEqual(
        cArray1, cArray2), data1 <= data2)


####################################################################################
def test_greater_equal():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorGreaterEqual(
        cArray, value), data >= value)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, ]).item(
    ) + 1j * np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorGreaterEqual(
        cArray, value), data >= value)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = np.random.randint(1, 100, [1, ]).astype(np.uint32).item()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorGreaterEqual(
        value, cArray), value >= data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, ]).item(
    ) + 1j * np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorGreaterEqual(
        value, cArray), value >= data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorGreaterEqual(
        cArray1, cArray2), data1 >= data2)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayComplexDouble(shape)
    cArray2 = NumCpp.NdArrayComplexDouble(shape)
    real1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    real2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag1 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag2 = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data1 = real1 + 1j * imag1
    data2 = real2 + 1j * imag2
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorGreaterEqual(
        cArray1, cArray2), data1 >= data2)


####################################################################################
def test_plus_plus():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorPostPlusPlus(cArray), data)
    assert np.array_equal(cArray.getNumpyArray(), data + 1)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorPrePlusPlus(cArray), data + 1)


####################################################################################
def test_minus_minus():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorPostMinusMinus(cArray), data)
    assert np.array_equal(cArray.getNumpyArray(), data - 1)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorPreMinusMinus(cArray), data - 1)


####################################################################################
def test_modulus():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorModulusScaler(
        cArray, randScaler), data % randScaler)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    randScaler = float(np.random.randint(1, 100, [1, ]).item())
    assert np.array_equal(NumCpp.operatorModulusScaler(
        cArray, randScaler), data % randScaler)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorModulusScaler(
        randScaler, cArray), randScaler % data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    randScaler = float(np.random.randint(1, 100, [1, ]).item())
    assert np.array_equal(NumCpp.operatorModulusScaler(
        randScaler, cArray), randScaler % data)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt32(shape)
    cArray2 = NumCpp.NdArrayUInt32(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorModulusArray(
        cArray1, cArray2), data1 % data2)

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorModulusArray(
        cArray1, cArray2), data1 % data2)


####################################################################################
def test_bitwise_or():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorBitwiseOrScaler(
        cArray, randScaler), np.bitwise_or(data, randScaler))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorBitwiseOrScaler(
        randScaler, cArray), np.bitwise_or(randScaler, data))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt32(shape)
    cArray2 = NumCpp.NdArrayUInt32(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorBitwiseOrArray(
        cArray1, cArray2), np.bitwise_or(data1, data2))


####################################################################################
def test_bitwise_and():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorBitwiseAndScaler(
        cArray, randScaler), np.bitwise_and(data, randScaler))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorBitwiseAndScaler(
        randScaler, cArray), np.bitwise_and(randScaler, data))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt32(shape)
    cArray2 = NumCpp.NdArrayUInt32(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorBitwiseAndArray(
        cArray1, cArray2), np.bitwise_and(data1, data2))


####################################################################################
def test_bitwise_xor():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorBitwiseXorScaler(
        cArray, randScaler), np.bitwise_xor(data, randScaler))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorBitwiseXorScaler(
        randScaler, cArray), np.bitwise_xor(randScaler, data))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt32(shape)
    cArray2 = NumCpp.NdArrayUInt32(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorBitwiseXorArray(
        cArray1, cArray2), np.bitwise_xor(data1, data2))


####################################################################################
def test_bitwise_not():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorBitwiseNot(cArray), ~data)


####################################################################################
def test_logical_and():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt32(shape)
    cArray2 = NumCpp.NdArrayUInt32(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorLogicalAndArray(
        cArray1, cArray2), np.logical_and(data1, data2))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScalar = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorLogicalAndScalar(
        cArray, randScalar), np.logical_and(data, randScalar))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScalar = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorLogicalAndScalar(
        randScalar, cArray), np.logical_and(randScalar, data))


####################################################################################
def test_logical_or():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArrayUInt32(shape)
    cArray2 = NumCpp.NdArrayUInt32(shape)
    data1 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorLogicalOrArray(
        cArray1, cArray2), np.logical_or(data1, data2))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScalar = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorLogicalOrScalar(
        cArray, randScalar), np.logical_or(data, randScalar))

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScalar = np.random.randint(1, 100, [1, ]).item()
    assert np.array_equal(NumCpp.operatorLogicalOrScalar(
        randScalar, cArray), np.logical_or(randScalar, data))


####################################################################################
def test_not():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorNot(cArray), np.logical_not(data))


####################################################################################
def test_bitshift_left():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 10, [1, ]).item()
    assert np.array_equal(NumCpp.operatorBitshiftLeft(
        cArray, randScaler), data << randScaler)


####################################################################################
def test_bitshift_right():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(
        1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randScaler = np.random.randint(1, 10, [1, ]).item()
    assert np.array_equal(NumCpp.operatorBitshiftRight(
        cArray, randScaler), data >> randScaler)
