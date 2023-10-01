import numpy as np

import NumCppPy as NumCpp  # noqa E402

np.random.seed(666)


####################################################################################
def test_plus_equal():
    # (1) Arithmetic Arrays
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhsC), lhs + rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhsC), lhs + rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhsC), lhs + rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhsC), lhs + rhs)

    # (1) Complex Arrays
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhsC), lhs + rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float) + 1j * np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhsC), lhs + rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhsC), lhs + rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhsC), lhs + rhs)

    # (2) Complex Array, Arithmetic Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhsC), lhs + rhs)

    # (3) Arithmetic Array, Arithmetic scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhs), lhs + rhs)

    # (3) Complex Array, Complex scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhs), lhs + rhs)

    # (4) Complex Array, Arithmetic scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlusEqual(lhsC, rhs), lhs + rhs)


####################################################################################
def test_plus():
    # (1) Arithmetic Arrays
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArray(1, shape.cols)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # (1) Complex Arrays
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float) + 1j * np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # (2) Arithmetic Array, Complex Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float) + 1j * np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArray(1, shape.cols)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # (3) Complex Array, Arithmetic Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhsC), lhs + rhs)

    # (4) Arithmetic Array, Arithmetic scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhs), lhs + rhs)

    # (4) Complex Array, Complex scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhs), lhs + rhs)

    # (5) Arithmetic scalar, Arithmetic Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    rhsC = NumCpp.NdArray(shape)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhs = float(np.random.randint(0, 100))
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhs, rhsC), lhs + rhs)

    # (5) Complex scalar, Complex Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    lhs = float(np.random.randint(0, 100)) + 1j * float(np.random.randint(0, 100))
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorPlus(lhs, rhsC), lhs + rhs)

    # (6) Arithmetic Array, Complex scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhs), lhs + rhs)

    # (7) Complex scalar, Arithmetic Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlus(rhs, lhsC), rhs + lhs)

    # (8) Complex Array, Arithmetic scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlus(lhsC, rhs), lhs + rhs)

    # (9) Arithmetic scalar, Complex Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorPlus(rhs, lhsC), rhs + lhs)


####################################################################################
def test_negative():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorNegative(cArray), -data)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
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
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhsC), lhs - rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhsC), lhs - rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhsC), lhs - rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhsC), lhs - rhs)

    # (1) Complex Arrays
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhsC), lhs - rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float) + 1j * np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhsC), lhs - rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhsC), lhs - rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhsC), lhs - rhs)

    # (2) Complex Array, Arithmetic Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhsC), lhs - rhs)

    # (3) Arithmetic Array, Arithmetic scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhs), lhs - rhs)

    # (3) Complex Array, Complex scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhs), lhs - rhs)

    # (4) Complex Array, Arithmetic scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinusEqual(lhsC, rhs), lhs - rhs)


####################################################################################
def test_minus():
    # (1) Arithmetic Arrays
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArray(1, shape.cols)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # (1) Complex Arrays
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float) + 1j * np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # (2) Arithmetic Array, Complex Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float) + 1j * np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArray(1, shape.cols)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # (3) Complex Array, Arithmetic Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhsC), lhs - rhs)

    # (4) Arithmetic Array, Arithmetic scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhs), lhs - rhs)

    # (4) Complex Array, Complex scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhs), lhs - rhs)

    # (5) Arithmetic scalar, Arithmetic Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    rhsC = NumCpp.NdArray(shape)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhs = float(np.random.randint(0, 100))
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhs, rhsC), lhs - rhs)

    # (5) Complex scalar, Complex Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    lhs = float(np.random.randint(0, 100)) + 1j * float(np.random.randint(0, 100))
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMinus(lhs, rhsC), lhs - rhs)

    # (6) Arithmetic Array, Complex scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhs), lhs - rhs)

    # (7) Complex scalar, Arithmetic Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinus(rhs, lhsC), rhs - lhs)

    # (8) Complex Array, Arithmetic scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinus(lhsC, rhs), lhs - rhs)

    # (9) Arithmetic scalar, Complex Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMinus(rhs, lhsC), rhs - lhs)


####################################################################################
def test_multiply_equal():
    # (1) Arithmetic Arrays
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhsC), lhs * rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhsC), lhs * rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhsC), lhs * rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhsC), lhs * rhs)

    # (1) Complex Arrays
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhsC), lhs * rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float) + 1j * np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhsC), lhs * rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhsC), lhs * rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhsC), lhs * rhs)

    # (2) Complex Array, Arithmetic Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhsC), lhs * rhs)

    # (3) Arithmetic Array, Arithmetic scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhs), lhs * rhs)

    # (3) Complex Array, Complex scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhs), lhs * rhs)

    # (4) Complex Array, Arithmetic scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiplyEqual(lhsC, rhs), lhs * rhs)


####################################################################################
def test_multiply():
    # (1) Arithmetic Arrays
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArray(1, shape.cols)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # (1) Complex Arrays
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float) + 1j * np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # (2) Arithmetic Array, Complex Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float) + 1j * np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArray(1, shape.cols)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # (3) Complex Array, Arithmetic Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    rhs = np.random.randint(0, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhsC), lhs * rhs)

    # (4) Arithmetic Array, Arithmetic scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhs), lhs * rhs)

    # (4) Complex Array, Complex scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(0, 100)) * 1j * float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhs), lhs * rhs)

    # (5) Arithmetic scalar, Arithmetic Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    rhsC = NumCpp.NdArray(shape)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    lhs = float(np.random.randint(0, 100))
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhs, rhsC), lhs * rhs)

    # (5) Complex scalar, Complex Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    rhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    lhs = float(np.random.randint(0, 100)) + 1j * float(np.random.randint(0, 100))
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhs, rhsC), lhs * rhs)

    # (6) Arithmetic Array, Complex scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhs), lhs * rhs)

    # (7) Complex scalar, Arithmetic Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(0, 100)) + 1j * float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiply(rhs, lhsC), rhs * lhs)

    # (8) Complex Array, Arithmetic scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiply(lhsC, rhs), lhs * rhs)

    # (9) Arithmetic scalar, Complex Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(0, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorMultiply(rhs, lhsC), rhs * lhs)


####################################################################################
def test_divide_equal():
    # (1) Arithmetic Arrays
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [1, shape.cols]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # (1) Complex Arrays
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        1, 100, [shape.rows, shape.cols]
    ).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(1, 100, [1, 1]).astype(float) + 1j * np.random.randint(1, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(1, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        1, 100, [1, shape.cols]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        1, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # (2) Complex Array, Arithmetic Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # (3) Arithmetic Array, Arithmetic scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(1, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhs), 8), np.round(lhs / rhs, 8))

    # (3) Complex Array, Complex scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(1, 100)) + 1j * float(np.random.randint(1, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhs), 8), np.round(lhs / rhs, 8))

    # (4) Complex Array, Arithmetic scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(1, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivideEqual(lhsC, rhs), 8), np.round(lhs / rhs, 8))


####################################################################################
def test_divide():
    # (1) Arithmetic Arrays
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [1, shape.cols]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [1, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArray(1, shape.cols)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # (1) Complex Arrays
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        1, 100, [shape.rows, shape.cols]
    ).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(1, 100, [1, 1]).astype(float) + 1j * np.random.randint(1, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(1, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        1, 100, [1, shape.cols]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        1, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        1, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # (2) Arithmetic Array, Complex Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        1, 100, [shape.rows, shape.cols]
    ).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [1, 1]).astype(float) + 1j * np.random.randint(1, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        1, 100, [1, shape.cols]
    ).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        1, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [1, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, 1]).astype(float) + 1j * np.random.randint(
        1, 100, [shape.rows, 1]
    ).astype(float)
    lhsC = NumCpp.NdArray(1, shape.cols)
    rhsC = NumCpp.NdArrayComplexDouble(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # (3) Complex Array, Arithmetic Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(1, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(1, 100, [1, shape.cols]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [1, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [1, shape.cols]
    ).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArrayComplexDouble(1, shape.cols)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhsC), 8), np.round(lhs / rhs, 8))

    # (4) Arithmetic Array, Arithmetic scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(1, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhs), 8), np.round(lhs / rhs, 8))

    # (4) Complex Array, Complex scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(1, 100)) + 1j * float(np.random.randint(1, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhs), 8), np.round(lhs / rhs, 8))

    # (5) Arithmetic scalar, Arithmetic Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    rhsC = NumCpp.NdArray(shape)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    lhs = float(np.random.randint(1, 100))
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhs, rhsC), 8), np.round(lhs / rhs, 8))

    # (5) Complex scalar, Complex Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    rhsC = NumCpp.NdArrayComplexDouble(shape)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    lhs = float(np.random.randint(1, 100)) + 1j * float(np.random.randint(1, 100))
    rhsC.setArray(rhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhs, rhsC), 8), np.round(lhs / rhs, 8))

    # (6) Arithmetic Array, Complex scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(1, 100)) + 1j * float(np.random.randint(1, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhs), 8), np.round(lhs / rhs, 8))

    # (7) Complex scalar, Arithmetic Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = float(np.random.randint(1, 100)) + 1j * float(np.random.randint(1, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(rhs, lhsC), 8), np.round(rhs / lhs, 8))

    # (8) Complex Array, Arithmetic scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        0, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(1, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(lhsC, rhs), 8), np.round(lhs / rhs, 8))

    # (9) Arithmetic scalar, Complex Array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayComplexDouble(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float) + 1j * np.random.randint(
        1, 100, [shape.rows, shape.cols]
    ).astype(float)
    rhs = float(np.random.randint(1, 100))
    lhsC.setArray(lhs)
    assert np.array_equal(np.round(NumCpp.operatorDivide(rhs, lhsC), 8), np.round(rhs / lhs, 8))


####################################################################################
def test_equality():
    # array scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = (
        np.random.randint(
            1,
            100,
            [
                1,
            ],
        )
        .astype(np.uint32)
        .item()
    )
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorEquality(cArray, value), data == value)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(
        1,
        100,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.operatorEquality(cArray, value), data == value)

    # scalar array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = (
        np.random.randint(
            1,
            100,
            [
                1,
            ],
        )
        .astype(np.uint32)
        .item()
    )
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorEquality(value, cArray), value == data)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(
        1,
        100,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.operatorEquality(value, cArray), value == data)

    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorEquality(cArray1, cArray2), data1 == data2)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
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
    assert np.array_equal(NumCpp.operatorEquality(cArray1, cArray2), data1 == data2)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, 1]).astype(np.uint32)
    cValue = NumCpp.NdArray(*value.shape)
    cValue.setArray(value)
    assert np.array_equal(NumCpp.operatorEquality(cArray, cValue), data == value)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, 1])
    cValue = NumCpp.NdArrayComplexDouble(*value.shape)
    cValue.setArray(value)
    assert np.array_equal(NumCpp.operatorEquality(cArray, cValue), data == value)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [1, shape.cols]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorEquality(cArray, cValues), data == values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [1, shape.cols])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorEquality(cArray, cValues), data == values)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorEquality(cArray, cValues), data == values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorEquality(cArray, cValues), data == values)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [1, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorEquality(cArray, cValues), data == values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorEquality(cArray, cValues), data == values)


####################################################################################
def test_not_equality():
    # array scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = (
        np.random.randint(
            1,
            100,
            [
                1,
            ],
        )
        .astype(np.uint32)
        .item()
    )
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorNotEquality(cArray, value), data != value)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(
        1,
        100,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.operatorNotEquality(cArray, value), data != value)

    # scalar array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = (
        np.random.randint(
            1,
            100,
            [
                1,
            ],
        )
        .astype(np.uint32)
        .item()
    )
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorNotEquality(value, cArray), value != data)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(
        1,
        100,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.operatorNotEquality(value, cArray), value != data)

    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorNotEquality(cArray1, cArray2), data1 != data2)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
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
    assert np.array_equal(NumCpp.operatorNotEquality(cArray1, cArray2), data1 != data2)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, 1]).astype(np.uint32)
    cValue = NumCpp.NdArray(*value.shape)
    cValue.setArray(value)
    assert np.array_equal(NumCpp.operatorNotEquality(cArray, cValue), data != value)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, 1])
    cValue = NumCpp.NdArrayComplexDouble(*value.shape)
    cValue.setArray(value)
    assert np.array_equal(NumCpp.operatorNotEquality(cArray, cValue), data != value)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [1, shape.cols]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorNotEquality(cArray, cValues), data != values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [1, shape.cols])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorNotEquality(cArray, cValues), data != values)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorNotEquality(cArray, cValues), data != values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorNotEquality(cArray, cValues), data != values)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [1, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorNotEquality(cArray, cValues), data != values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorNotEquality(cArray, cValues), data != values)


####################################################################################
def test_less():
    # array scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = (
        np.random.randint(
            1,
            100,
            [
                1,
            ],
        )
        .astype(np.uint32)
        .item()
    )
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorLess(cArray, value), data < value)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(
        1,
        100,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.operatorLess(cArray, value), data < value)

    # scalar array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = (
        np.random.randint(
            1,
            100,
            [
                1,
            ],
        )
        .astype(np.uint32)
        .item()
    )
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorLess(value, cArray), value < data)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(
        1,
        100,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.operatorLess(value, cArray), value < data)

    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorLess(cArray1, cArray2), data1 < data2)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
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

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, 1]).astype(np.uint32)
    cValue = NumCpp.NdArray(*value.shape)
    cValue.setArray(value)
    assert np.array_equal(NumCpp.operatorLess(cArray, cValue), data < value)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, 1])
    cValue = NumCpp.NdArrayComplexDouble(*value.shape)
    cValue.setArray(value)
    assert np.array_equal(NumCpp.operatorLess(cArray, cValue), data < value)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [1, shape.cols]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorLess(cArray, cValues), data < values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [1, shape.cols])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorLess(cArray, cValues), data < values)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorLess(cArray, cValues), data < values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorLess(cArray, cValues), data < values)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [1, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorLess(cArray, cValues), data < values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorLess(cArray, cValues), data < values)


####################################################################################
def test_greater():
    # array scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = (
        np.random.randint(
            1,
            100,
            [
                1,
            ],
        )
        .astype(np.uint32)
        .item()
    )
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorGreater(cArray, value), data > value)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(
        1,
        100,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.operatorGreater(cArray, value), data > value)

    # scalar array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = (
        np.random.randint(
            1,
            100,
            [
                1,
            ],
        )
        .astype(np.uint32)
        .item()
    )
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorGreater(value, cArray), value > data)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(
        1,
        100,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.operatorGreater(value, cArray), value > data)

    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorGreater(cArray1, cArray2), data1 > data2)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
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
    assert np.array_equal(NumCpp.operatorGreater(cArray1, cArray2), data1 > data2)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, 1]).astype(np.uint32)
    cValue = NumCpp.NdArray(*value.shape)
    cValue.setArray(value)
    assert np.array_equal(NumCpp.operatorGreater(cArray, cValue), data > value)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, 1])
    cValue = NumCpp.NdArrayComplexDouble(*value.shape)
    cValue.setArray(value)
    assert np.array_equal(NumCpp.operatorGreater(cArray, cValue), data > value)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [1, shape.cols]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorGreater(cArray, cValues), data > values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [1, shape.cols])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorGreater(cArray, cValues), data > values)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorGreater(cArray, cValues), data > values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorGreater(cArray, cValues), data > values)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [1, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorGreater(cArray, cValues), data > values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorGreater(cArray, cValues), data > values)


####################################################################################
def test_less_equal():
    # array scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = (
        np.random.randint(
            1,
            100,
            [
                1,
            ],
        )
        .astype(np.uint32)
        .item()
    )
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorLessEqual(cArray, value), data <= value)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(
        1,
        100,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.operatorLessEqual(cArray, value), data <= value)

    # scalar array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = (
        np.random.randint(
            1,
            100,
            [
                1,
            ],
        )
        .astype(np.uint32)
        .item()
    )
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorLessEqual(value, cArray), value <= data)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(
        1,
        100,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.operatorLessEqual(value, cArray), value <= data)

    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorLessEqual(cArray1, cArray2), data1 <= data2)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
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
    assert np.array_equal(NumCpp.operatorLessEqual(cArray1, cArray2), data1 <= data2)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, 1]).astype(np.uint32)
    cValue = NumCpp.NdArray(*value.shape)
    cValue.setArray(value)
    assert np.array_equal(NumCpp.operatorLessEqual(cArray, cValue), data <= value)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, 1])
    cValue = NumCpp.NdArrayComplexDouble(*value.shape)
    cValue.setArray(value)
    assert np.array_equal(NumCpp.operatorLessEqual(cArray, cValue), data <= value)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [1, shape.cols]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorLessEqual(cArray, cValues), data <= values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [1, shape.cols])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorLessEqual(cArray, cValues), data <= values)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorLessEqual(cArray, cValues), data <= values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorLessEqual(cArray, cValues), data <= values)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [1, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorLessEqual(cArray, cValues), data <= values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorLessEqual(cArray, cValues), data <= values)


####################################################################################
def test_greater_equal():
    # array scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = (
        np.random.randint(
            1,
            100,
            [
                1,
            ],
        )
        .astype(np.uint32)
        .item()
    )
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorGreaterEqual(cArray, value), data >= value)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(
        1,
        100,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.operatorGreaterEqual(cArray, value), data >= value)

    # scalar array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    value = (
        np.random.randint(
            1,
            100,
            [
                1,
            ],
        )
        .astype(np.uint32)
        .item()
    )
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorGreaterEqual(value, cArray), value >= data)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray.setArray(data)
    value = np.random.randint(
        1,
        100,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.operatorGreaterEqual(value, cArray), value >= data)

    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray1 = NumCpp.NdArray(shape)
    cArray2 = NumCpp.NdArray(shape)
    data1 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    data2 = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    assert np.array_equal(NumCpp.operatorGreaterEqual(cArray1, cArray2), data1 >= data2)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
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
    assert np.array_equal(NumCpp.operatorGreaterEqual(cArray1, cArray2), data1 >= data2)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, 1]).astype(np.uint32)
    cValue = NumCpp.NdArray(*value.shape)
    cValue.setArray(value)
    assert np.array_equal(NumCpp.operatorGreaterEqual(cArray, cValue), data >= value)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    value = np.random.randint(1, 100, [1, 1])
    cValue = NumCpp.NdArrayComplexDouble(*value.shape)
    cValue.setArray(value)
    assert np.array_equal(NumCpp.operatorGreaterEqual(cArray, cValue), data >= value)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [1, shape.cols]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorGreaterEqual(cArray, cValues), data >= values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [1, shape.cols])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorGreaterEqual(cArray, cValues), data >= values)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorGreaterEqual(cArray, cValues), data >= values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [shape.rows, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorGreaterEqual(cArray, cValues), data >= values)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    data = np.random.randint(1, 100, [1, shape.cols]).astype(np.uint32)
    cArray = NumCpp.NdArray(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1]).astype(np.uint32)
    cValues = NumCpp.NdArray(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorGreaterEqual(cArray, cValues), data >= values)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    real = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    imag = np.random.randint(0, 100, [1, shape.cols]).astype(float)
    data = real + 1j * imag
    cArray = NumCpp.NdArrayComplexDouble(shape)
    cArray.setArray(data)
    values = np.random.randint(1, 100, [shape.rows, 1])
    cValues = NumCpp.NdArrayComplexDouble(*values.shape)
    cValues.setArray(values)
    assert np.array_equal(NumCpp.operatorGreaterEqual(cArray, cValues), data >= values)


####################################################################################
def test_plus_plus():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorPostPlusPlus(cArray), data)
    assert np.array_equal(cArray.getNumpyArray(), data + 1)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorPrePlusPlus(cArray), data + 1)


####################################################################################
def test_minus_minus():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorPostMinusMinus(cArray), data)
    assert np.array_equal(cArray.getNumpyArray(), data - 1)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorPreMinusMinus(cArray), data - 1)


####################################################################################
def test_modulus_equal():
    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusEqualArray(lhsC, rhsC), lhs % rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusEqualArray(lhsC, rhsC), lhs % rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, shape.cols], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusEqualArray(lhsC, rhsC), lhs % rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusEqualArray(lhsC, rhsC), lhs % rhs)

    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusEqualArray(lhsC, rhsC), lhs % rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusEqualArray(lhsC, rhsC), lhs % rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [1, shape.cols]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusEqualArray(lhsC, rhsC), lhs % rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusEqualArray(lhsC, rhsC), lhs % rhs)


####################################################################################
def test_modulus():
    # array scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    randscalar = np.random.randint(
        1,
        100,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.operatorModulusScalar(cArray, randscalar), data % randscalar)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    randscalar = float(
        np.random.randint(
            1,
            100,
            [
                1,
            ],
        ).item()
    )
    assert np.array_equal(NumCpp.operatorModulusScalar(cArray, randscalar), data % randscalar)

    # scalar array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    cArray.setArray(data)
    randscalar = np.random.randint(
        1,
        100,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.operatorModulusScalar(randscalar, cArray), randscalar % data)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    randscalar = float(
        np.random.randint(
            1,
            100,
            [
                1,
            ],
        ).item()
    )
    assert np.array_equal(NumCpp.operatorModulusScalar(randscalar, cArray), randscalar % data)

    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusArray(lhsC, rhsC), lhs % rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusArray(lhsC, rhsC), lhs % rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, shape.cols], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusArray(lhsC, rhsC), lhs % rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusArray(lhsC, rhsC), lhs % rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [1, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(1, shape.cols)
    rhsC = NumCpp.NdArrayUInt32(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusArray(lhsC, rhsC), lhs % rhs)

    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusArray(lhsC, rhsC), lhs % rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [1, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusArray(lhsC, rhsC), lhs % rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [1, shape.cols]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusArray(lhsC, rhsC), lhs % rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArray(shape)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusArray(lhsC, rhsC), lhs % rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [1, shape.cols]).astype(float)
    rhs = np.random.randint(1, 100, [shape.rows, 1]).astype(float)
    lhsC = NumCpp.NdArray(1, shape.cols)
    rhsC = NumCpp.NdArray(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorModulusArray(lhsC, rhsC), lhs % rhs)


####################################################################################
def test_bitwise_or_equal():
    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseOrEqualArray(lhsC, rhsC), lhs | rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseOrEqualArray(lhsC, rhsC), lhs | rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, shape.cols], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseOrEqualArray(lhsC, rhsC), lhs | rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseOrEqualArray(lhsC, rhsC), lhs | rhs)


####################################################################################
def test_bitwise_or():
    # array scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(
        1,
        100,
        [
            1,
        ],
        dtype=np.uint32,
    ).item()
    lhsC = NumCpp.NdArrayUInt32(shape)
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorBitwiseOrScalar(lhsC, rhs), lhs | rhs)

    # scalar array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(
        1,
        100,
        [
            1,
        ],
        dtype=np.uint32,
    ).item()
    lhsC = NumCpp.NdArrayUInt32(shape)
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorBitwiseOrScalar(rhs, lhsC), lhs | rhs)

    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseOrArray(lhsC, rhsC), lhs | rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseOrArray(lhsC, rhsC), lhs | rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, shape.cols], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseOrArray(lhsC, rhsC), lhs | rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseOrArray(lhsC, rhsC), lhs | rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [1, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(1, shape.cols)
    rhsC = NumCpp.NdArrayUInt32(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseOrArray(lhsC, rhsC), lhs | rhs)


####################################################################################
def test_bitwise_and_equal():
    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseAndEqualArray(lhsC, rhsC), lhs & rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseAndEqualArray(lhsC, rhsC), lhs & rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, shape.cols], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseAndEqualArray(lhsC, rhsC), lhs & rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseAndEqualArray(lhsC, rhsC), lhs & rhs)


####################################################################################
def test_bitwise_and():
    # array scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(
        1,
        100,
        [
            1,
        ],
        dtype=np.uint32,
    ).item()
    lhsC = NumCpp.NdArrayUInt32(shape)
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorBitwiseAndScalar(lhsC, rhs), lhs & rhs)

    # scalar array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(
        1,
        100,
        [
            1,
        ],
        dtype=np.uint32,
    ).item()
    lhsC = NumCpp.NdArrayUInt32(shape)
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorBitwiseAndScalar(rhs, lhsC), lhs & rhs)

    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseAndArray(lhsC, rhsC), lhs & rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseAndArray(lhsC, rhsC), lhs & rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, shape.cols], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseAndArray(lhsC, rhsC), lhs & rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseAndArray(lhsC, rhsC), lhs & rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [1, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(1, shape.cols)
    rhsC = NumCpp.NdArrayUInt32(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseAndArray(lhsC, rhsC), lhs & rhs)


####################################################################################
def test_bitwise_xor_equal():
    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseXorEqualArray(lhsC, rhsC), lhs ^ rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseXorEqualArray(lhsC, rhsC), lhs ^ rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, shape.cols], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseXorEqualArray(lhsC, rhsC), lhs ^ rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseXorEqualArray(lhsC, rhsC), lhs ^ rhs)


####################################################################################
def test_bitwise_xor():
    # array scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(
        1,
        100,
        [
            1,
        ],
        dtype=np.uint32,
    ).item()
    lhsC = NumCpp.NdArrayUInt32(shape)
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorBitwiseXorScalar(lhsC, rhs), lhs ^ rhs)

    # scalar array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(
        1,
        100,
        [
            1,
        ],
        dtype=np.uint32,
    ).item()
    lhsC = NumCpp.NdArrayUInt32(shape)
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorBitwiseXorScalar(rhs, lhsC), lhs ^ rhs)

    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseXorArray(lhsC, rhsC), lhs ^ rhs)

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseXorArray(lhsC, rhsC), lhs ^ rhs)

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, shape.cols], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseXorArray(lhsC, rhsC), lhs ^ rhs)

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseXorArray(lhsC, rhsC), lhs ^ rhs)

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [1, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(1, shape.cols)
    rhsC = NumCpp.NdArrayUInt32(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorBitwiseXorArray(lhsC, rhsC), lhs ^ rhs)


####################################################################################
def test_bitwise_not():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorBitwiseNot(cArray), ~data)


####################################################################################
def test_logical_and():
    # array scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(
        1,
        100,
        [
            1,
        ],
        dtype=np.uint32,
    ).item()
    lhsC = NumCpp.NdArrayUInt32(shape)
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorLogicalAndScalar(lhsC, rhs), np.logical_and(lhs, rhs))

    # scalar array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(
        1,
        100,
        [
            1,
        ],
        dtype=np.uint32,
    ).item()
    lhsC = NumCpp.NdArrayUInt32(shape)
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorLogicalAndScalar(rhs, lhsC), np.logical_and(lhs, rhs))

    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorLogicalAndArray(lhsC, rhsC), np.logical_and(lhs, rhs))

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorLogicalAndArray(lhsC, rhsC), np.logical_and(lhs, rhs))

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, shape.cols], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorLogicalAndArray(lhsC, rhsC), np.logical_and(lhs, rhs))

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorLogicalAndArray(lhsC, rhsC), np.logical_and(lhs, rhs))

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [1, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(1, shape.cols)
    rhsC = NumCpp.NdArrayUInt32(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorLogicalAndArray(lhsC, rhsC), np.logical_and(lhs, rhs))


####################################################################################
def test_logical_or():
    # array scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(
        1,
        100,
        [
            1,
        ],
        dtype=np.uint32,
    ).item()
    lhsC = NumCpp.NdArrayUInt32(shape)
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorLogicalOrScalar(lhsC, rhs), np.logical_or(lhs, rhs))

    # scalar array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(
        1,
        100,
        [
            1,
        ],
        dtype=np.uint32,
    ).item()
    lhsC = NumCpp.NdArrayUInt32(shape)
    lhsC.setArray(lhs)
    assert np.array_equal(NumCpp.operatorLogicalOrScalar(rhs, lhsC), np.logical_or(lhs, rhs))

    # array array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape)
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorLogicalOrArray(lhsC, rhsC), np.logical_or(lhs, rhs))

    # broadcast scalar
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorLogicalOrArray(lhsC, rhsC), np.logical_or(lhs, rhs))

    # broadcast row array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [1, shape.cols], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(1, shape.cols)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorLogicalOrArray(lhsC, rhsC), np.logical_or(lhs, rhs))

    # broadcast col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [shape.rows, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(shape)
    rhsC = NumCpp.NdArrayUInt32(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorLogicalOrArray(lhsC, rhsC), np.logical_or(lhs, rhs))

    # broadcast row array col array
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    lhs = np.random.randint(1, 100, [1, shape.cols], dtype=np.uint32)
    rhs = np.random.randint(1, 100, [shape.rows, 1], dtype=np.uint32)
    lhsC = NumCpp.NdArrayUInt32(1, shape.cols)
    rhsC = NumCpp.NdArrayUInt32(shape.rows, 1)
    lhsC.setArray(lhs)
    rhsC.setArray(rhs)
    assert np.array_equal(NumCpp.operatorLogicalOrArray(lhsC, rhsC), np.logical_or(lhs, rhs))


####################################################################################
def test_not():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    assert np.array_equal(NumCpp.operatorNot(cArray), np.logical_not(data))


####################################################################################
def test_bitshift_left():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randscalar = np.random.randint(
        1,
        10,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.operatorBitshiftLeft(cArray, randscalar), data << randscalar)


####################################################################################
def test_bitshift_right():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayUInt32(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(np.uint32)
    cArray.setArray(data)
    randscalar = np.random.randint(
        1,
        10,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.operatorBitshiftRight(cArray, randscalar), data >> randscalar)
