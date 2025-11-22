import numpy as np

import NumCppPy as NumCpp  # noqa E402


####################################################################################
def test_seed():
    np.random.seed(357)


####################################################################################
def test_fft():
    # real input, axis none, default n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(
        np.round(NumCpp.fft(cArray, NumCpp.Axis.NONE), 6).flatten(), np.round(np.fft.fft(data.flatten()), 6)
    )

    # real input, axis none, smaller n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    n = np.random.randint(1, data.size)
    assert np.array_equal(
        np.round(NumCpp.fft(cArray, n, NumCpp.Axis.NONE), 6).flatten(), np.round(np.fft.fft(data.flatten(), n), 6)
    )

    # real input, axis none, larger n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    n = np.random.randint(data.size, data.size + 20)
    assert np.array_equal(
        np.round(NumCpp.fft(cArray, n, NumCpp.Axis.NONE), 6).flatten(), np.round(np.fft.fft(data.flatten(), n), 6)
    )

    # complex input, axis none, default n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(
        np.round(NumCpp.fft(cArray, NumCpp.Axis.NONE), 6).flatten(), np.round(np.fft.fft(data.flatten()), 6)
    )

    # complex input, axis none, smaller n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    n = np.random.randint(1, data.size)
    assert np.array_equal(
        np.round(NumCpp.fft(cArray, n, NumCpp.Axis.NONE), 6).flatten(), np.round(np.fft.fft(data.flatten(), n), 6)
    )

    # complex input, axis none, larger n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    n = np.random.randint(data.size, data.size + 20)
    assert np.array_equal(
        np.round(NumCpp.fft(cArray, n, NumCpp.Axis.NONE), 6).flatten(), np.round(np.fft.fft(data.flatten(), n), 6)
    )

    # real input, axis row, default n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.fft(cArray, NumCpp.Axis.ROW), 6), np.round(np.fft.fft(data, axis=0), 6))

    # real input, axis row, smaller n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    n = np.random.randint(1, shape.rows)
    assert np.array_equal(np.round(NumCpp.fft(cArray, n, NumCpp.Axis.ROW), 6), np.round(np.fft.fft(data, n, axis=0), 6))

    # real input, axis row, larger n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    n = np.random.randint(shape.rows, shape.rows + 20)
    assert np.array_equal(np.round(NumCpp.fft(cArray, NumCpp.Axis.ROW), 6), np.round(np.fft.fft(data, axis=0), 6))

    # complex input, axis row, default n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.fft(cArray, NumCpp.Axis.ROW), 6), np.round(np.fft.fft(data, axis=0), 6))

    # complex input, axis row, smaller n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    n = np.random.randint(1, shape.rows)
    assert np.array_equal(np.round(NumCpp.fft(cArray, n, NumCpp.Axis.ROW), 6), np.round(np.fft.fft(data, n, axis=0), 6))

    # complex input, axis row, larger n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    n = np.random.randint(shape.rows, shape.rows + 20)
    assert np.array_equal(np.round(NumCpp.fft(cArray, n, NumCpp.Axis.ROW), 6), np.round(np.fft.fft(data, n, axis=0), 6))

    # real input, axis col, default n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.fft(cArray, NumCpp.Axis.COL), 6), np.round(np.fft.fft(data, axis=1), 6))

    # real input, axis col, smaller n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    n = np.random.randint(1, shape.cols)
    assert np.array_equal(np.round(NumCpp.fft(cArray, n, NumCpp.Axis.COL), 6), np.round(np.fft.fft(data, n, axis=1), 6))

    # real input, axis col, larger n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    n = np.random.randint(shape.cols, shape.cols + 20)
    assert np.array_equal(np.round(NumCpp.fft(cArray, n, NumCpp.Axis.COL), 6), np.round(np.fft.fft(data, n, axis=1), 6))

    # complex input, axis col, default n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.fft(cArray, NumCpp.Axis.COL), 6), np.round(np.fft.fft(data, axis=1), 6))

    # complex input, axis col, smaller n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    n = np.random.randint(1, shape.cols)
    assert np.array_equal(np.round(NumCpp.fft(cArray, n, NumCpp.Axis.COL), 6), np.round(np.fft.fft(data, n, axis=1), 6))

    # complex input, axis col, larger n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    n = np.random.randint(shape.cols, shape.cols + 20)
    assert np.array_equal(np.round(NumCpp.fft(cArray, n, NumCpp.Axis.COL), 6), np.round(np.fft.fft(data, n, axis=1), 6))

####################################################################################
def test_ifft():
    # real input, axis none, default n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(
        np.round(NumCpp.ifft(cArray, NumCpp.Axis.NONE), 6).flatten(), np.round(np.fft.ifft(data.flatten()), 6)
    )

    # real input, axis none, smaller n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    n = np.random.randint(1, data.size)
    assert np.array_equal(
        np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.NONE), 6).flatten(), np.round(np.fft.ifft(data.flatten(), n), 6)
    )

    # real input, axis none, larger n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    n = np.random.randint(data.size, data.size + 20)
    assert np.array_equal(
        np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.NONE), 6).flatten(), np.round(np.fft.ifft(data.flatten(), n), 6)
    )

    # complex input, axis none, default n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(
        np.round(NumCpp.ifft(cArray, NumCpp.Axis.NONE), 6).flatten(), np.round(np.fft.ifft(data.flatten()), 6)
    )

    # complex input, axis none, smaller n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    n = np.random.randint(1, data.size)
    assert np.array_equal(
        np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.NONE), 6).flatten(), np.round(np.fft.ifft(data.flatten(), n), 6)
    )

    # complex input, axis none, larger n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    n = np.random.randint(data.size, data.size + 20)
    assert np.array_equal(
        np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.NONE), 6).flatten(), np.round(np.fft.ifft(data.flatten(), n), 6)
    )

    # real input, axis row, default n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.ifft(cArray, NumCpp.Axis.ROW), 6), np.round(np.fft.ifft(data, axis=0), 6))

    # real input, axis row, smaller n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    n = np.random.randint(1, shape.rows)
    assert np.array_equal(np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.ROW), 6), np.round(np.fft.ifft(data, n, axis=0), 6))

    # real input, axis row, larger n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    n = np.random.randint(shape.rows, shape.rows + 20)
    assert np.array_equal(np.round(NumCpp.ifft(cArray, NumCpp.Axis.ROW), 6), np.round(np.fft.ifft(data, axis=0), 6))

    # complex input, axis row, default n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.ifft(cArray, NumCpp.Axis.ROW), 6), np.round(np.fft.ifft(data, axis=0), 6))

    # complex input, axis row, smaller n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    n = np.random.randint(1, shape.rows)
    assert np.array_equal(np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.ROW), 6), np.round(np.fft.ifft(data, n, axis=0), 6))

    # complex input, axis row, larger n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    n = np.random.randint(shape.rows, shape.rows + 20)
    assert np.array_equal(np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.ROW), 6), np.round(np.fft.ifft(data, n, axis=0), 6))

    # real input, axis col, default n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.ifft(cArray, NumCpp.Axis.COL), 6), np.round(np.fft.ifft(data, axis=1), 6))

    # real input, axis col, smaller n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    n = np.random.randint(1, shape.cols)
    assert np.array_equal(np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.COL), 6), np.round(np.fft.ifft(data, n, axis=1), 6))

    # real input, axis col, larger n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    n = np.random.randint(shape.cols, shape.cols + 20)
    assert np.array_equal(np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.COL), 6), np.round(np.fft.ifft(data, n, axis=1), 6))

    # complex input, axis col, default n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    assert np.array_equal(np.round(NumCpp.ifft(cArray, NumCpp.Axis.COL), 6), np.round(np.fft.ifft(data, axis=1), 6))

    # complex input, axis col, smaller n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    n = np.random.randint(1, shape.cols)
    assert np.array_equal(np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.COL), 6), np.round(np.fft.ifft(data, n, axis=1), 6))

    # complex input, axis col, larger n
    shapeInput = np.random.randint(
        10,
        30,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArrayComplexDouble(shape)
    real = np.random.randint(1, 100, [shape.rows, shape.cols])
    imag = np.random.randint(1, 100, [shape.rows, shape.cols])
    data = real + 1j * imag
    cArray.setArray(data)
    n = np.random.randint(shape.cols, shape.cols + 20)
    assert np.array_equal(np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.COL), 6), np.round(np.fft.ifft(data, n, axis=1), 6))
