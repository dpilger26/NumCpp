import numpy as np

import NumCppPy as NumCpp  # noqa E402

NUM_TRIALS = 5
ROUNDING_DIGITS = 5


####################################################################################
def test_seed():
    np.random.seed(357)


####################################################################################
def test_fft():
    for _ in range(NUM_TRIALS):
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
            np.round(NumCpp.fft(cArray, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.fft(data.flatten()), ROUNDING_DIGITS)
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
            np.round(NumCpp.fft(cArray, n, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.fft(data.flatten(), n), ROUNDING_DIGITS)
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
            np.round(NumCpp.fft(cArray, n, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.fft(data.flatten(), n), ROUNDING_DIGITS)
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
            np.round(NumCpp.fft(cArray, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.fft(data.flatten()), ROUNDING_DIGITS)
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
            np.round(NumCpp.fft(cArray, n, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.fft(data.flatten(), n), ROUNDING_DIGITS)
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
            np.round(NumCpp.fft(cArray, n, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.fft(data.flatten(), n), ROUNDING_DIGITS)
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
        assert np.array_equal(np.round(NumCpp.fft(cArray, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.fft(data, axis=0), ROUNDING_DIGITS))

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
        assert np.array_equal(np.round(NumCpp.fft(cArray, n, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.fft(data, n, axis=0), ROUNDING_DIGITS))

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
        assert np.array_equal(np.round(NumCpp.fft(cArray, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.fft(data, axis=0), ROUNDING_DIGITS))

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
        assert np.array_equal(np.round(NumCpp.fft(cArray, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.fft(data, axis=0), ROUNDING_DIGITS))

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
        assert np.array_equal(np.round(NumCpp.fft(cArray, n, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.fft(data, n, axis=0), ROUNDING_DIGITS))

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
        assert np.array_equal(np.round(NumCpp.fft(cArray, n, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.fft(data, n, axis=0), ROUNDING_DIGITS))

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
        assert np.array_equal(np.round(NumCpp.fft(cArray, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.fft(data, axis=1), ROUNDING_DIGITS))

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
        assert np.array_equal(np.round(NumCpp.fft(cArray, n, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.fft(data, n, axis=1), ROUNDING_DIGITS))

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
        assert np.array_equal(np.round(NumCpp.fft(cArray, n, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.fft(data, n, axis=1), ROUNDING_DIGITS))

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
        assert np.array_equal(np.round(NumCpp.fft(cArray, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.fft(data, axis=1), ROUNDING_DIGITS))

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
        assert np.array_equal(np.round(NumCpp.fft(cArray, n, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.fft(data, n, axis=1), ROUNDING_DIGITS))

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
        assert np.array_equal(np.round(NumCpp.fft(cArray, n, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.fft(data, n, axis=1), ROUNDING_DIGITS))


####################################################################################
def test_ifft():
    for _ in range(NUM_TRIALS):
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
            np.round(NumCpp.ifft(cArray, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.ifft(data.flatten()), ROUNDING_DIGITS)
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
            np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.ifft(data.flatten(), n), ROUNDING_DIGITS)
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
            np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.ifft(data.flatten(), n), ROUNDING_DIGITS)
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
            np.round(NumCpp.ifft(cArray, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.ifft(data.flatten()), ROUNDING_DIGITS)
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
            np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.ifft(data.flatten(), n), ROUNDING_DIGITS)
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
            np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.ifft(data.flatten(), n), ROUNDING_DIGITS)
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
        assert np.array_equal(np.round(NumCpp.ifft(cArray, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.ifft(data, axis=0), ROUNDING_DIGITS))

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
        assert np.array_equal(
            np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.ifft(data, n, axis=0), ROUNDING_DIGITS)
        )

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
        assert np.array_equal(np.round(NumCpp.ifft(cArray, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.ifft(data, axis=0), ROUNDING_DIGITS))

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
        assert np.array_equal(np.round(NumCpp.ifft(cArray, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.ifft(data, axis=0), ROUNDING_DIGITS))

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
        assert np.array_equal(
            np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.ifft(data, n, axis=0), ROUNDING_DIGITS)
        )

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
        assert np.array_equal(
            np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.ifft(data, n, axis=0), ROUNDING_DIGITS)
        )

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
        assert np.array_equal(np.round(NumCpp.ifft(cArray, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.ifft(data, axis=1), ROUNDING_DIGITS))

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
        assert np.array_equal(
            np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.ifft(data, n, axis=1), ROUNDING_DIGITS)
        )

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
        assert np.array_equal(
            np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.ifft(data, n, axis=1), ROUNDING_DIGITS)
        )

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
        assert np.array_equal(np.round(NumCpp.ifft(cArray, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.ifft(data, axis=1), ROUNDING_DIGITS))

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
        assert np.array_equal(
            np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.ifft(data, n, axis=1), ROUNDING_DIGITS)
        )

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
        assert np.array_equal(
            np.round(NumCpp.ifft(cArray, n, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.ifft(data, n, axis=1), ROUNDING_DIGITS)
        )


####################################################################################
def test_fftfreq():
    for _ in range(NUM_TRIALS):
        n = np.random.randint(10, 1000)
        d = np.random.rand()
        assert np.array_equal(np.round(NumCpp.fftfreq(n, d).flatten(), 8), np.round(np.fft.fftfreq(n, d), 8))


####################################################################################
def test_fftshift():
    for _ in range(NUM_TRIALS):
        n = np.random.randint(10, 1000)
        d = np.random.rand()
        freqs = np.fft.fftfreq(n, d)
        cFreqs = NumCpp.NdArray(1, freqs.size)
        cFreqs.setArray(freqs)
        assert np.array_equal(
            np.round(NumCpp.fftshift(cFreqs, NumCpp.Axis.NONE).flatten(), 8), np.round(np.fft.fftshift(freqs), 8)
        )

        dim0 = np.random.randint(10, 100)
        dim1 = np.random.randint(10, 100)
        n = dim0 * dim1
        d = np.random.rand()
        freqs = np.fft.fftfreq(n, d).reshape(dim0, dim1)
        cFreqs = NumCpp.NdArray(freqs.shape[0], freqs.shape[1])
        cFreqs.setArray(freqs)
        assert np.array_equal(
            np.round(NumCpp.fftshift(cFreqs, NumCpp.Axis.ROW), 8), np.round(np.fft.fftshift(freqs, axes=0), 8)
        )
        assert np.array_equal(
            np.round(NumCpp.fftshift(cFreqs, NumCpp.Axis.COL), 8), np.round(np.fft.fftshift(freqs, axes=1), 8)
        )


####################################################################################
def test_ifftshift():
    for _ in range(NUM_TRIALS):
        n = np.random.randint(10, 1000)
        d = np.random.rand()
        freqs = np.fft.fftfreq(n, d)
        cFreqs = NumCpp.NdArray(1, freqs.size)
        cFreqs.setArray(freqs)
        assert np.array_equal(
            np.round(NumCpp.ifftshift(cFreqs, NumCpp.Axis.NONE).flatten(), 8), np.round(np.fft.ifftshift(freqs), 8)
        )

        dim0 = np.random.randint(10, 100)
        dim1 = np.random.randint(10, 100)
        n = dim0 * dim1
        d = np.random.rand()
        freqs = np.fft.fftfreq(n, d).reshape(dim0, dim1)
        cFreqs = NumCpp.NdArray(freqs.shape[0], freqs.shape[1])
        cFreqs.setArray(freqs)
        assert np.array_equal(
            np.round(NumCpp.ifftshift(cFreqs, NumCpp.Axis.ROW), 8), np.round(np.fft.ifftshift(freqs, axes=0), 8)
        )
        assert np.array_equal(
            np.round(NumCpp.ifftshift(cFreqs, NumCpp.Axis.COL), 8), np.round(np.fft.ifftshift(freqs, axes=1), 8)
        )


####################################################################################
def test_fft2():
    for _ in range(NUM_TRIALS):
        shapeInput = np.random.randint(
            10,
            30,
            [
                2,
            ],
        )
        data = np.random.randint(0, 100, shapeInput)
        cShape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(cShape)
        cArray.setArray(data)
        assert np.array_equal(np.round(NumCpp.fft2(cArray), ROUNDING_DIGITS), np.round(np.fft.fft2(data), ROUNDING_DIGITS))

        shapeInput = np.random.randint(
            10,
            30,
            [
                2,
            ],
        )
        data = np.random.randint(0, 100, shapeInput)
        cShape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(cShape)
        cArray.setArray(data)
        s = [np.random.randint(1, shapeInput[0]), np.random.randint(1, shapeInput[1])]
        assert np.array_equal(np.round(NumCpp.fft2(cArray, NumCpp.Shape(*s)), ROUNDING_DIGITS), np.round(np.fft.fft2(data, s), ROUNDING_DIGITS))


####################################################################################
def test_ifft2():
    for _ in range(NUM_TRIALS):
        shapeInput = np.random.randint(
            10,
            30,
            [
                2,
            ],
        )
        data = np.random.randint(0, 100, shapeInput)
        cShape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(cShape)
        cArray.setArray(data)
        assert np.array_equal(np.round(NumCpp.ifft2(cArray), ROUNDING_DIGITS), np.round(np.fft.ifft2(data), ROUNDING_DIGITS))

        shapeInput = np.random.randint(
            10,
            30,
            [
                2,
            ],
        )
        data = np.random.randint(0, 100, shapeInput)
        cShape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(cShape)
        cArray.setArray(data)
        s = [np.random.randint(1, shapeInput[0]), np.random.randint(1, shapeInput[1])]
        assert np.array_equal(np.round(NumCpp.ifft2(cArray, NumCpp.Shape(*s)), ROUNDING_DIGITS), np.round(np.fft.ifft2(data, s), ROUNDING_DIGITS))


####################################################################################
def test_rfft():
    for _ in range(NUM_TRIALS):
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
            np.round(NumCpp.rfft(cArray, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.rfft(data.flatten()), ROUNDING_DIGITS)
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
            np.round(NumCpp.rfft(cArray, n, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.rfft(data.flatten(), n), ROUNDING_DIGITS)
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
            np.round(NumCpp.rfft(cArray, n, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.rfft(data.flatten(), n), ROUNDING_DIGITS)
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
        assert np.array_equal(np.round(NumCpp.rfft(cArray, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.rfft(data, axis=0), ROUNDING_DIGITS))

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
        assert np.array_equal(
            np.round(NumCpp.rfft(cArray, n, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.rfft(data, n, axis=0), ROUNDING_DIGITS)
        )

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
        assert np.array_equal(np.round(NumCpp.rfft(cArray, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.rfft(data, axis=0), ROUNDING_DIGITS))

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
        assert np.array_equal(np.round(NumCpp.rfft(cArray, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.rfft(data, axis=1), ROUNDING_DIGITS))

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
        assert np.array_equal(
            np.round(NumCpp.rfft(cArray, n, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.rfft(data, n, axis=1), ROUNDING_DIGITS)
        )

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
        assert np.array_equal(
            np.round(NumCpp.rfft(cArray, n, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.rfft(data, n, axis=1), ROUNDING_DIGITS)
        )


####################################################################################
def test_irfft():
    for _ in range(NUM_TRIALS):
        # axis none, default n
        length = np.random.randint(100, 300)
        data = np.random.randint(0, 100, [length]).astype(float)
        rfft = np.fft.rfft(data)
        cShape = NumCpp.Shape(1, rfft.size)
        cArray = NumCpp.NdArrayComplexDouble(cShape)
        cArray.setArray(rfft)
        assert np.array_equal(
            np.round(NumCpp.irfft(cArray, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.irfft(rfft.flatten()), ROUNDING_DIGITS)
        )

        # axis none, smaller n
        length = np.random.randint(100, 300)
        n = np.random.randint(1, length)
        data = np.random.randint(0, 100, [length]).astype(float)
        rfft = np.fft.rfft(data, n)
        cShape = NumCpp.Shape(1, rfft.size)
        cArray = NumCpp.NdArrayComplexDouble(cShape)
        cArray.setArray(rfft)
        assert np.array_equal(
            np.round(NumCpp.irfft(cArray, n, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.irfft(rfft.flatten(), n), ROUNDING_DIGITS)
        )

        # axis none, larger n
        length = np.random.randint(100, 300)
        n = np.random.randint(length + 1, length + 20)
        data = np.random.randint(0, 100, [length]).astype(float)
        rfft = np.fft.rfft(data, n)
        cShape = NumCpp.Shape(1, rfft.size)
        cArray = NumCpp.NdArrayComplexDouble(cShape)
        cArray.setArray(rfft)
        assert np.array_equal(
            np.round(NumCpp.irfft(cArray, n, NumCpp.Axis.NONE), ROUNDING_DIGITS).flatten(), np.round(np.fft.irfft(rfft.flatten(), n), ROUNDING_DIGITS)
        )

        # axis Row, default n
        shapeInput = np.random.randint(
            10,
            30,
            [
                2,
            ],
        )
        data = np.random.randint(0, 100, shapeInput).astype(float)
        rfft = np.fft.rfft(data, axis=0)
        cShape = NumCpp.Shape(rfft.shape[0], rfft.shape[1])
        cArray = NumCpp.NdArrayComplexDouble(cShape)
        cArray.setArray(rfft)
        assert np.array_equal(np.round(NumCpp.irfft(cArray, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.irfft(rfft, axis=0), ROUNDING_DIGITS))

        # axis Row, smaller n
        shapeInput = np.random.randint(
            10,
            30,
            [
                2,
            ],
        )
        data = np.random.randint(0, 100, shapeInput).astype(float)
        n = np.random.randint(5, data.shape[0])
        rfft = np.fft.rfft(data, n, axis=0)
        cShape = NumCpp.Shape(rfft.shape[0], rfft.shape[1])
        cArray = NumCpp.NdArrayComplexDouble(cShape)
        cArray.setArray(rfft)
        # assert np.array_equal(
        #     np.round(NumCpp.irfft(cArray, n, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.irfft(rfft, axis=0), ROUNDING_DIGITS)
        # )
        print(
            np.array_equal(
                np.round(NumCpp.irfft(cArray, n, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.irfft(rfft, n, axis=0), ROUNDING_DIGITS)
            )
        )

        # axis Row, larger n
        shapeInput = np.random.randint(
            10,
            30,
            [
                2,
            ],
        )
        data = np.random.randint(0, 100, shapeInput).astype(float)
        n = np.random.randint(data.shape[0] + 1, data.shape[0] + 20)
        rfft = np.fft.rfft(data, axis=0)
        cShape = NumCpp.Shape(rfft.shape[0], rfft.shape[1])
        cArray = NumCpp.NdArrayComplexDouble(cShape)
        cArray.setArray(rfft)
        assert np.array_equal(
            np.round(NumCpp.irfft(cArray, n, NumCpp.Axis.ROW), ROUNDING_DIGITS), np.round(np.fft.irfft(rfft, n, axis=0), ROUNDING_DIGITS)
        )

        # axis Col, default n
        shapeInput = np.random.randint(
            10,
            30,
            [
                2,
            ],
        )
        data = np.random.randint(0, 100, shapeInput).astype(float)
        rfft = np.fft.rfft(data, axis=1)
        cShape = NumCpp.Shape(rfft.shape[0], rfft.shape[1])
        cArray = NumCpp.NdArrayComplexDouble(cShape)
        cArray.setArray(rfft)
        assert np.array_equal(np.round(NumCpp.irfft(cArray, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.irfft(rfft, axis=1), ROUNDING_DIGITS))

        # axis Col, smaller n
        shapeInput = np.random.randint(
            10,
            30,
            [
                2,
            ],
        )
        data = np.random.randint(0, 100, shapeInput).astype(float)
        n = np.random.randint(5, data.shape[1])
        rfft = np.fft.rfft(data, n, axis=1)
        cShape = NumCpp.Shape(rfft.shape[0], rfft.shape[1])
        cArray = NumCpp.NdArrayComplexDouble(cShape)
        cArray.setArray(rfft)
        assert np.array_equal(
            np.round(NumCpp.irfft(cArray, n, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.irfft(rfft, n, axis=1), ROUNDING_DIGITS)
        )

        # axis Col, larger n
        shapeInput = np.random.randint(
            10,
            30,
            [
                2,
            ],
        )
        data = np.random.randint(0, 100, shapeInput).astype(float)
        n = np.random.randint(data.shape[1] + 1, data.shape[1] + 20)
        rfft = np.fft.rfft(data, axis=1)
        cShape = NumCpp.Shape(rfft.shape[0], rfft.shape[1])
        cArray = NumCpp.NdArrayComplexDouble(cShape)
        cArray.setArray(rfft)
        assert np.array_equal(
            np.round(NumCpp.irfft(cArray, n, NumCpp.Axis.COL), ROUNDING_DIGITS), np.round(np.fft.irfft(rfft, n, axis=1), ROUNDING_DIGITS)
        )


####################################################################################
def test_rfftfreq():
    for _ in range(NUM_TRIALS):
        n = np.random.randint(10, 1000)
        d = np.random.rand()
        assert np.array_equal(np.round(NumCpp.rfftfreq(n, d).flatten(), 8), np.round(np.fft.rfftfreq(n, d), 8))


####################################################################################
def test_rfft2():
    for _ in range(NUM_TRIALS):
        shapeInput = np.random.randint(
            10,
            30,
            [
                2,
            ],
        )
        data = np.random.randint(0, 100, shapeInput)
        cShape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(cShape)
        cArray.setArray(data)
        assert np.array_equal(np.round(NumCpp.rfft2(cArray), ROUNDING_DIGITS), np.round(np.fft.rfft2(data), ROUNDING_DIGITS))

        shapeInput = np.random.randint(
            10,
            30,
            [
                2,
            ],
        )
        data = np.random.randint(0, 100, shapeInput)
        cShape = NumCpp.Shape(*shapeInput)
        cArray = NumCpp.NdArray(cShape)
        cArray.setArray(data)
        s = [np.random.randint(1, shapeInput[0]), np.random.randint(1, shapeInput[1])]
        assert np.array_equal(np.round(NumCpp.rfft2(cArray, NumCpp.Shape(*s)), ROUNDING_DIGITS), np.round(np.fft.rfft2(data, s), ROUNDING_DIGITS))


####################################################################################
def test_irfft2():
    for _ in range(NUM_TRIALS):
        shapeInput = np.random.randint(
            10,
            30,
            [
                2,
            ],
        )
        data = np.random.randint(0, 100, shapeInput)
        rfft2 = np.fft.rfft2(data)
        cShape = NumCpp.Shape(*rfft2.shape)
        cArray = NumCpp.NdArrayComplexDouble(cShape)
        cArray.setArray(rfft2)
        assert np.array_equal(np.round(NumCpp.irfft2(cArray), ROUNDING_DIGITS), np.round(np.fft.irfft2(rfft2), ROUNDING_DIGITS))
