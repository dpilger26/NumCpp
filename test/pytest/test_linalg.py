import numpy as np

import NumCppPy as NumCpp  # noqa E402


####################################################################################
def test_seed():
    np.random.seed(666)


####################################################################################
def test_cholesky():
    shapeInput = np.random.randint(
        5,
        50,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    a = np.random.randint(1, 100, [shape.rows, shape.cols]).flatten()
    aL = np.tril(a)
    b = aL.dot(aL.transpose())
    cArray.setArray(b)
    assert np.array_equal(np.round(NumCpp.cholesky(cArray).getNumpyArray()), np.round(aL))


####################################################################################
def test_det():
    for order in range(1, 5):
        shape = NumCpp.Shape(order)
        cArray = NumCpp.NdArray(shape)
        data = np.random.rand(order, order) * 10
        cArray.setArray(data)
        assert round(NumCpp.det(cArray), 8) == round(np.linalg.det(data), 8)

    for order in range(1, 10):
        shape = NumCpp.Shape(order)
        cArray = NumCpp.NdArrayInt64(shape)
        data = np.random.randint(0, 10, [order, order])
        cArray.setArray(data)
        assert NumCpp.det(cArray) == round(np.linalg.det(data))


####################################################################################
def test_eig():
    for _ in range(50):
        shape = np.random.randint(5, 50, [2,])
        data = np.random.randint(0, 100, shape).astype(float)
        data = np.dot(data, data.T) # real symmetric
        cArray = NumCpp.NdArray(*shape)
        cArray.setArray(data)
        cEigenValues, cEigenVectors = NumCpp.eig(cArray)
        eigenValues, eigenVectors = np.linalg.eig(data)
        assert np.array_equal(np.round(np.abs(cEigenValues.flatten()), 5), np.flip(np.sort(np.round(np.abs(eigenValues), 5))).real)

        for idx, eigenValue in enumerate(cEigenValues.flatten()):
            eigenVector = cEigenVectors[:, idx]
            assert np.round(np.linalg.norm(eigenVector), 8) == 1.0
            aTimesV = np.round(np.linalg.norm(np.dot(data, eigenVector)), 5)
            assert aTimesV == np.round(eigenValue, 5)


####################################################################################
def test_eigvals():
    for _ in range(50):
        size = np.random.randint(5, 50)
        data = np.random.randint(0, 100, [size, size])
        data = np.dot(data, data.T)
        cArray = NumCpp.NdArray()
        cArray.setArray(data)
        cEigenValues = NumCpp.eigvals(cArray)
        eigenValues = np.linalg.eigvals(data)
        assert np.array_equal(np.round(np.abs(cEigenValues.flatten()), 5), np.flip(np.sort(np.round(np.abs(eigenValues), 5))).real)


####################################################################################
def test_hat():
    shape = NumCpp.Shape(1, 3)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(0, 100, [shape.rows, shape.cols]).flatten()
    cArray.setArray(data)
    assert np.array_equal(NumCpp.hat(cArray), hat(data))


####################################################################################
def hat(inVec):
    mat = np.zeros([3, 3])
    mat[0, 1] = -inVec[2]
    mat[0, 2] = inVec[1]
    mat[1, 0] = inVec[2]
    mat[1, 2] = -inVec[0]
    mat[2, 0] = -inVec[1]
    mat[2, 1] = inVec[0]

    return mat


####################################################################################
def test_inv():
    max_order = 30

    for order in range(1, max_order):
        shape = NumCpp.Shape(order)
        cArray = NumCpp.NdArray(shape)
        data = np.random.randint(1, 100, [order, order])
        cArray.setArray(data)
        assert np.array_equal(np.round(NumCpp.inv(cArray).getNumpyArray(), 7), np.round(np.linalg.inv(data), 7))

    # test zero on diagnol
    for order in range(2, max_order):
        shape = NumCpp.Shape(order)
        cArray = NumCpp.NdArray(shape)
        data = np.random.randint(1, 100, [order, order])
        idx = np.random.randint(0, order)
        data[idx, idx] = 0
        cArray.setArray(data)
        assert np.array_equal(np.round(NumCpp.inv(cArray).getNumpyArray(), 7), np.round(np.linalg.inv(data), 7))


####################################################################################
def test_lstsq():
    shapeInput = np.random.randint(
        5,
        50,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    aArray = NumCpp.NdArray(shape)
    bArray = NumCpp.NdArray(1, shape.rows)
    aData = np.random.randint(1, 100, [shape.rows, shape.cols])
    bData = np.random.randint(
        1,
        100,
        [
            shape.rows,
        ],
    )
    aArray.setArray(aData)
    bArray.setArray(bData)
    x = NumCpp.lstsq(aArray, bArray).getNumpyArray().flatten()
    assert np.array_equal(np.round(x, 7), np.round(np.linalg.lstsq(aData, bData, rcond=None)[0], 7)), f"{shapeInput}"

    shapeInput = np.random.randint(
        5,
        50,
        [
            2,
        ],
    )
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    aArray = NumCpp.NdArray(shape)
    bCols = np.random.randint(1, 100)
    bArray = NumCpp.NdArray(shape.rows, bCols)
    aData = np.random.randint(1, 100, [shape.rows, shape.cols])
    bData = np.random.randint(
        1,
        100,
        [shape.rows, bCols],
    )
    aArray.setArray(aData)
    bArray.setArray(bData)
    x = NumCpp.lstsq(aArray, bArray).getNumpyArray()
    assert np.array_equal(np.round(x, 7), np.round(np.linalg.lstsq(aData, bData, rcond=None)[0], 7)), f"{shapeInput}"


####################################################################################
def test_lu_decomposition():
    sizeInput = np.random.randint(5, 50)
    shape = NumCpp.Shape(sizeInput)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    l, u = NumCpp.lu_decomposition(cArray)
    p = np.round(np.dot(l.getNumpyArray(), u.getNumpyArray())).astype(int)
    assert np.array_equal(p, data)


####################################################################################
def test_matrix_power():
    order = np.random.randint(
        5,
        50,
        [
            1,
        ],
    ).item()
    shape = NumCpp.Shape(order)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.matrix_power(cArray, 0).getNumpyArray(), np.linalg.matrix_power(data, 0))

    order = np.random.randint(
        5,
        50,
        [
            1,
        ],
    ).item()
    shape = NumCpp.Shape(order)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(NumCpp.matrix_power(cArray, 1).getNumpyArray(), np.linalg.matrix_power(data, 1))

    order = np.random.randint(
        5,
        50,
        [
            1,
        ],
    ).item()
    shape = NumCpp.Shape(order)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert np.array_equal(
        np.round(NumCpp.matrix_power(cArray, -1).getNumpyArray(), 8), np.round(np.linalg.matrix_power(data, -1), 8)
    )

    order = np.random.randint(
        5,
        50,
        [
            1,
        ],
    ).item()
    shape = NumCpp.Shape(order)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 5, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    power = np.random.randint(
        2,
        9,
        [
            1,
        ],
    ).item()
    assert np.array_equal(NumCpp.matrix_power(cArray, power).getNumpyArray(), np.linalg.matrix_power(data, power))

    order = np.random.randint(
        5,
        50,
        [
            1,
        ],
    ).item()
    shape = NumCpp.Shape(order)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    power = (
        np.random.randint(
            2,
            9,
            [
                1,
            ],
        ).item()
        * -1
    )
    assert np.array_equal(
        np.round(NumCpp.matrix_power(cArray, power).getNumpyArray(), 8),
        np.round(np.linalg.matrix_power(data, power), 8),
    )


####################################################################################
def test_multi_dot():
    shapeInput = np.random.randint(
        5,
        50,
        [
            2,
        ],
    )
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(
        shape1.cols,
        np.random.randint(
            5,
            50,
            [
                1,
            ],
        ).item(),
    )
    shape3 = NumCpp.Shape(
        shape2.cols,
        np.random.randint(
            5,
            50,
            [
                1,
            ],
        ).item(),
    )
    shape4 = NumCpp.Shape(
        shape3.cols,
        np.random.randint(
            5,
            50,
            [
                1,
            ],
        ).item(),
    )
    cArray1 = NumCpp.NdArray(shape1)
    cArray2 = NumCpp.NdArray(shape2)
    cArray3 = NumCpp.NdArray(shape3)
    cArray4 = NumCpp.NdArray(shape4)
    data1 = np.random.randint(1, 10, [shape1.rows, shape1.cols])
    data2 = np.random.randint(1, 10, [shape2.rows, shape2.cols])
    data3 = np.random.randint(1, 10, [shape3.rows, shape3.cols])
    data4 = np.random.randint(1, 10, [shape4.rows, shape4.cols])
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    assert np.array_equal(
        np.round(NumCpp.multi_dot(cArray1, cArray2, cArray3, cArray4), 8),
        np.round(np.linalg.multi_dot([data1, data2, data3, data4]), 8),
    )

    shapeInput = np.random.randint(
        5,
        50,
        [
            2,
        ],
    )
    shape1 = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape2 = NumCpp.Shape(
        shape1.cols,
        np.random.randint(
            5,
            50,
            [
                1,
            ],
        ).item(),
    )
    shape3 = NumCpp.Shape(
        shape2.cols,
        np.random.randint(
            5,
            50,
            [
                1,
            ],
        ).item(),
    )
    shape4 = NumCpp.Shape(
        shape3.cols,
        np.random.randint(
            5,
            50,
            [
                1,
            ],
        ).item(),
    )
    cArray1 = NumCpp.NdArrayComplexDouble(shape1)
    cArray2 = NumCpp.NdArrayComplexDouble(shape2)
    cArray3 = NumCpp.NdArrayComplexDouble(shape3)
    cArray4 = NumCpp.NdArrayComplexDouble(shape4)
    real1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    imag1 = np.random.randint(1, 100, [shape1.rows, shape1.cols])
    data1 = real1 + 1j * imag1
    real2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    imag2 = np.random.randint(1, 100, [shape2.rows, shape2.cols])
    data2 = real2 + 1j * imag2
    real3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    imag3 = np.random.randint(1, 100, [shape3.rows, shape3.cols])
    data3 = real3 + 1j * imag3
    real4 = np.random.randint(1, 100, [shape4.rows, shape4.cols])
    imag4 = np.random.randint(1, 100, [shape4.rows, shape4.cols])
    data4 = real4 + 1j * imag4
    cArray1.setArray(data1)
    cArray2.setArray(data2)
    cArray3.setArray(data3)
    cArray4.setArray(data4)
    assert np.array_equal(
        np.round(NumCpp.multi_dot(cArray1, cArray2, cArray3, cArray4), 8),
        np.round(np.linalg.multi_dot([data1, data2, data3, data4]), 8),
    )


####################################################################################
def test_pinv():
    max_order = 30

    for order in range(10, max_order):
        shape = NumCpp.Shape(order + 5, order)
        cArray = NumCpp.NdArray(shape)
        data = np.random.randint(1, 100, [shape.rows, shape.cols])
        cArray.setArray(data)
        assert np.array_equal(np.round(NumCpp.pinv(cArray).getNumpyArray(), 8), np.round(np.linalg.pinv(data), 8))

        shape = NumCpp.Shape(order, order + 5)
        cArray = NumCpp.NdArray(shape)
        data = np.random.randint(1, 100, [shape.rows, shape.cols])
        cArray.setArray(data)
        assert np.array_equal(np.round(NumCpp.pinv(cArray).getNumpyArray(), 8), np.round(np.linalg.pinv(data), 8))


####################################################################################
def test_pivotLU_decomposition():
    sizeInput = np.random.randint(5, 50)
    shape = NumCpp.Shape(sizeInput)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols])
    cArray.setArray(data)
    l, u, p = NumCpp.pivotLU_decomposition(cArray)
    lhs = p.dot(data)
    rhs = l.dot(u)
    assert np.array_equal(np.round(lhs, 10), np.round(rhs, 10))


####################################################################################
def test_solve():
    sizeInput = np.random.randint(5, 50)
    shape = NumCpp.Shape(sizeInput)
    aArray = NumCpp.NdArray(shape)
    a = np.random.randint(1, 100, [shape.rows, shape.cols])
    aArray.setArray(a)
    b = np.random.randint(1, 100, [shape.rows, 1])
    bArray = NumCpp.NdArray(*b.shape)
    bArray.setArray(b)
    assert np.array_equal(np.round(NumCpp.solve(aArray, bArray), 8), np.round(np.linalg.solve(a, b), 8))


####################################################################################
def test_svd():
    shape = np.random.randint(
        10,
        50,
        [
            2,
        ],
    )
    cArray = NumCpp.NdArray(*shape)
    data = np.random.randint(1, 100, shape).astype(float)
    cArray.setArray(data)
    uArray = NumCpp.NdArray()
    sArray = NumCpp.NdArray()
    vtArray = NumCpp.NdArray()
    NumCpp.svd(cArray, uArray, sArray, vtArray)
    
    u = uArray.getNumpyArray()
    s = sArray.getNumpyArray().flatten()
    vt = vtArray.getNumpyArray()

    ss = np.zeros(shape)
    for i in range(s.size):
        ss[i, i] = s[i]

    assert np.array_equal(np.round(np.linalg.multi_dot([u, ss, vt]), 8), data)


####################################################################################
def test_svdvals():
    shape = np.random.randint(
        10,
        50,
        [
            2,
        ],
    )
    shape = np.array([3, 5])
    data = np.random.randint(1, 100, shape).astype(float)
    cArray = NumCpp.NdArray(*shape)
    cArray.setArray(data)

    assert np.array_equal(np.round(NumCpp.svdvals(cArray).flatten(), 8), np.round(np.linalg.svdvals(data), 8))
