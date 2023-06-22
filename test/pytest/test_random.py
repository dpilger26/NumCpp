import numpy as np

import NumCppPy as NumCpp  # noqa E402

np.random.seed(666)


# it is kind of hard to test randomness so my criteria for passing will
# simply be whether or not it crashes
####################################################################################
def test_bernoulli():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    p = np.random.rand()
    assert np.array_equal(NumCpp.bernoulli(inShape, p).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.bernoulli(p)) is bool


####################################################################################
def test_beta():
    if NumCpp.NUMCPP_NO_USE_BOOST:
        return

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    alpha, beta = np.random.randint(
        1,
        100,
        [
            2,
        ],
    ).astype(float)
    assert np.array_equal(NumCpp.beta(inShape, alpha, beta).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.beta(alpha, beta)) is float


####################################################################################
def test_binomial():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    n = np.random.randint(100)
    p = np.random.rand(1)
    assert np.array_equal(NumCpp.binomial(inShape, n, p).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.binomial(n, p)) is int


####################################################################################
def test_cauchy():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    mean, sigma = np.random.randint(
        1,
        100,
        [
            2,
        ],
    ).astype(float)
    assert np.array_equal(NumCpp.cauchy(inShape, mean, sigma).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.cauchy(mean, sigma)) is float


####################################################################################
def test_chiSquare():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    dof = np.random.randint(1, 100)
    assert np.array_equal(NumCpp.chiSquare(inShape, dof).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.chiSquare(dof)) is float


####################################################################################
def test_choice():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    cArray = NumCpp.NdArray(inShape)
    data = np.random.randint(0, 100, shapeInput)
    cArray.setArray(data)
    assert type(NumCpp.choiceSingle(cArray)) is float
    assert NumCpp.choiceSingle(cArray) in data

    num = np.random.randint(data.size)
    assert NumCpp.choiceMultiple(cArray, num, False).size == num
    assert NumCpp.choiceMultiple(cArray, num, True).size == num


####################################################################################
def test_discrete():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    cWeights = NumCpp.NdArray(inShape)
    weights = np.random.randint(0, 100, shapeInput)
    cWeights.setArray(weights)
    assert np.array_equal(NumCpp.discrete(inShape, cWeights).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.discrete(cWeights)) is int


####################################################################################
def test_exponential():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    scaleValue = np.random.randint(1, 100)
    rng = NumCpp.RNG()
    assert np.array_equal(NumCpp.exponential(inShape, scaleValue).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.exponential(scaleValue)) is float


####################################################################################
def test_extremeValue():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    a, b = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    assert np.array_equal(NumCpp.extremeValue(inShape, a, b).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.extremeValue(a, b)) is float


####################################################################################
def test_f():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    dof_n, dof_d = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    assert np.array_equal(NumCpp.f(inShape, dof_n, dof_d).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.f(dof_n, dof_d)) is float


####################################################################################
def test_gamma():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    gamma_shape, scale_value = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    assert np.array_equal(NumCpp.gamma(inShape, gamma_shape, scale_value).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.gamma(gamma_shape, scale_value)) is float


####################################################################################
def test_geometric():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    p = np.random.rand()
    assert np.array_equal(NumCpp.geometric(inShape, p).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.geometric(p)) is int


####################################################################################
def test_laplace():
    if NumCpp.NUMCPP_NO_USE_BOOST:
        return

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    loc, scale = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    assert np.array_equal(NumCpp.laplace(inShape, loc, scale).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.laplace(loc, scale)) is float


####################################################################################
def test_lognormal():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    mean, sigma = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    assert np.array_equal(NumCpp.lognormal(inShape, mean, sigma).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.lognormal(mean, sigma)) is float


####################################################################################
def test_negativeBinomial():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    n = np.random.randint(100)
    p = np.random.rand()
    assert np.array_equal(NumCpp.negativeBinomial(inShape, n, p).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.negativeBinomial(n, p)) is int


####################################################################################
def test_nonCentralChiSquared():
    if NumCpp.NUMCPP_NO_USE_BOOST:
        return

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    k, lambda_ = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    assert np.array_equal(NumCpp.nonCentralChiSquared(inShape, k, lambda_).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.nonCentralChiSquared(k, lambda_)) is float


####################################################################################
def test_normal():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    mean, sigma = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    assert np.array_equal(NumCpp.normal(inShape, mean, sigma).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.normal(mean, sigma)) is float


####################################################################################
def test_permutation():
    size = np.random.randint(50, 100)
    permutation = NumCpp.permutationScalar(size).flatten()
    assert permutation.size == size
    assert np.array_equal(np.sort(permutation), np.arange(size).astype(float))

    shape = NumCpp.Shape(1, size)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    permutation = NumCpp.permutationArray(cArray).flatten()
    assert permutation.size == size
    assert np.array_equal(np.unique(permutation), np.unique(data))


####################################################################################
def test_poisson():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    mean, sigma = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    assert np.array_equal(NumCpp.normal(inShape, mean, sigma).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.normal(mean, sigma)) is float


####################################################################################
def test_rand():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    randValues = NumCpp.rand(inShape).getNumpyArray()
    assert np.array_equal(randValues.shape, shapeInput)
    assert np.all(randValues < 1)
    assert type(NumCpp.rand()) is float


####################################################################################
def test_randFloat():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    low, high = np.sort(
        np.random.randint(
            1,
            100,
            [
                2,
            ],
        )
    )
    randValues = NumCpp.randFloat(inShape, low, high).getNumpyArray()
    assert np.array_equal(randValues.shape, shapeInput)
    assert np.all(np.logical_and(randValues >= low, randValues < high))
    assert type(NumCpp.randFloat(low, high)) is float


####################################################################################
def test_randInt():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    low, high = np.sort(
        np.random.randint(
            1,
            100,
            [
                2,
            ],
        )
    )
    randValues = NumCpp.randInt(inShape, low, high).getNumpyArray()
    assert np.array_equal(randValues.shape, shapeInput)
    assert np.all(np.logical_and(randValues >= low, randValues < high))
    assert type(NumCpp.randInt(low, high)) is int


####################################################################################
def test_randN():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    assert np.array_equal(NumCpp.randN(inShape).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.randN()) is float


####################################################################################
def test_seed():
    seed = np.random.randint(0, 100000)
    NumCpp.seed(seed)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + 1)
    values = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    values.sort()

    values1 = NumCpp.randInt(inShape, values[0].item(), values[1].item()).getNumpyArray()
    NumCpp.seed(seed)
    values2 = NumCpp.randInt(inShape, values[0].item(), values[1].item()).getNumpyArray()
    assert np.array_equal(values1, values2)


####################################################################################
def test_shuffle():
    size = np.random.randint(50, 100)
    shape = NumCpp.Shape(1, size)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    NumCpp.shuffle(cArray)
    assert cArray.size() == size
    assert np.array_equal(np.unique(cArray.getNumpyArray()), np.unique(data))


####################################################################################
def test_standardNormal():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    assert np.array_equal(NumCpp.standardNormal(inShape).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.standardNormal()) is float


####################################################################################
def test_studentT():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    dof = np.random.randint(1, 100)
    assert np.array_equal(NumCpp.studentT(inShape, dof).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.studentT(dof)) is float


####################################################################################
def test_triangle():
    if NumCpp.NUMCPP_NO_USE_BOOST:
        return

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    a, b, c = np.sort(
        np.random.randint(
            1,
            100,
            [
                3,
            ],
        )
    )
    assert np.array_equal(NumCpp.triangle(inShape, a, b, c).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.triangle(a, b, c)) is float


####################################################################################
def test_uniform():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    low, high = np.sort(
        np.random.randint(
            1,
            100,
            [
                2,
            ],
        )
    )
    randValues = NumCpp.uniform(inShape, low, high).getNumpyArray()
    assert np.array_equal(randValues.shape, shapeInput)
    assert np.all(np.logical_and(randValues >= low, randValues < high))
    assert type(NumCpp.randFloat(low, high)) is float


####################################################################################
def test_uniformOnSphere():
    if NumCpp.NUMCPP_NO_USE_BOOST:
        return

    numPoints = np.random.randint(10, 100)
    numDims = np.random.randint(2, 10)
    assert np.array_equal(NumCpp.uniformOnSphere(numPoints, numDims).getNumpyArray().shape, [numPoints, numDims])


####################################################################################
def test_weibull():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    a, b = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    assert np.array_equal(NumCpp.weibull(inShape, a, b).getNumpyArray().shape, shapeInput)
    assert type(NumCpp.weibull(a, b)) is float


####################################################################################
def test_RNG_bernoulli():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    p = np.random.rand()
    rng = NumCpp.RNG()
    assert np.array_equal(rng.bernoulli(inShape, p).shape, shapeInput)
    assert type(rng.bernoulli(p)) is bool


####################################################################################
def test_RNG_beta():
    if NumCpp.NUMCPP_NO_USE_BOOST:
        return

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    alpha, beta = np.random.randint(
        1,
        100,
        [
            2,
        ],
    ).astype(float)
    rng = NumCpp.RNG()
    assert np.array_equal(rng.beta(inShape, alpha, beta).shape, shapeInput)
    assert type(rng.beta(alpha, beta)) is float


####################################################################################
def test_RNG_binomial():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    n = np.random.randint(100)
    p = np.random.rand(1)
    rng = NumCpp.RNG()
    assert np.array_equal(rng.binomial(inShape, n, p).shape, shapeInput)
    assert type(rng.binomial(n, p)) is int


####################################################################################
def test_RNG_cauchy():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    mean, sigma = np.random.randint(
        1,
        100,
        [
            2,
        ],
    ).astype(float)
    rng = NumCpp.RNG()
    assert np.array_equal(rng.cauchy(inShape, mean, sigma).shape, shapeInput)
    assert type(rng.cauchy(mean, sigma)) is float


####################################################################################
def test_RNG_chiSquare():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    dof = np.random.randint(1, 100)
    rng = NumCpp.RNG()
    assert np.array_equal(rng.chiSquare(inShape, dof).shape, shapeInput)
    assert type(rng.chiSquare(dof)) is float


####################################################################################
def test_RNG_choice():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    cArray = NumCpp.NdArray(inShape)
    data = np.random.randint(0, 100, shapeInput)
    cArray.setArray(data)
    rng = NumCpp.RNG()
    assert type(rng.choice(cArray)) is float
    assert rng.choice(cArray) in data

    num = np.random.randint(data.size)
    assert rng.choice(cArray, num, False).size == num
    assert rng.choice(cArray, num, True).size == num


####################################################################################
def test_RNG_discrete():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    cWeights = NumCpp.NdArray(inShape)
    weights = np.random.randint(0, 100, shapeInput)
    cWeights.setArray(weights)
    rng = NumCpp.RNG()
    assert np.array_equal(rng.discrete(inShape, cWeights).shape, shapeInput)
    assert type(rng.discrete(cWeights)) is int


####################################################################################
def test_RNG_exponential():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    scaleValue = np.random.randint(1, 100)
    rng = NumCpp.RNG()
    assert np.array_equal(rng.exponential(inShape, scaleValue).shape, shapeInput)
    assert type(rng.exponential(scaleValue)) is float


####################################################################################
def test_RNG_extremeValue():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    a, b = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    rng = NumCpp.RNG()
    assert np.array_equal(rng.extremeValue(inShape, a, b).shape, shapeInput)
    assert type(rng.extremeValue(a, b)) is float


####################################################################################
def test_RNG_f():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    dof_n, dof_d = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    rng = NumCpp.RNG()
    assert np.array_equal(rng.f(inShape, dof_n, dof_d).shape, shapeInput)
    assert type(rng.f(dof_n, dof_d)) is float


####################################################################################
def test_RNG_gamma():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    gamma_shape, scale_value = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    rng = NumCpp.RNG()
    assert np.array_equal(rng.gamma(inShape, gamma_shape, scale_value).shape, shapeInput)
    assert type(rng.gamma(gamma_shape, scale_value)) is float


####################################################################################
def test_RNG_geometric():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    p = np.random.rand()
    rng = NumCpp.RNG()
    assert np.array_equal(rng.geometric(inShape, p).shape, shapeInput)
    assert type(rng.geometric(p)) is int


####################################################################################
def test_RNG_laplace():
    if NumCpp.NUMCPP_NO_USE_BOOST:
        return

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    loc, scale = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    rng = NumCpp.RNG()
    assert np.array_equal(rng.laplace(inShape, loc, scale).shape, shapeInput)
    assert type(rng.laplace(loc, scale)) is float


####################################################################################
def test_RNG_lognormal():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    mean, sigma = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    rng = NumCpp.RNG()
    assert np.array_equal(rng.lognormal(inShape, mean, sigma).shape, shapeInput)
    assert type(rng.lognormal(mean, sigma)) is float


####################################################################################
def test_RNG_negativeBinomial():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    n = np.random.randint(100)
    p = np.random.rand()
    rng = NumCpp.RNG()
    assert np.array_equal(rng.negativeBinomial(inShape, n, p).shape, shapeInput)
    assert type(rng.negativeBinomial(n, p)) is int


####################################################################################
def test_RNG_nonCentralChiSquared():
    if NumCpp.NUMCPP_NO_USE_BOOST:
        return

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    k, lambda_ = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    rng = NumCpp.RNG()
    assert np.array_equal(rng.nonCentralChiSquared(inShape, k, lambda_).shape, shapeInput)
    assert type(rng.nonCentralChiSquared(k, lambda_)) is float


####################################################################################
def test_RNG_normal():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    mean, sigma = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    rng = NumCpp.RNG()
    assert np.array_equal(rng.normal(inShape, mean, sigma).shape, shapeInput)
    assert type(rng.normal(mean, sigma)) is float


####################################################################################
def test_RNG_permutation():
    rng = NumCpp.RNG()
    size = np.random.randint(50, 100)
    permutation = rng.permutation(size).flatten()
    assert permutation.size == size
    assert np.array_equal(np.sort(permutation), np.arange(size).astype(float))

    shape = NumCpp.Shape(1, size)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    permutation = rng.permutation(cArray).flatten()
    assert permutation.size == size
    assert np.array_equal(np.unique(permutation), np.unique(data))


####################################################################################
def test_RNG_poisson():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    mean, sigma = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    rng = NumCpp.RNG()
    assert np.array_equal(rng.normal(inShape, mean, sigma).shape, shapeInput)
    assert type(rng.normal(mean, sigma)) is float


####################################################################################
def test_RNG_rand():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    rng = NumCpp.RNG()
    randValues = rng.rand(inShape)
    assert np.array_equal(randValues.shape, shapeInput)
    assert np.all(randValues < 1)
    assert type(rng.rand()) is float


####################################################################################
def test_RNG_randFloat():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    low, high = np.sort(
        np.random.randint(
            1,
            100,
            [
                2,
            ],
        )
    )
    rng = NumCpp.RNG()
    randValues = rng.randFloat(inShape, low, high)
    assert np.array_equal(randValues.shape, shapeInput)
    assert np.all(np.logical_and(randValues >= low, randValues < high))
    assert type(rng.randFloat(low, high)) is float


####################################################################################
def test_RNG_randInt():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    low, high = np.sort(
        np.random.randint(
            1,
            100,
            [
                2,
            ],
        )
    )
    rng = NumCpp.RNG()
    randValues = rng.randInt(inShape, low, high)
    assert np.array_equal(randValues.shape, shapeInput)
    assert np.all(np.logical_and(randValues >= low, randValues < high))
    assert type(rng.randInt(low, high)) is int


####################################################################################
def test_RNG_randN():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    rng = NumCpp.RNG()
    assert np.array_equal(rng.randN(inShape).shape, shapeInput)
    assert type(rng.randN()) is float


####################################################################################
def test_RNG_seed():
    seed = np.random.randint(0, 100000)
    rng = NumCpp.RNG()
    rng.seed(seed)

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    )
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + 1)
    values = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    values.sort()

    values1 = rng.randInt(inShape, values[0].item(), values[1].item())
    rng.seed(seed)
    values2 = rng.randInt(inShape, values[0].item(), values[1].item())
    assert np.array_equal(values1, values2)


####################################################################################
def test_RNG_shuffle():
    rng = NumCpp.RNG()
    size = np.random.randint(50, 100)
    shape = NumCpp.Shape(1, size)
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 100, [shape.rows, shape.cols]).astype(float)
    cArray.setArray(data)
    rng.shuffle(cArray)
    assert cArray.size() == size
    assert np.array_equal(np.unique(cArray.getNumpyArray()), np.unique(data))


####################################################################################
def test_RNG_standardNormal():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    rng = NumCpp.RNG()
    assert np.array_equal(rng.standardNormal(inShape).shape, shapeInput)
    assert type(rng.standardNormal()) is float


####################################################################################
def test_RNG_studentT():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    dof = np.random.randint(1, 100)
    rng = NumCpp.RNG()
    assert np.array_equal(rng.studentT(inShape, dof).shape, shapeInput)
    assert type(rng.studentT(dof)) is float


####################################################################################
def test_RNG_triangle():
    if NumCpp.NUMCPP_NO_USE_BOOST:
        return

    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    a, b, c = np.sort(
        np.random.randint(
            1,
            100,
            [
                3,
            ],
        )
    )
    rng = NumCpp.RNG()
    assert np.array_equal(rng.triangle(inShape, a, b, c).shape, shapeInput)
    assert type(rng.triangle(a, b, c)) is float


####################################################################################
def test_RNG_uniform():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    low, high = np.sort(
        np.random.randint(
            1,
            100,
            [
                2,
            ],
        )
    )
    rng = NumCpp.RNG()
    randValues = rng.uniform(inShape, low, high)
    assert np.array_equal(randValues.shape, shapeInput)
    assert np.all(np.logical_and(randValues >= low, randValues < high))
    assert type(rng.randFloat(low, high)) is float


####################################################################################
def test_RNG_uniformOnSphere():
    if NumCpp.NUMCPP_NO_USE_BOOST:
        return

    numPoints = np.random.randint(10, 100)
    numDims = np.random.randint(2, 10)
    rng = NumCpp.RNG()
    assert np.array_equal(rng.uniformOnSphere(numPoints, numDims).shape, [numPoints, numDims])


####################################################################################
def test_RNG_weibull():
    shapeInput = np.random.randint(
        2,
        100,
        [
            2,
        ],
    ).flatten()
    inShape = NumCpp.Shape(*shapeInput)
    a, b = np.random.randint(
        1,
        100,
        [
            2,
        ],
    )
    rng = NumCpp.RNG()
    assert np.array_equal(rng.weibull(inShape, a, b).shape, shapeInput)
    assert type(rng.weibull(a, b)) is float
