import numpy as np
import os
import sys
sys.path.append(os.path.abspath(r'../lib'))
import NumCpp  # noqa E402

np.random.seed(666)

# it is kind of hard to test randomness so my criteria for passing will
# simply be whether or not it crashes
####################################################################################
def test_bernoulli():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    p = np.random.rand()
    assert NumCpp.bernoulli(inShape, p) is not None
    assert NumCpp.bernoulli(p) is not None


####################################################################################
def test_beta():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    alpha = np.random.rand()
    beta = np.random.rand()
    assert NumCpp.beta(inShape, alpha, beta) is not None
    assert NumCpp.beta(alpha, beta) is not None


####################################################################################
def test_binomial():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    n = np.random.randint(1, 100, [1, ]).item()
    p = np.random.rand()
    assert NumCpp.binomial(inShape, n, p) is not None
    assert NumCpp.binomial(n, p) is not None


####################################################################################
def test_cauchy():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    mean = np.random.randn() * 10
    sigma = np.random.rand() * 10
    assert NumCpp.cauchy(inShape, mean, sigma) is not None
    assert NumCpp.cauchy(mean, sigma) is not None


####################################################################################
def test_chiSquare():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    dof = np.random.randint(1, 100, [1, ]).item()
    assert NumCpp.chiSquare(inShape, dof) is not None
    assert NumCpp.chiSquare(dof) is not None


####################################################################################
def test_choice():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    assert NumCpp.choiceSingle(cArray) is not None

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    num = np.random.randint(1, data.size, [1, ]).item()
    assert NumCpp.choiceMultiple(cArray, num) is not None


####################################################################################
def test_discrete():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    weights = np.random.randint(1, 10, [shape.rows, shape.cols])
    cArray.setArray(weights)
    assert NumCpp.discrete(shape, cArray) is not None
    assert NumCpp.discrete(cArray) is not None


####################################################################################
def test_exponential():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    scale = np.random.rand() * 10
    assert NumCpp.exponential(inShape, scale) is not None
    assert NumCpp.exponential(scale) is not None


####################################################################################
def test_extremeValue():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    a = np.random.rand() * 10
    b = np.random.rand() * 100
    assert NumCpp.extremeValue(inShape, a, b) is not None
    assert NumCpp.extremeValue(a, b) is not None


####################################################################################
def test_f():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    dofN = np.random.rand() * 10
    dofD = np.random.rand() * 100
    assert NumCpp.f(inShape, dofN, dofD) is not None
    assert NumCpp.f(dofN, dofD) is not None


####################################################################################
def test_gamma():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape = np.random.rand() * 10
    scale = np.random.rand() * 100
    assert NumCpp.gamma(inShape, shape, scale) is not None
    assert NumCpp.gamma(shape, scale) is not None


####################################################################################
def test_geometric():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    p = np.random.rand()
    assert NumCpp.geometric(inShape, p) is not None
    assert NumCpp.geometric(p) is not None


####################################################################################
def test_laplace():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    loc = np.random.rand() * 10
    scale = np.random.rand() * 100
    assert NumCpp.laplace(inShape, loc, scale) is not None
    assert NumCpp.laplace(loc, scale) is not None


####################################################################################
def test_lognormal():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    mean = np.random.randn() * 10
    sigma = np.random.rand() * 10
    assert NumCpp.lognormal(inShape, mean, sigma) is not None
    assert NumCpp.lognormal(mean, sigma) is not None


####################################################################################
def test_negativeBinomial():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    n = np.random.randint(1, 100, [1, ]).item()
    p = np.random.rand()
    assert NumCpp.negativeBinomial(inShape, n, p) is not None
    assert NumCpp.negativeBinomial(n, p) is not None


####################################################################################
def test_nonCentralChiSquared():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    k = np.random.rand() * 10
    ll = np.random.rand() * 100
    assert NumCpp.nonCentralChiSquared(inShape, k, ll) is not None
    assert NumCpp.nonCentralChiSquared(k, ll) is not None


####################################################################################
def test_normal():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    mean = np.random.randn() * 10
    sigma = np.random.rand() * 10
    assert NumCpp.normal(inShape, mean, sigma) is not None
    assert NumCpp.normal(mean, sigma) is not None


####################################################################################
def test_permutation():
    assert NumCpp.permutationScaler(np.random.randint(1, 100, [1, ]).item()) is not None

    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 10, [shape.rows, shape.cols])
    cArray.setArray(data)
    assert NumCpp.permutationArray(cArray) is not None


####################################################################################
def test_poisson():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    mean = np.random.rand() * 10
    assert NumCpp.poisson(inShape, mean) is not None
    assert NumCpp.poisson(mean) is not None


####################################################################################
def test_rand():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    assert NumCpp.rand(inShape) is not None
    assert NumCpp.rand() is not None


####################################################################################
def test_randFloat():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    values = np.random.randint(1, 100, [2, ])
    values.sort()
    assert NumCpp.randFloat(inShape, values[0].item(), values[1].item() + 1) is not None
    assert NumCpp.randFloat(values[0].item(), values[1].item() + 1) is not None


####################################################################################
def test_randInt():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + 1)
    values = np.random.randint(1, 100, [2, ])
    values.sort()
    assert NumCpp.randInt(inShape, values[0].item(), values[1].item()) is not None
    assert NumCpp.randInt(values[0].item(), values[1].item() + 1) is not None


####################################################################################
def test_randN():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    assert NumCpp.randN(inShape) is not None
    assert NumCpp.randN() is not None


####################################################################################
def test_seed():
    NumCpp.seed(np.random.randint(0, 100000, [1, ]).item())


####################################################################################
def test_shuffle():
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 10, [shape.rows, shape.cols])
    cArray.setArray(data)
    NumCpp.shuffle(cArray)


####################################################################################
def test_standardNormal():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    assert NumCpp.standardNormal(inShape) is not None
    assert NumCpp.standardNormal() is not None


####################################################################################
def test_studentT():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    dof = np.random.randint(1, 100, [1, ]).item()
    assert NumCpp.studentT(inShape, dof) is not None
    assert NumCpp.studentT(dof) is not None


####################################################################################
def test_triangle():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    values = np.random.rand(3)
    values.sort()
    assert NumCpp.triangle(inShape, values[0].item(), values[1].item(), values[2].item()) is not None
    assert NumCpp.triangle(values[0].item(), values[1].item(), values[2].item()) is not None


####################################################################################
def test_uniform():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    values = np.random.randint(1, 100, [2, ])
    values.sort()
    assert NumCpp.uniform(inShape, values[0].item(), values[1].item()) is not None
    assert NumCpp.uniform(values[0].item(), values[1].item()) is not None


####################################################################################
def test_uniformOnSphere():
    inputs = np.random.randint(1, 100, [2, ])
    assert NumCpp.uniformOnSphere(inputs[0].item(), inputs[1].item()) is not None


####################################################################################
def test_weibull():
    shapeInput = np.random.randint(1, 100, [2, ])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    inputs = np.random.rand(2)
    assert NumCpp.weibull(inShape, inputs[0].item(), inputs[1].item()) is not None
    assert NumCpp.weibull(inputs[0].item(), inputs[1].item()) is not None
