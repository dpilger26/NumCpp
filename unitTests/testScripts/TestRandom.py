import numpy as np
from termcolor import colored
import sys
if sys.platform == 'linux':
    sys.path.append(r'../src/build')
    import libNumCpp as NumCpp
else:
    sys.path.append(r'../build/x64/Release')
    import NumCpp

####################################################################################
def doTest():
    print(colored('Testing Random Module', 'magenta'))

    # it is kind of hard to test randomness so my criteria for passing will
    # simply be whether or not it crashes

    print(colored('Testing bernoulli', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    p = np.random.rand()
    r = NumCpp.Random.bernoulli(inShape, p)
    print(colored('\tPASS', 'green'))

    print(colored('Testing beta', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    alpha = np.random.rand()
    beta = np.random.rand()
    r = NumCpp.Random.beta(inShape, alpha, beta)
    print(colored('\tPASS', 'green'))

    print(colored('Testing binomial', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    n = np.random.randint(1, 100, [1,]).item()
    p = np.random.rand()
    r = NumCpp.Random.binomial(inShape, n, p)
    print(colored('\tPASS', 'green'))

    print(colored('Testing chiSquare', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    dof = np.random.randint(1, 100, [1,]).item()
    r = NumCpp.Random.chiSquare(inShape, dof)
    print(colored('\tPASS', 'green'))

    print(colored('Testing choice: Single', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    r = NumCpp.Random.choiceSingle(cArray)
    print(colored('\tPASS', 'green'))

    print(colored('Testing choice: Multiple', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    num = np.random.randint(1, data.size, [1,]).item()
    r = NumCpp.Random.choiceMultiple(cArray, num)
    if r.size == num:
        print(colored('\tPASS', 'green'))

    print(colored('Testing cauchy', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    mean = np.random.randn() * 10
    sigma = np.random.rand() * 10
    r = NumCpp.Random.cauchy(inShape, mean, sigma)
    print(colored('\tPASS', 'green'))

    print(colored('Testing discrete', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    weights = np.random.randint(1, 10, [shape.rows, shape.cols])
    cArray.setArray(weights)
    r = NumCpp.Random.discrete(inShape, cArray)
    print(colored('\tPASS', 'green'))

    print(colored('Testing exponential', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    scale = np.random.rand() * 10
    r = NumCpp.Random.exponential(inShape, scale)
    print(colored('\tPASS', 'green'))

    print(colored('Testing extremeValue', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    a = np.random.rand() * 10
    b = np.random.rand() * 100
    r = NumCpp.Random.extremeValue(inShape, a, b)
    print(colored('\tPASS', 'green'))

    print(colored('Testing f', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    dofN = np.random.rand() * 10
    dofD = np.random.rand() * 100
    r = NumCpp.Random.f(inShape, dofN, dofD)
    print(colored('\tPASS', 'green'))

    print(colored('Testing gamma', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape = np.random.rand() * 10
    scale = np.random.rand() * 100
    r = NumCpp.Random.gamma(inShape, shape, scale)
    print(colored('\tPASS', 'green'))

    print(colored('Testing geometric', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    p = np.random.rand()
    r = NumCpp.Random.geometric(inShape, p)
    print(colored('\tPASS', 'green'))

    print(colored('Testing laplace', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    loc = np.random.rand() * 10
    scale = np.random.rand() * 100
    r = NumCpp.Random.laplace(inShape, loc, scale)
    print(colored('\tPASS', 'green'))

    print(colored('Testing lognormal', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    mean = np.random.randn() * 10
    sigma = np.random.rand() * 10
    r = NumCpp.Random.lognormal(inShape, mean, sigma)
    print(colored('\tPASS', 'green'))

    print(colored('Testing negativeBinomial', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    n = np.random.randint(1, 100, [1,]).item()
    p = np.random.rand()
    r = NumCpp.Random.negativeBinomial(inShape, n, p)
    print(colored('\tPASS', 'green'))

    print(colored('Testing nonCentralChiSquared', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    k = np.random.rand() * 10
    l = np.random.rand() * 100
    r = NumCpp.Random.nonCentralChiSquared(inShape, k, l)
    print(colored('\tPASS', 'green'))

    print(colored('Testing normal', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    mean = np.random.randn() * 10
    sigma = np.random.rand() * 10
    r = NumCpp.Random.normal(inShape, mean, sigma)
    print(colored('\tPASS', 'green'))

    print(colored('Testing permutation scalar', 'cyan'))
    r = NumCpp.Random.permutationScaler(np.random.randint(1,100, [1,]).item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing permutation array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 10, [shape.rows, shape.cols])
    cArray.setArray(data)
    r = NumCpp.Random.permutationArray(cArray)
    print(colored('\tPASS', 'green'))

    print(colored('Testing poisson', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    mean = np.random.rand() * 10
    r = NumCpp.Random.poisson(inShape, mean)
    print(colored('\tPASS', 'green'))

    print(colored('Testing rand', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    r = NumCpp.Random.rand(inShape)
    print(colored('\tPASS', 'green'))

    print(colored('Testing randFloat', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    values = np.random.randint(1, 100, [2, ])
    r = NumCpp.Random.randFloat(inShape, values[0].item(), values[1].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing randInt', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    values = np.random.randint(1, 100, [2, ])
    r = NumCpp.Random.randInt(inShape, values[0].item(), values[1].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing randN', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    r = NumCpp.Random.randN(inShape)
    print(colored('\tPASS', 'green'))

    print(colored('Testing seed', 'cyan'))
    NumCpp.Random.seed(np.random.randint(0, 100000, [1,]).item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing shuffle array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 10, [shape.rows, shape.cols])
    cArray.setArray(data)
    NumCpp.Random.shuffle(cArray)
    print(colored('\tPASS', 'green'))

    print(colored('Testing standardNormal', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    r = NumCpp.Random.standardNormal(inShape)
    print(colored('\tPASS', 'green'))

    print(colored('Testing studentT', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    dof = np.random.randint(1, 100, [1, ]).item()
    r = NumCpp.Random.studentT(inShape, dof)
    print(colored('\tPASS', 'green'))

    print(colored('Testing triangle', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    values = np.random.rand(3)
    values.sort()
    r = NumCpp.Random.triangle(inShape, values[0].item(), values[1].item(), values[2].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing uniform', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    values = np.random.randint(1, 100, [2, ])
    r = NumCpp.Random.uniform(inShape, values[0].item(), values[1].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing uniformOnSphere', 'cyan'))
    inputs = np.random.randint(1, 100, [2,])
    r = NumCpp.Random.uniformOnSphere(inputs[0].item(), inputs[1].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing weibull', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    inputs = np.random.rand(2)
    r = NumCpp.Random.weibull(inShape, inputs[0].item(), inputs[1].item())
    print(colored('\tPASS', 'green'))

####################################################################################
if __name__ == '__main__':
    doTest()
