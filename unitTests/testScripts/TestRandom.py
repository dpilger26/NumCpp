import numpy as np
from termcolor import colored
import sys
sys.path.append(r'../build/x64/Release')
import NumC

####################################################################################
def doTest():
    print(colored('Testing Random Module', 'magenta'))

    # it is kind of hard to test randomness so my criteria for passing will
    # simply be whether or not it crashes

    print(colored('Testing bernoulli', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    p = np.random.rand()
    r = NumC.bernoulli(inShape, p)
    print(colored('\tPASS', 'green'))

    print(colored('Testing beta', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    alpha = np.random.rand()
    beta = np.random.rand()
    r = NumC.beta(inShape, alpha, beta)
    print(colored('\tPASS', 'green'))

    print(colored('Testing binomial', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    n = np.random.randint(1, 100, [1,]).item()
    p = np.random.rand()
    r = NumC.binomial(inShape, n, p)
    print(colored('\tPASS', 'green'))

    print(colored('Testing chiSquare', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    dof = np.random.randint(1, 100, [1,]).item()
    r = NumC.chiSquare(inShape, dof)
    print(colored('\tPASS', 'green'))

    print(colored('Testing choice', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    r = NumC.choice(cArray)
    print(colored('\tPASS', 'green'))

    print(colored('Testing cauchy', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    mean = np.random.randn() * 10
    sigma = np.random.rand() * 10
    r = NumC.cauchy(inShape, mean, sigma)
    print(colored('\tPASS', 'green'))

    print(colored('Testing discrete', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    weights = np.random.randint(1, 10, [shape.rows, shape.cols])
    cArray.setArray(weights)
    r = NumC.discrete(inShape, cArray)
    print(colored('\tPASS', 'green'))

    print(colored('Testing exponential', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    scale = np.random.rand() * 10
    r = NumC.exponential(inShape, scale)
    print(colored('\tPASS', 'green'))

    print(colored('Testing extremeValue', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    a = np.random.rand() * 10
    b = np.random.rand() * 100
    r = NumC.extremeValue(inShape, a, b)
    print(colored('\tPASS', 'green'))

    print(colored('Testing f', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    dofN = np.random.rand() * 10
    dofD = np.random.rand() * 100
    r = NumC.f(inShape, dofN, dofD)
    print(colored('\tPASS', 'green'))

    print(colored('Testing gamma', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape = np.random.rand() * 10
    scale = np.random.rand() * 100
    r = NumC.gamma(inShape, shape, scale)
    print(colored('\tPASS', 'green'))

    print(colored('Testing geometric', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    p = np.random.rand()
    r = NumC.geometric(inShape, p)
    print(colored('\tPASS', 'green'))

    print(colored('Testing laplace', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    loc = np.random.rand() * 10
    scale = np.random.rand() * 100
    r = NumC.laplace(inShape, loc, scale)
    print(colored('\tPASS', 'green'))

    print(colored('Testing lognormal', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    mean = np.random.randn() * 10
    sigma = np.random.rand() * 10
    r = NumC.lognormal(inShape, mean, sigma)
    print(colored('\tPASS', 'green'))

    print(colored('Testing negativeBinomial', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    n = np.random.randint(1, 100, [1,]).item()
    p = np.random.rand()
    r = NumC.negativeBinomial(inShape, n, p)
    print(colored('\tPASS', 'green'))

    print(colored('Testing nonCentralChiSquared', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    k = np.random.rand() * 10
    l = np.random.rand() * 100
    r = NumC.nonCentralChiSquared(inShape, k, l)
    print(colored('\tPASS', 'green'))

    print(colored('Testing normal', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    mean = np.random.randn() * 10
    sigma = np.random.rand() * 10
    r = NumC.normal(inShape, mean, sigma)
    print(colored('\tPASS', 'green'))

    print(colored('Testing permutation scalar', 'cyan'))
    r = NumC.permutation(np.random.randint(1,100, [1,]).item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing permutation array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 10, [shape.rows, shape.cols])
    cArray.setArray(data)
    r = NumC.permutation(cArray)
    print(colored('\tPASS', 'green'))

    print(colored('Testing poisson', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    mean = np.random.rand() * 10
    r = NumC.poisson(inShape, mean)
    print(colored('\tPASS', 'green'))

    print(colored('Testing rand', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    r = NumC.rand(inShape)
    print(colored('\tPASS', 'green'))

    print(colored('Testing randFloat', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    values = np.random.randint(1, 100, [2, ])
    r = NumC.randFloat(inShape, values[0].item(), values[1].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing randInt', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    values = np.random.randint(1, 100, [2, ])
    r = NumC.randInt(inShape, values[0].item(), values[1].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing randN', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    r = NumC.randN(inShape)
    print(colored('\tPASS', 'green'))

    print(colored('Testing seed', 'cyan'))
    NumC.seed(np.random.randint(0, 100000, [1,]).item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing shuffle array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumC.NdArray(shape)
    data = np.random.randint(1, 10, [shape.rows, shape.cols])
    cArray.setArray(data)
    NumC.shuffle(cArray)
    print(colored('\tPASS', 'green'))

    print(colored('Testing standardNormal', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    r = NumC.standardNormal(inShape)
    print(colored('\tPASS', 'green'))

    print(colored('Testing studentT', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    dof = np.random.randint(1, 100, [1, ]).item()
    r = NumC.studentT(inShape, dof)
    print(colored('\tPASS', 'green'))

    print(colored('Testing normal', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    values = np.random.rand(3)
    values.sort()
    r = NumC.triangle(inShape, values[0].item(), values[1].item(), values[2].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing uniform', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    values = np.random.randint(1, 100, [2, ])
    r = NumC.uniform(inShape, values[0].item(), values[1].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing uniformOnSphere', 'cyan'))
    inputs = np.random.randint(1, 100, [2,])
    r = NumC.uniformOnSphere(inputs[0].item(), inputs[1].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing weibull', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumC.Shape(shapeInput[0].item(), shapeInput[1].item())
    inputs = np.random.rand(2)
    r = NumC.weibull(inShape, inputs[0].item(), inputs[1].item())
    print(colored('\tPASS', 'green'))

####################################################################################
if __name__ == '__main__':
    doTest()