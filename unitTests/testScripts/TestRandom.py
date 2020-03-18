import numpy as np
from termcolor import colored
import sys
if sys.platform == 'linux':
    sys.path.append(r'../lib')
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
    r = NumCpp.bernoulli(inShape, p)
    print(colored('\tPASS', 'green'))

    print(colored('Testing bernoulli scalar', 'cyan'))
    r = NumCpp.bernoulli(p)
    print(colored('\tPASS', 'green'))

    print(colored('Testing beta', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    alpha = np.random.rand()
    beta = np.random.rand()
    r = NumCpp.beta(inShape, alpha, beta)
    print(colored('\tPASS', 'green'))

    print(colored('Testing beta scalar', 'cyan'))
    r = NumCpp.beta(alpha, beta)
    print(colored('\tPASS', 'green'))

    print(colored('Testing binomial', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    n = np.random.randint(1, 100, [1,]).item()
    p = np.random.rand()
    r = NumCpp.binomial(inShape, n, p)
    print(colored('\tPASS', 'green'))

    print(colored('Testing binomial scalar', 'cyan'))
    r = NumCpp.binomial(n, p)
    print(colored('\tPASS', 'green'))

    print(colored('Testing cauchy', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    mean = np.random.randn() * 10
    sigma = np.random.rand() * 10
    r = NumCpp.cauchy(inShape, mean, sigma)
    print(colored('\tPASS', 'green'))

    print(colored('Testing cauchy scalar', 'cyan'))
    r = NumCpp.cauchy( mean, sigma)
    print(colored('\tPASS', 'green'))

    print(colored('Testing chiSquare', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    dof = np.random.randint(1, 100, [1,]).item()
    r = NumCpp.chiSquare(inShape, dof)
    print(colored('\tPASS', 'green'))

    print(colored('Testing chiSquare scalar', 'cyan'))
    r = NumCpp.chiSquare(dof)
    print(colored('\tPASS', 'green'))

    print(colored('Testing choice: Single', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    r = NumCpp.choiceSingle(cArray)
    print(colored('\tPASS', 'green'))

    print(colored('Testing choice: Multiple', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.rand(shape.rows, shape.cols)
    cArray.setArray(data)
    num = np.random.randint(1, data.size, [1,]).item()
    r = NumCpp.choiceMultiple(cArray, num)
    if r.size == num:
        print(colored('\tPASS', 'green'))

    print(colored('Testing discrete', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    weights = np.random.randint(1, 10, [shape.rows, shape.cols])
    cArray.setArray(weights)
    r = NumCpp.discrete(inShape, cArray)
    print(colored('\tPASS', 'green'))

    print(colored('Testing discrete scalar', 'cyan'))
    r = NumCpp.discrete(cArray)
    print(colored('\tPASS', 'green'))

    print(colored('Testing exponential', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    scale = np.random.rand() * 10
    r = NumCpp.exponential(inShape, scale)
    print(colored('\tPASS', 'green'))

    print(colored('Testing exponential scalar', 'cyan'))
    r = NumCpp.exponential(scale)
    print(colored('\tPASS', 'green'))

    print(colored('Testing extremeValue', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    a = np.random.rand() * 10
    b = np.random.rand() * 100
    r = NumCpp.extremeValue(inShape, a, b)
    print(colored('\tPASS', 'green'))

    print(colored('Testing extremeValue scalar', 'cyan'))
    r = NumCpp.extremeValue(a, b)
    print(colored('\tPASS', 'green'))

    print(colored('Testing f', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    dofN = np.random.rand() * 10
    dofD = np.random.rand() * 100
    r = NumCpp.f(inShape, dofN, dofD)
    print(colored('\tPASS', 'green'))

    print(colored('Testing f scalar', 'cyan'))
    r = NumCpp.f(dofN, dofD)
    print(colored('\tPASS', 'green'))

    print(colored('Testing gamma', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    shape = np.random.rand() * 10
    scale = np.random.rand() * 100
    r = NumCpp.gamma(inShape, shape, scale)
    print(colored('\tPASS', 'green'))

    print(colored('Testing gamma scalar', 'cyan'))
    r = NumCpp.gamma(shape, scale)
    print(colored('\tPASS', 'green'))

    print(colored('Testing geometric', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    p = np.random.rand()
    r = NumCpp.geometric(inShape, p)
    print(colored('\tPASS', 'green'))

    print(colored('Testing geometric scalar', 'cyan'))
    r = NumCpp.geometric(p)
    print(colored('\tPASS', 'green'))

    print(colored('Testing laplace', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    loc = np.random.rand() * 10
    scale = np.random.rand() * 100
    r = NumCpp.laplace(inShape, loc, scale)
    print(colored('\tPASS', 'green'))

    print(colored('Testing laplace scalar', 'cyan'))
    r = NumCpp.laplace(loc, scale)
    print(colored('\tPASS', 'green'))

    print(colored('Testing lognormal', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    mean = np.random.randn() * 10
    sigma = np.random.rand() * 10
    r = NumCpp.lognormal(inShape, mean, sigma)
    print(colored('\tPASS', 'green'))

    print(colored('Testing lognormal scalar', 'cyan'))
    r = NumCpp.lognormal(mean, sigma)
    print(colored('\tPASS', 'green'))

    print(colored('Testing negativeBinomial', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    n = np.random.randint(1, 100, [1,]).item()
    p = np.random.rand()
    r = NumCpp.negativeBinomial(inShape, n, p)
    print(colored('\tPASS', 'green'))

    print(colored('Testing negativeBinomial scalar', 'cyan'))
    r = NumCpp.negativeBinomial(n, p)
    print(colored('\tPASS', 'green'))

    print(colored('Testing nonCentralChiSquared', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    k = np.random.rand() * 10
    l = np.random.rand() * 100
    r = NumCpp.nonCentralChiSquared(inShape, k, l)
    print(colored('\tPASS', 'green'))

    print(colored('Testing nonCentralChiSquared scalar', 'cyan'))
    r = NumCpp.nonCentralChiSquared(k, l)
    print(colored('\tPASS', 'green'))

    print(colored('Testing normal', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    mean = np.random.randn() * 10
    sigma = np.random.rand() * 10
    r = NumCpp.normal(inShape, mean, sigma)
    print(colored('\tPASS', 'green'))

    print(colored('Testing normal scalar', 'cyan'))
    r = NumCpp.normal(mean, sigma)
    print(colored('\tPASS', 'green'))

    print(colored('Testing permutation scalar', 'cyan'))
    r = NumCpp.permutationScaler(np.random.randint(1,100, [1,]).item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing permutation array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 10, [shape.rows, shape.cols])
    cArray.setArray(data)
    r = NumCpp.permutationArray(cArray)
    print(colored('\tPASS', 'green'))

    print(colored('Testing poisson', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    mean = np.random.rand() * 10
    r = NumCpp.poisson(inShape, mean)
    print(colored('\tPASS', 'green'))

    print(colored('Testing poisson scalar', 'cyan'))
    r = NumCpp.poisson(mean)
    print(colored('\tPASS', 'green'))

    print(colored('Testing rand', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    r = NumCpp.rand(inShape)
    print(colored('\tPASS', 'green'))

    print(colored('Testing rand scalar', 'cyan'))
    r = NumCpp.rand()
    print(colored('\tPASS', 'green'))

    print(colored('Testing randFloat', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    values = np.random.randint(1, 100, [2, ])
    values.sort()
    r = NumCpp.randFloat(inShape, values[0].item(), values[1].item() + 1)
    print(colored('\tPASS', 'green'))

    print(colored('Testing randFloat scalar', 'cyan'))
    r = NumCpp.randFloat(values[0].item(), values[1].item() + 1)
    print(colored('\tPASS', 'green'))

    print(colored('Testing randInt', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item() + 1)
    values = np.random.randint(1, 100, [2, ])
    values.sort()
    r = NumCpp.randInt(inShape, values[0].item(), values[1].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing randInt scalar', 'cyan'))
    r = NumCpp.randInt(values[0].item(), values[1].item() + 1)
    print(colored('\tPASS', 'green'))

    print(colored('Testing randN', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    r = NumCpp.randN(inShape)
    print(colored('\tPASS', 'green'))

    print(colored('Testing randN scalar', 'cyan'))
    r = NumCpp.randN()
    print(colored('\tPASS', 'green'))

    print(colored('Testing seed', 'cyan'))
    NumCpp.seed(np.random.randint(0, 100000, [1,]).item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing shuffle array', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    cArray = NumCpp.NdArray(shape)
    data = np.random.randint(1, 10, [shape.rows, shape.cols])
    cArray.setArray(data)
    NumCpp.shuffle(cArray)
    print(colored('\tPASS', 'green'))

    print(colored('Testing standardNormal', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    r = NumCpp.standardNormal(inShape)
    print(colored('\tPASS', 'green'))

    print(colored('Testing standardNormal scalar', 'cyan'))
    r = NumCpp.standardNormal()
    print(colored('\tPASS', 'green'))

    print(colored('Testing studentT', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    dof = np.random.randint(1, 100, [1, ]).item()
    r = NumCpp.studentT(inShape, dof)
    print(colored('\tPASS', 'green'))

    print(colored('Testing studentT scalar', 'cyan'))
    r = NumCpp.studentT(dof)
    print(colored('\tPASS', 'green'))

    print(colored('Testing triangle', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    values = np.random.rand(3)
    values.sort()
    r = NumCpp.triangle(inShape, values[0].item(), values[1].item(), values[2].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing triangle scalar', 'cyan'))
    r = NumCpp.triangle(values[0].item(), values[1].item(), values[2].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing uniform', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    values = np.random.randint(1, 100, [2, ])
    values.sort()
    r = NumCpp.uniform(inShape, values[0].item(), values[1].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing uniform scalar', 'cyan'))
    r = NumCpp.uniform(values[0].item(), values[1].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing uniformOnSphere', 'cyan'))
    inputs = np.random.randint(1, 100, [2,])
    r = NumCpp.uniformOnSphere(inputs[0].item(), inputs[1].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing weibull', 'cyan'))
    shapeInput = np.random.randint(1, 100, [2,])
    inShape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    inputs = np.random.rand(2)
    r = NumCpp.weibull(inShape, inputs[0].item(), inputs[1].item())
    print(colored('\tPASS', 'green'))

    print(colored('Testing weibull scalar', 'cyan'))
    r = NumCpp.weibull(inputs[0].item(), inputs[1].item())
    print(colored('\tPASS', 'green'))

####################################################################################
if __name__ == '__main__':
    doTest()
