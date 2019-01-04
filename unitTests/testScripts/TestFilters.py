import numpy as np
from termcolor import colored
import scipy.ndimage.filters as filters
import sys
if sys.platform == 'linux':
    sys.path.append(r'../lib')
else:
    sys.path.append(r'../build/x64/Release')
import NumCpp

####################################################################################
def doTest():
    print(colored('Testing Filters', 'magenta'))
    test1D()
    test2D()

####################################################################################
def test1D():
    modes = {'reflect' : NumCpp.Mode.REFLECT,
             'constant': NumCpp.Mode.CONSTANT,
             'nearest': NumCpp.Mode.NEAREST,
             'mirror': NumCpp.Mode.MIRROR,
             'wrap': NumCpp.Mode.WRAP}

    for mode in modes.keys():
        print(colored(f'Testing complementaryMedianFilter1d: mode = {mode}', 'cyan'))
        size = np.random.randint(1000, 2000, [1,]).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, [size,])
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumCpp.complementaryMedianFilter1d(cArray, kernalSize, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = data - filters.generic_filter(data, np.median, footprint=np.ones([kernalSize,]), mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing convolve1d: mode = {mode}', 'cyan'))
        size = np.random.randint(1000, 2000, [1,]).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, [size,]).astype(np.double)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        weights = np.random.randint(1, 5, [kernalSize,])
        cWeights = NumCpp.NdArray(1, kernalSize)
        cWeights.setArray(weights)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumCpp.convolve1d(cArray, cWeights, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = filters.convolve(data, weights, mode=mode, cval=constantValue)
        if np.array_equal(np.round(dataOutC, 8), np.round(dataOutPy, 8)):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing gaussianFilter1d: mode = {mode}', 'cyan'))
        size = np.random.randint(1000, 2000, [1,]).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, [size,]).astype(np.double)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        sigma = np.random.rand(1).item() * 2
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumCpp.gaussianFilter1d(cArray, sigma, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = filters.gaussian_filter(data, sigma, mode=mode, cval=constantValue)
        if np.array_equal(np.round(dataOutC, 7), np.round(dataOutPy, 7)):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing maximumFilter1d: mode = {mode}', 'cyan'))
        size = np.random.randint(1000, 2000, [1,]).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, [size,])
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumCpp.maximumFilter1d(cArray, kernalSize, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = filters.generic_filter(data, np.max, footprint=np.ones([kernalSize,]), mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing medianFilter1d: mode = {mode}', 'cyan'))
        size = np.random.randint(1000, 2000, [1,]).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, [size,])
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumCpp.medianFilter1d(cArray, kernalSize, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = filters.generic_filter(data, np.median, footprint=np.ones([kernalSize,]), mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing minumumFilter1d: mode = {mode}', 'cyan'))
        size = np.random.randint(1000, 2000, [1,]).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, [size,])
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumCpp.minumumFilter1d(cArray, kernalSize, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = filters.generic_filter(data, np.min, footprint=np.ones([kernalSize,]), mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing percentileFilter1d: mode = {mode}', 'cyan'))
        size = np.random.randint(1000, 2000, [1,]).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, [size,]).astype(np.double)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        percentile = np.random.randint(0, 101, [1,]).item()
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumCpp.percentileFilter1d(cArray, kernalSize, percentile, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = filters.generic_filter(data, np.percentile, footprint=np.ones([kernalSize,]), mode=mode, cval=constantValue, extra_arguments=(percentile,))
        if np.array_equal(np.round(dataOutC, 7), np.round(dataOutPy, 7)):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing rankFilter1d: mode = {mode}', 'cyan'))
        size = np.random.randint(1000, 2000, [1,]).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, [size,]).astype(np.double)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        rank = np.random.randint(0, kernalSize - 1, [1, ]).item()
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumCpp.rankFilter1d(cArray, kernalSize, rank, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = filters.rank_filter(data, rank, footprint=np.ones([kernalSize,]), mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing uniformFilter1d: mode = {mode}', 'cyan'))
        size = np.random.randint(1000, 2000, [1,]).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, [size,]).astype(np.double)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumCpp.uniformFilter1d(cArray, kernalSize, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = filters.generic_filter(data, np.mean, footprint=np.ones([kernalSize,]), mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

####################################################################################
def test2D():
    modes = {'reflect' : NumCpp.Mode.REFLECT,
             'constant': NumCpp.Mode.CONSTANT,
             'nearest': NumCpp.Mode.NEAREST,
             'mirror': NumCpp.Mode.MIRROR,
             'wrap': NumCpp.Mode.WRAP}

    for mode in modes.keys():
        print(colored(f'Testing complementaryMedianFilter: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumCpp.complementaryMedianFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = data - filters.median_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing convolve: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(10, 20, shape).astype(np.double)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        weights = np.random.randint(-2, 3, [kernalSize, kernalSize]).astype(np.double)
        cWeights = NumCpp.NdArray(kernalSize)
        cWeights.setArray(weights)
        dataOutC = NumCpp.convolve(cArray, kernalSize, cWeights, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.convolve(data, weights, mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing gaussianFilter: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape).astype(np.double)
        cArray.setArray(data)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        sigma = np.random.rand(1).item() * 2
        dataOutC = NumCpp.gaussianFilter(cArray, sigma, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.gaussian_filter(data, sigma, mode=mode, cval=constantValue)
        if np.array_equal(np.round(dataOutC, 2), np.round(dataOutPy, 2)):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing maximumFilter: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumCpp.maximumFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.maximum_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing medianFilter: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumCpp.medianFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.median_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing minimumFilter: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumCpp.minimumFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.minimum_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing percentileFilter: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        percentile = np.random.randint(0, 101, [1,]).item()
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumCpp.percentileFilter(cArray, kernalSize, percentile, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.percentile_filter(data, percentile, size=kernalSize, mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing rankFilter: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        rank = np.random.randint(0, kernalSize**2 - 1, [1,]).item()
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumCpp.rankFilter(cArray, kernalSize, rank, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.rank_filter(data, rank, size=kernalSize, mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing uniformFilter: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape).astype(np.double)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumCpp.uniformFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.uniform_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        if np.array_equal(np.round(dataOutC, 8), np.round(dataOutPy, 8)):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

####################################################################################
if __name__ == '__main__':
    # test1D()
    # test2D()
    doTest()
