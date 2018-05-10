import numpy as np
from termcolor import colored
import scipy.ndimage.filters as filters
import sys
sys.path.append(r'../build/x64/Release')
import NumC

####################################################################################
def doTest():
    print(colored('Testing Filters', 'magenta'))

    modes = {'reflect' : NumC.Mode.REFLECT,
             'constant': NumC.Mode.CONSTANT,
             'nearest': NumC.Mode.NEAREST,
             'mirror': NumC.Mode.MIRROR,
             'wrap': NumC.Mode.WRAP}

    for mode in modes.keys():
        print(colored(f'Testing complementaryMedianFilter: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumC.Shape(shape[0], shape[1])
        cArray = NumC.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumC.Filter.complementaryMedianFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = data - filters.median_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing convolve: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumC.Shape(shape[0], shape[1])
        cArray = NumC.NdArray(cShape)
        data = np.random.randint(10, 20, shape).astype(np.double)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        weights = np.random.randint(-2, 3, [kernalSize, kernalSize]).astype(np.double)
        cWeights = NumC.NdArray(kernalSize)
        cWeights.setArray(weights)
        dataOutC = NumC.Filter.convolve(cArray, kernalSize, cWeights, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.convolve(data, weights, mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing gaussianFilter: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumC.Shape(shape[0], shape[1])
        cArray = NumC.NdArray(cShape)
        data = np.random.randint(100, 1000, shape).astype(np.double)
        cArray.setArray(data)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        sigma = np.random.rand(1).item() * 2
        dataOutC = NumC.Filter.gaussianFilter(cArray, sigma, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.gaussian_filter(data, sigma, mode=mode, cval=constantValue)
        if np.array_equal(np.round(dataOutC, 5), np.round(dataOutPy, 5)):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing maximumFilter: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumC.Shape(shape[0], shape[1])
        cArray = NumC.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumC.Filter.maximumFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.maximum_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing medianFilter: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumC.Shape(shape[0], shape[1])
        cArray = NumC.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumC.Filter.medianFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.median_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing minimumFilter: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumC.Shape(shape[0], shape[1])
        cArray = NumC.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumC.Filter.minimumFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.minimum_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing percentileFilter: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumC.Shape(shape[0], shape[1])
        cArray = NumC.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        percentile = np.random.randint(0, 101, [1,]).item()
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumC.Filter.percentileFilter(cArray, kernalSize, percentile, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.percentile_filter(data, percentile, size=kernalSize, mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing rankFilter: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumC.Shape(shape[0], shape[1])
        cArray = NumC.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        rank = np.random.randint(0, kernalSize**2 - 1, [1,]).item()
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumC.Filter.rankFilter(cArray, kernalSize, rank, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.rank_filter(data, rank, size=kernalSize, mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        print(colored(f'Testing uniformFilter: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumC.Shape(shape[0], shape[1])
        cArray = NumC.NdArray(cShape)
        data = np.random.randint(100, 1000, shape).astype(np.double)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        dataOutC = NumC.Filter.uniformFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.uniform_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        if np.array_equal(np.round(dataOutC, 8), np.round(dataOutPy, 8)):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

####################################################################################
if __name__ == '__main__':
    doTest()