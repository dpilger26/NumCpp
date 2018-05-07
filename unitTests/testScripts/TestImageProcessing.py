import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import scipy.ndimage.filters as filters
import sys
sys.path.append(r'../build/x64/Release')
import NumC

####################################################################################
def doTest():
    print(colored('Testing Image Processing Module', 'magenta'))
    doFilters()
    doCentroiding()

####################################################################################
def doFilters():
    print(colored('Testing Filters', 'magenta'))

    modes = {'reflect' : NumC.Mode.REFLECT,
             'constant': NumC.Mode.CONSTANT,
             'nearest': NumC.Mode.NEAREST,
             'mirror': NumC.Mode.MIRROR,
             'wrap': NumC.Mode.WRAP}

    for mode in modes.keys():
        # print(colored(f'Testing complementaryMedianFilter: mode = {mode}', 'cyan'))
        # shape = np.random.randint(1000, 2000, [2,]).tolist()
        # cShape = NumC.Shape(shape[0], shape[1])
        # cArray = NumC.NdArray(cShape)
        # data = np.random.randint(100, 1000, shape)
        # cArray.setArray(data)
        # kernalSize = 0
        # while kernalSize % 2 == 0:
        #     kernalSize = np.random.randint(5, 15)
        # constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        # dataOutC = NumC.Filter.complementaryMedianFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        # dataOutPy = data - filters.median_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        # if np.array_equal(dataOutC, dataOutPy):
        #     print(colored('\tPASS', 'green'))
        # else:
        #     print(colored('\tFAIL', 'red'))

        print(colored(f'Testing convolve: mode = {mode}', 'cyan'))
        shape = np.random.randint(1000, 2000, [2,]).tolist()
        cShape = NumC.Shape(shape[0], shape[1])
        cArray = NumC.NdArray(cShape)
        data = np.random.randint(100, 1000, shape).astype(np.double)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        weights = np.random.randint(-2, 3, [kernalSize, kernalSize])
        cWeights = NumC.NdArray(1, weights.size)
        cWeights.setArray(weights)
        dataOutC = NumC.Filter.convolve(cArray, kernalSize, cWeights, modes[mode], constantValue).getNumpyArray()
        dataOutPy = filters.convolve(data, weights, mode=mode, cval=constantValue)
        if np.array_equal(dataOutC, dataOutPy):
            print(colored('\tPASS', 'green'))
        else:
            print(colored('\tFAIL', 'red'))

        # print(colored(f'Testing medianFilter: mode = {mode}', 'cyan'))
        # shape = np.random.randint(1000, 2000, [2,]).tolist()
        # cShape = NumC.Shape(shape[0], shape[1])
        # cArray = NumC.NdArray(cShape)
        # data = np.random.randint(100, 1000, shape)
        # cArray.setArray(data)
        # kernalSize = 0
        # while kernalSize % 2 == 0:
        #     kernalSize = np.random.randint(5, 15)
        # constantValue = np.random.randint(0, 5, [1,]).item() # only actaully needed for constant boundary condition
        # dataOutC = NumC.Filter.medianFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        # dataOutPy = filters.median_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        # if np.array_equal(dataOutC, dataOutPy):
        #     print(colored('\tPASS', 'green'))
        # else:
        #     print(colored('\tFAIL', 'red'))

####################################################################################
def doCentroiding():
    print(colored('Testing Centroiding', 'magenta'))

    # generate a random noise
    imageSize = 1024
    noiseStd = np.random.rand(1) * 4
    noiseMean = np.random.randint(75, 100, [1, ]).item()
    noise = np.round(np.random.randn(imageSize, imageSize) * noiseStd + noiseMean)

    # scatter some point sources on it
    pointSize = 5
    pointHalfSize = pointSize // 2
    pointSource = np.asarray([[1,1,1,1,1],[1,5,30,5,1],[1,30,100,30,1],[1,5,30,5,1],[1,1,1,1,1]])

    scene = noise.copy()
    numPointSources = 3000
    for point in range(numPointSources):
        row = np.random.randint(pointHalfSize, imageSize - pointHalfSize, [1, ]).item()
        col = np.random.randint(pointHalfSize, imageSize - pointHalfSize, [1, ]).item()

        cutout = scene[row - pointHalfSize: row + pointHalfSize + 1, col - pointHalfSize: col + pointHalfSize + 1]
        cutout = cutout + pointSource
        scene[row - pointHalfSize: row + pointHalfSize + 1, col - pointHalfSize: col + pointHalfSize + 1] = cutout

    # generate centroids from the image
    thresholdRate = 0.014
    borderWidth = np.random.randint(0, 4, [1,]).item()
    cScene = NumC.NdArray(imageSize)
    cScene.setArray(scene)

    threshold = NumC.generateThreshold(cScene, thresholdRate)
    print(f'Scene Min = {scene.min()}')
    print(f'Scene Max = {scene.max()}')
    print(f'Threshold = {threshold}')
    print(f'Desired Rate = {thresholdRate}')
    print(f'Actual Rate(Threshold) = {np.count_nonzero(scene > threshold) / scene.size}')
    print(f'Actual Rate(Threshold - 1) = {np.count_nonzero(scene > threshold - 1) / scene.size}')

    centroids = list(NumC.generateCentroids(cScene, thresholdRate, 'pre', borderWidth))
    print(f'Window Pre Number of Centroids (Border = {borderWidth}) = {len(centroids)}')

    # plt the results
    plt.figure()
    plt.imshow(scene)
    plt.colorbar()
    plt.clim([threshold, threshold + 1])
    plt.xlabel('Rows')
    plt.ylabel('Cols')
    plt.title(f'Window Pre Centroids\nNumber of Centroids = {len(centroids)}')

    for centroid in centroids:
        plt.plot(centroid.col(), centroid.row(), 'og', fillstyle='none')

    plt.show()

    centroidInfo = np.asarray([[centroid.intensity(), centroid.eod()] for centroid in centroids])

    plt.figure()
    plt.plot(np.sort(centroidInfo[:,0].flatten()))
    plt.title('Window Pre Centroid Intensities')
    plt.xlabel('Centroid #')
    plt.ylabel('Counts')
    plt.show()

    plt.figure()
    plt.plot(np.sort(centroidInfo[:,1].flatten() * 100))
    plt.title('Window Pre Centroid EOD')
    plt.xlabel('Centroid #')
    plt.ylabel('EOD (%)')
    plt.show()

    centroids = list(NumC.generateCentroids(cScene, thresholdRate, 'post', borderWidth))
    print(f'Window Post Number of Centroids (Border = {borderWidth}) = {len(centroids)}')

    # plt the results
    plt.figure()
    plt.imshow(scene)
    plt.colorbar()
    plt.clim([threshold, threshold + 1])
    plt.xlabel('Rows')
    plt.ylabel('Cols')
    plt.title(f'Window Post Centroids\nNumber of Centroids = {len(centroids)}')

    for centroid in centroids:
        plt.plot(centroid.col(), centroid.row(), 'og', fillstyle='none')

    plt.show()

    centroidInfo = np.asarray([[centroid.intensity(), centroid.eod()] for centroid in centroids])

    plt.figure()
    plt.plot(np.sort(centroidInfo[:,0].flatten()))
    plt.title('Window Post Centroid Intensities')
    plt.xlabel('Centroid #')
    plt.ylabel('Counts')
    plt.show()

    plt.figure()
    plt.plot(np.sort(centroidInfo[:,1].flatten() * 100))
    plt.title('Window Post Centroid EOD')
    plt.xlabel('Centroid #')
    plt.ylabel('EOD (%)')
    plt.show()

    print(colored('\tPASS', 'green'))

####################################################################################
if __name__ == '__main__':
    # doCentroiding()
    doFilters()