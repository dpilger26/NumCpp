import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import sys
sys.path.append(r'../build/x64/Release')
import NumC

####################################################################################
def doTest():
    print(colored('Testing Image Processing Module', 'magenta'))

    # generate a random noise
    imageSize = 1024
    noiseStd = 4
    noiseMean = 100
    noise = np.round(np.random.randn(imageSize, imageSize) * noiseStd + noiseMean)

    # clip any negative values at zero
    noise[noise < 0] = 0

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
    windowType = 'pre'
    borderWidth = 1
    cScene = NumC.NdArray(imageSize)
    cScene.setArray(scene)

    threshold = NumC.generateThreshold(cScene, thresholdRate)
    print(f'Threshold = {threshold}')

    centroids = list(NumC.generateCentroids(cScene, thresholdRate, windowType, borderWidth))

    # plt the results
    plt.figure()
    plt.imshow(scene)
    plt.colorbar()
    plt.clim([threshold, threshold + 1])
    plt.xlabel('Rows')
    plt.ylabel('Cols')
    plt.title(f'Centroids\nNumber of Centroids = {len(centroids)}')

    for centroid in centroids:
        plt.plot(centroid.col(), centroid.row(), 'og')

    plt.show()

    centroidInfo = [[centroid.intensity(), centroid.eod()] for centroid in centroids]

    plt.figure()
    plt.plot(sorted(centroidInfo[:,0]))
    plt.title('Centroid Intensities')
    plt.xlabel('Centroid #')
    plt.ylabel('Counts')
    plt.show()

    plt.figure()
    plt.plot(sorted(centroidInfo[:,1] * 100))
    plt.title('Centroid EOD')
    plt.xlabel('Centroid #')
    plt.ylabel('EOD (%)')
    plt.show()

    print(colored('\tPASS', 'green'))

####################################################################################
if __name__ == '__main__':
    doTest()