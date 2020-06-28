import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(r'../lib'))
import NumCppPy as NumCpp  # noqa E402

np.random.seed(666)

PLOT_SHOW = False


####################################################################################
def test_imageProcessing():
    # generate a random noise
    imageSize = 512
    noiseStd = np.random.rand(1) * 4
    noiseMean = np.random.randint(75, 100, [1, ]).item()
    noise = np.round(np.random.randn(imageSize, imageSize) * noiseStd + noiseMean)

    # scatter some point sources on it
    pointSize = 5
    pointHalfSize = pointSize // 2
    pointSource = np.asarray([[1, 1, 1, 1, 1],
                              [1, 5, 30, 5, 1],
                              [1, 30, 100, 30, 1],
                              [1, 5, 30, 5, 1],
                              [1, 1, 1, 1, 1]])

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
    borderWidth = np.random.randint(0, 4, [1, ]).item()
    cScene = NumCpp.NdArray(imageSize)
    cScene.setArray(scene)

    threshold = NumCpp.generateThreshold(cScene, thresholdRate)
    print(f'Scene Min = {scene.min()}')
    print(f'Scene Max = {scene.max()}')
    print(f'Threshold = {threshold}')
    print(f'Desired Rate = {thresholdRate}')
    print(f'Actual Rate(Threshold) = {np.count_nonzero(scene > threshold) / scene.size}')
    print(f'Actual Rate(Threshold - 1) = {np.count_nonzero(scene > threshold - 1) / scene.size}')

    centroids = list(NumCpp.generateCentroids(cScene, thresholdRate, 'pre', borderWidth))
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

    if PLOT_SHOW:
        plt.show()

    centroidInfo = np.asarray([[centroid.intensity(), centroid.eod()] for centroid in centroids])

    plt.figure()
    plt.plot(np.sort(centroidInfo[:, 0].flatten()))
    plt.title('Window Pre Centroid Intensities')
    plt.xlabel('Centroid #')
    plt.ylabel('Counts')
    if PLOT_SHOW:
        plt.show()

    plt.figure()
    plt.plot(np.sort(centroidInfo[:, 1].flatten() * 100))
    plt.title('Window Pre Centroid EOD')
    plt.xlabel('Centroid #')
    plt.ylabel('EOD (%)')
    if PLOT_SHOW:
        plt.show()

    centroids = list(NumCpp.generateCentroids(cScene, thresholdRate, 'post', borderWidth))
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

    if PLOT_SHOW:
        plt.show()

    centroidInfo = np.asarray([[centroid.intensity(), centroid.eod()] for centroid in centroids])

    plt.figure()
    plt.plot(np.sort(centroidInfo[:, 0].flatten()))
    plt.title('Window Post Centroid Intensities')
    plt.xlabel('Centroid #')
    plt.ylabel('Counts')
    if PLOT_SHOW:
        plt.show()

    plt.figure()
    plt.plot(np.sort(centroidInfo[:, 1].flatten() * 100))
    plt.title('Window Post Centroid EOD')
    plt.xlabel('Centroid #')
    plt.ylabel('EOD (%)')
    if PLOT_SHOW:
        plt.show()

    plt.close('all')
