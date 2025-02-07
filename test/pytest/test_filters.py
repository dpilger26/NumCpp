import numpy as np
import scipy.ndimage as ndimage

import NumCppPy as NumCpp  # noqa E402


modes = {
    "reflect": NumCpp.Mode.REFLECT,
    "constant": NumCpp.Mode.CONSTANT,
    "nearest": NumCpp.Mode.NEAREST,
    "mirror": NumCpp.Mode.MIRROR,
    "wrap": NumCpp.Mode.WRAP,
}


####################################################################################
def test_seed():
    np.random.seed(523)


####################################################################################
def test_complementaryMeanFilter1d():
    for mode in modes.keys():
        size = np.random.randint(
            1000,
            2000,
            [
                1,
            ],
        ).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(
            100,
            1000,
            [
                size,
            ],
        )
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = (
            NumCpp.complementaryMeanFilter1d(cArray, kernalSize, modes[mode], constantValue).getNumpyArray().flatten()
        )
        dataOutPy = data - ndimage.generic_filter(
            data.astype(float),
            np.mean,
            footprint=np.ones(
                [
                    kernalSize,
                ]
            ),
            mode=mode,
            cval=constantValue,
        )
        assert np.array_equal(np.round(dataOutC, 8), np.round(dataOutPy, 8))


####################################################################################
def test_complementaryMedianFilter1d():
    for mode in modes.keys():
        size = np.random.randint(
            1000,
            2000,
            [
                1,
            ],
        ).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(
            100,
            1000,
            [
                size,
            ],
        )
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = (
            NumCpp.complementaryMedianFilter1d(cArray, kernalSize, modes[mode], constantValue).getNumpyArray().flatten()
        )
        dataOutPy = data - ndimage.generic_filter(
            data,
            np.median,
            footprint=np.ones(
                [
                    kernalSize,
                ]
            ),
            mode=mode,
            cval=constantValue,
        )
        assert np.array_equal(dataOutC, dataOutPy)


####################################################################################
def test_convolve1d():
    for mode in modes.keys():
        size = np.random.randint(
            1000,
            2000,
            [
                1,
            ],
        ).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(
            100,
            1000,
            [
                size,
            ],
        ).astype(float)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        weights = np.random.randint(
            1,
            5,
            [
                kernalSize,
            ],
        )
        cWeights = NumCpp.NdArray(1, kernalSize)
        cWeights.setArray(weights)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.convolve1d(cArray, cWeights, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = ndimage.convolve(data, weights, mode=mode, cval=constantValue)
        assert np.array_equal(np.round(dataOutC, 8), np.round(dataOutPy, 8))


####################################################################################
def test_gaussianFilter1d():
    for mode in modes.keys():
        size = np.random.randint(
            1000,
            2000,
            [
                1,
            ],
        ).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(
            100,
            1000,
            [
                size,
            ],
        ).astype(float)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        sigma = np.random.rand(1).item() * 2
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.gaussianFilter1d(cArray, sigma, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = ndimage.gaussian_filter(data, sigma, mode=mode, cval=constantValue)
        assert np.array_equal(np.round(dataOutC, 7), np.round(dataOutPy, 7))


####################################################################################
def test_maximumFilter1d():
    for mode in modes.keys():
        size = np.random.randint(
            1000,
            2000,
            [
                1,
            ],
        ).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(
            100,
            1000,
            [
                size,
            ],
        )
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.maximumFilter1d(cArray, kernalSize, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = ndimage.generic_filter(
            data,
            np.max,
            footprint=np.ones(
                [
                    kernalSize,
                ]
            ),
            mode=mode,
            cval=constantValue,
        )
        assert np.array_equal(dataOutC, dataOutPy)


####################################################################################
def test_meanFilter1d():
    for mode in modes.keys():
        size = np.random.randint(
            1000,
            2000,
            [
                1,
            ],
        ).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(
            100,
            1000,
            [
                size,
            ],
        )
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.meanFilter1d(cArray, kernalSize, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = ndimage.generic_filter(
            data.astype(float),
            np.mean,
            footprint=np.ones(
                [
                    kernalSize,
                ]
            ),
            mode=mode,
            cval=constantValue,
        )
        assert np.array_equal(np.round(dataOutC, 8), np.round(dataOutPy, 8))


####################################################################################
def test_medianFilter1d():
    for mode in modes.keys():
        size = np.random.randint(
            1000,
            2000,
            [
                1,
            ],
        ).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(
            100,
            1000,
            [
                size,
            ],
        )
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.medianFilter1d(cArray, kernalSize, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = ndimage.generic_filter(
            data,
            np.median,
            footprint=np.ones(
                [
                    kernalSize,
                ]
            ),
            mode=mode,
            cval=constantValue,
        )
        assert np.array_equal(dataOutC, dataOutPy)


####################################################################################
def test_minumumFilter1d():
    for mode in modes.keys():
        size = np.random.randint(
            1000,
            2000,
            [
                1,
            ],
        ).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(
            100,
            1000,
            [
                size,
            ],
        )
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.minumumFilter1d(cArray, kernalSize, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = ndimage.generic_filter(
            data,
            np.min,
            footprint=np.ones(
                [
                    kernalSize,
                ]
            ),
            mode=mode,
            cval=constantValue,
        )
        assert np.array_equal(dataOutC, dataOutPy)


####################################################################################
def test_percentileFilter1d():
    for mode in modes.keys():
        size = np.random.randint(
            1000,
            2000,
            [
                1,
            ],
        ).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(
            100,
            1000,
            [
                size,
            ],
        ).astype(float)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        percentile = np.random.randint(
            0,
            101,
            [
                1,
            ],
        ).item()
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = (
            NumCpp.percentileFilter1d(cArray, kernalSize, percentile, modes[mode], constantValue)
            .getNumpyArray()
            .flatten()
        )
        dataOutPy = ndimage.generic_filter(
            data,
            np.percentile,
            footprint=np.ones(
                [
                    kernalSize,
                ]
            ),
            mode=mode,
            cval=constantValue,
            extra_arguments=(percentile,),
        )
        assert np.array_equal(np.round(dataOutC, 5), np.round(dataOutPy, 5))


####################################################################################
def test_rankFilter1d():
    for mode in modes.keys():
        size = np.random.randint(
            1000,
            2000,
            [
                1,
            ],
        ).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(
            100,
            1000,
            [
                size,
            ],
        ).astype(float)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15).item()
        rank = np.random.randint(
            0,
            kernalSize - 1,
            [
                1,
            ],
        ).item()
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.rankFilter1d(cArray, kernalSize, rank, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = ndimage.rank_filter(
            data,
            rank,
            footprint=np.ones(
                [
                    kernalSize,
                ]
            ),
            mode=mode,
            cval=constantValue,
        )
        # assert np.array_equal(dataOutC, dataOutPy)


####################################################################################
def test_uniformFilter1d():
    for mode in modes.keys():
        size = np.random.randint(
            1000,
            2000,
            [
                1,
            ],
        ).item()
        cShape = NumCpp.Shape(1, size)
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(
            100,
            1000,
            [
                size,
            ],
        ).astype(float)
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.uniformFilter1d(cArray, kernalSize, modes[mode], constantValue).getNumpyArray().flatten()
        dataOutPy = ndimage.generic_filter(
            data,
            np.mean,
            footprint=np.ones(
                [
                    kernalSize,
                ]
            ),
            mode=mode,
            cval=constantValue,
        )
        assert np.array_equal(dataOutC, dataOutPy)


####################################################################################
def test_complementaryMeanFilter():
    for mode in modes.keys():
        shape = np.random.randint(
            100,
            200,
            [
                2,
            ],
        ).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])  # noqa
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)  # noqa
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.complementaryMeanFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = data - ndimage.generic_filter(
            data.astype(float),
            np.mean,
            footprint=np.ones(
                [kernalSize, kernalSize],
            ),
            mode=mode,
            cval=constantValue,
        )
        assert np.array_equal(np.round(dataOutC, 8), np.round(dataOutPy, 8))


####################################################################################
def test_complementaryMedianFilter():
    for mode in modes.keys():
        shape = np.random.randint(
            100,
            200,
            [
                2,
            ],
        ).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])  # noqa
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)  # noqa
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.complementaryMedianFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = data - ndimage.median_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        assert np.array_equal(dataOutC, dataOutPy)


####################################################################################
def test_convolve():
    for mode in modes.keys():
        shape = np.random.randint(
            100,
            200,
            [
                2,
            ],
        ).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])  # noqa
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(10, 20, shape).astype(float)  # noqa
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        weights = np.random.randint(-2, 3, [kernalSize, kernalSize]).astype(float)
        cWeights = NumCpp.NdArray(kernalSize)
        cWeights.setArray(weights)
        dataOutC = NumCpp.convolve(cArray, kernalSize, cWeights, modes[mode], constantValue).getNumpyArray()
        dataOutPy = ndimage.convolve(data, weights, mode=mode, cval=constantValue)
        assert np.array_equal(dataOutC, dataOutPy)


####################################################################################
def test_gaussianFilter():
    for mode in modes.keys():
        shape = np.random.randint(
            100,
            200,
            [
                2,
            ],
        ).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])  # noqa
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape).astype(float)  # noqa
        cArray.setArray(data)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        sigma = np.random.rand(1).item() * 2
        dataOutC = NumCpp.gaussianFilter(cArray, sigma, modes[mode], constantValue).getNumpyArray()
        dataOutPy = ndimage.gaussian_filter(data, sigma, mode=mode, cval=constantValue)
        assert np.array_equal(np.round(dataOutC, 2), np.round(dataOutPy, 2))


####################################################################################
def test_laplaceFilter():
    for mode in modes.keys():
        shape = np.random.randint(
            100,
            200,
            [
                2,
            ],
        ).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])  # noqa
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape).astype(float)  # noqa
        cArray.setArray(data)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.laplaceFilter(cArray, modes[mode], constantValue).getNumpyArray()
        dataOutPy = ndimage.laplace(data, mode=mode, cval=constantValue)
        assert np.array_equal(dataOutC, dataOutPy)


####################################################################################
def test_maximumFilter():
    for mode in modes.keys():
        shape = np.random.randint(
            100,
            200,
            [
                2,
            ],
        ).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])  # noqa
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)  # noqa
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.maximumFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = ndimage.maximum_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        assert np.array_equal(dataOutC, dataOutPy)


####################################################################################
def test_mean_filter():
    for mode in modes.keys():
        shape = np.random.randint(
            100,
            200,
            [
                2,
            ],
        ).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])  # noqa
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)  # noqa
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.meanFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = ndimage.generic_filter(
            data.astype(float),
            np.mean,
            footprint=np.ones(
                [kernalSize, kernalSize],
            ),
            mode=mode,
            cval=constantValue,
        )
        assert np.array_equal(np.round(dataOutC, 8), np.round(dataOutPy, 8))


####################################################################################
def test_median_filter():
    for mode in modes.keys():
        shape = np.random.randint(
            100,
            200,
            [
                2,
            ],
        ).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])  # noqa
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)  # noqa
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.medianFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = ndimage.median_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        assert np.array_equal(dataOutC, dataOutPy)


####################################################################################
def test_minimum_filter():
    for mode in modes.keys():
        shape = np.random.randint(
            100,
            200,
            [
                2,
            ],
        ).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])  # noqa
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)  # noqa
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.minimumFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = ndimage.minimum_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        assert np.array_equal(dataOutC, dataOutPy)


####################################################################################
def test_percentileFilter():
    for mode in modes.keys():
        shape = np.random.randint(
            100,
            200,
            [
                2,
            ],
        ).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])  # noqa
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)  # noqa
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        percentile = np.random.randint(
            0,
            101,
            [
                1,
            ],
        ).item()
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.percentileFilter(cArray, kernalSize, percentile, modes[mode], constantValue).getNumpyArray()
        dataOutPy = ndimage.percentile_filter(data, percentile, size=kernalSize, mode=mode, cval=constantValue)
        assert np.array_equal(dataOutC, dataOutPy)


####################################################################################
def test_rankFilter():
    for mode in modes.keys():
        shape = np.random.randint(
            100,
            200,
            [
                2,
            ],
        ).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])  # noqa
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape)  # noqa
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        rank = np.random.randint(
            0,
            kernalSize**2 - 1,
            [
                1,
            ],
        ).item()
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.rankFilter(cArray, kernalSize, rank, modes[mode], constantValue).getNumpyArray()
        dataOutPy = ndimage.rank_filter(data, rank, size=kernalSize, mode=mode, cval=constantValue)
        assert np.array_equal(dataOutC, dataOutPy)


####################################################################################
def test_uniform_filter():
    for mode in modes.keys():
        shape = np.random.randint(
            100,
            200,
            [
                2,
            ],
        ).tolist()
        cShape = NumCpp.Shape(shape[0], shape[1])  # noqa
        cArray = NumCpp.NdArray(cShape)
        data = np.random.randint(100, 1000, shape).astype(float)  # noqa
        cArray.setArray(data)
        kernalSize = 0
        while kernalSize % 2 == 0:
            kernalSize = np.random.randint(5, 15)
        # only actually needed for constant boundary condition
        constantValue = np.random.randint(
            0,
            5,
            [
                1,
            ],
        ).item()
        dataOutC = NumCpp.uniformFilter(cArray, kernalSize, modes[mode], constantValue).getNumpyArray()
        dataOutPy = ndimage.uniform_filter(data, size=kernalSize, mode=mode, cval=constantValue)
        assert np.array_equal(np.round(dataOutC, 8), np.round(dataOutPy, 8))
