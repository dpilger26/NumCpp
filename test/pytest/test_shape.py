import numpy as np

import NumCppPy as NumCpp  # noqa E402


####################################################################################
def test_default_constructor():
    shape = NumCpp.Shape()
    assert shape.rows == 0 and shape.cols == 0


####################################################################################
def test_square_constructor():
    shapeInput = np.random.randint(0, 100, [1, ]).item()
    shape = NumCpp.Shape(shapeInput)
    assert shape.rows == shapeInput
    assert shape.cols == shapeInput


####################################################################################
def test_rec_constructor():
    shapeInput = np.random.randint(0, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    assert shape.rows == shapeInput[0]
    assert shape.cols == shapeInput[1]


####################################################################################
def test_copy_constructor():
    shapeInput = np.random.randint(0, 100, [2, ])
    shape = NumCpp.Shape(shapeInput[0].item(), shapeInput[1].item())
    assert shape.rows == shapeInput[0]
    assert shape.cols == shapeInput[1]

    shape2 = NumCpp.Shape(shape)
    assert shape2.rows == shape.rows
    assert shape2.cols == shape.cols


####################################################################################
def test_print():
    shape = NumCpp.Shape()
    shape.print()
