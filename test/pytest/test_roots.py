import numpy as np
import os
import sys
sys.path.append(os.path.abspath(r'../lib'))
import NumCpp  # noqa E402

np.random.seed(666)


####################################################################################
def test_bisection():
    root = np.random.randint(-50, 50, [1, ]).item()
    roots = np.array([root, root + np.random.randint(5, 50, [1, ]).item()])
    largestRoot = np.max(roots).item()
    rootsC = NumCpp.NdArray(1, roots.size)
    rootsC.setArray(roots)
    polyC = NumCpp.Poly1d(rootsC, True)
    rootC = int(np.round(NumCpp.bisection_roots(polyC, largestRoot - 1, largestRoot + 1)))
    assert rootC == largestRoot


####################################################################################
def test_brent():
    root = np.random.randint(-50, 50, [1, ]).item()
    roots = np.array([root, root + np.random.randint(5, 50, [1, ]).item()])
    largestRoot = np.max(roots).item()
    rootsC = NumCpp.NdArray(1, roots.size)
    rootsC.setArray(roots)
    polyC = NumCpp.Poly1d(rootsC, True)
    rootC = int(np.round(NumCpp.brent_roots(polyC, largestRoot - 1, largestRoot + 1)))
    assert rootC == largestRoot


####################################################################################
def test_dekker():
    root = np.random.randint(-50, 50, [1, ]).item()
    roots = np.array([root, root + np.random.randint(5, 50, [1, ]).item()])
    largestRoot = np.max(roots).item()
    rootsC = NumCpp.NdArray(1, roots.size)
    rootsC.setArray(roots)
    polyC = NumCpp.Poly1d(rootsC, True)
    rootC = int(np.round(NumCpp.dekker_roots(polyC, largestRoot - 1, largestRoot + 1)))
    assert rootC == largestRoot


####################################################################################
def test_newton():
    root = np.random.randint(-50, 50, [1, ]).item()
    roots = np.array([root, root + np.random.randint(5, 50, [1, ]).item()])
    largestRoot = np.max(roots).item()
    rootsC = NumCpp.NdArray(1, roots.size)
    rootsC.setArray(roots)
    polyC = NumCpp.Poly1d(rootsC, True)
    rootC = int(np.round(NumCpp.newton_roots(polyC, largestRoot)))
    assert rootC == largestRoot


####################################################################################
def test_secant():
    root = np.random.randint(-50, 50, [1, ]).item()
    roots = np.array([root, root + np.random.randint(5, 50, [1, ]).item()])
    largestRoot = np.max(roots).item()
    rootsC = NumCpp.NdArray(1, roots.size)
    rootsC.setArray(roots)
    polyC = NumCpp.Poly1d(rootsC, True)
    rootC = int(np.round(NumCpp.secant_roots(polyC, largestRoot - 1, largestRoot + 1)))
    assert rootC == largestRoot
