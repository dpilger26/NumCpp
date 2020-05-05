import numpy as np
import os
import sys
if sys.platform == 'linux':
    sys.path.append(r'../lib')
else:
    sys.path.append(os.path.abspath('../build/x64/Release'))
import NumCpp


####################################################################################
def test_roots():
    root = np.random.randint(-50, 50, [1, ]).item()
    roots = np.array([root, root + np.random.randint(5, 50, [1, ]).item()])
    largestRoot = np.max(roots).item()
    rootsC = NumCpp.NdArray(1, roots.size)
    rootsC.setArray(roots)
    polyC = NumCpp.Poly1d(rootsC, True)
    rootC = int(np.round(NumCpp.bisection_roots(polyC, largestRoot - 1, largestRoot + 1)))
    assert rootC == largestRoot

    root = np.random.randint(-50, 50, [1, ]).item()
    roots = np.array([root, root + np.random.randint(5, 50, [1, ]).item()])
    largestRoot = np.max(roots).item()
    rootsC = NumCpp.NdArray(1, roots.size)
    rootsC.setArray(roots)
    polyC = NumCpp.Poly1d(rootsC, True)
    rootC = int(np.round(NumCpp.brent_roots(polyC, largestRoot - 1, largestRoot + 1)))
    assert rootC == largestRoot

    root = np.random.randint(-50, 50, [1, ]).item()
    roots = np.array([root, root + np.random.randint(5, 50, [1, ]).item()])
    largestRoot = np.max(roots).item()
    rootsC = NumCpp.NdArray(1, roots.size)
    rootsC.setArray(roots)
    polyC = NumCpp.Poly1d(rootsC, True)
    rootC = int(np.round(NumCpp.dekker_roots(polyC, largestRoot - 1, largestRoot + 1)))
    assert rootC == largestRoot

    root = np.random.randint(-50, 50, [1, ]).item()
    roots = np.array([root, root + np.random.randint(5, 50, [1, ]).item()])
    largestRoot = np.max(roots).item()
    rootsC = NumCpp.NdArray(1, roots.size)
    rootsC.setArray(roots)
    polyC = NumCpp.Poly1d(rootsC, True)
    rootC = int(np.round(NumCpp.newton_roots(polyC, largestRoot)))
    assert rootC == largestRoot

    root = np.random.randint(-50, 50, [1, ]).item()
    roots = np.array([root, root + np.random.randint(5, 50, [1, ]).item()])
    largestRoot = np.max(roots).item()
    rootsC = NumCpp.NdArray(1, roots.size)
    rootsC.setArray(roots)
    polyC = NumCpp.Poly1d(rootsC, True)
    rootC = int(np.round(NumCpp.secant_roots(polyC, largestRoot - 1, largestRoot + 1)))
    assert rootC == largestRoot
