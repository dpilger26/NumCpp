# NumCpp: A Templatized Header Only C++ Implementation of the [Python NumPy Library](http://www.numpy.org/)
### <a href='https://dpilger26.github.io/NumCpp'>Full Documentation</a>
## From NumPy To NumCpp – A Quick Start Guide
This quick start guide is meant as a very brief overview of some of the things
that can be done with **NumCpp**.  For a full breakdown of everything available
in the **NumCpp** library please visit the [Full Documentation](https://dpilger26.github.io/NumCpp).

### CONTAINERS
The main data structure in **NumpCpp** is the `NdArray`.  It is inherently a 2D array class, with 1D arrays being implemented as 1xN arrays.  There is also a `DataCube` class that is provided as a convenience container for storing an array of 2D `NdArray`s, but it has limited usefulness past a simple container.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```a = np.array([[3, 4], [5, 6]])```                     | ```NC::NdArray<int> a{{3, 4}, {5, 6}}```                 |
| ```a.reshape([3, 4])```                                  | ```a.reshape(3, 4)```                                    |
| ```a.astype(np.double)```                                | ```a.astype<double>()```                                 |

### INITIALIZERS
Many initializer functions are provided that return `NdArray`s for common needs.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.linspace(1, 10, 5)```	                           | ```NC::linspace<dtype>(1, 10, 5)```                      |
| ```np.arange(3, 7)```                                    | ```NC::arrange<dtype>(3, 7)```                           |
| ```np.eye(4)```                                          | ```NC::eye<dtype>(4)```                                  |
| ```np.zeros([3, 4])```                                   | ```NC::zeros<dtype>(3, 4)```                             |
|                                                          | ```NC::NdArray<dtype>(3, 4) a = 0```                     |
| ```np.ones([3, 4])```                                    | ```NC::ones<dtype>(3, 4)```                              |
|                                                          | ```NC::NdArray<dtype>(3, 4) a = 1```                     |
| ```np.nans([3, 4])```                                    | ```NC::nans<double>(3, 4)```                             |
|                                                          | ```NC::NdArray<double>(3, 4) a = NC::Constants::nan```   |
| ```np.empty([3, 4])```                                   | ```NC::empty<dtype>(3, 4)```                             |
|                                                          | ```NC::NdArray<dtype>(3, 4) a;```                        |

### SLICING/BROADCASTING
**NumpCpp** offers **NumPy** style slicing and broadcasting.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```a[2, 3]```	                                           | ```a(2, 3)```                                            |
| ```a[2:5, 5:8]```                                        | ```a(NC::Slice(2, 5), NC::Slice(5, 8))```                |
| ```a[:, 7]```                                            | ```a(a.rSlice(), 7)```                                   |
| ```a[a > 5]```                                           | ```a[a > 50]```                                          |
| ```a[a > 5] = 0```                                       | ```a.putMask(a > 50, 666)```                             |

### RANDOM
The random module provides simple ways to create random arrays.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.random.seed(666)```	                               | ```NC::Random<>::seed(666)```                            |
| ```np.random.randn(3, 4)```                              | ```NC::Random<double>::randn(NC::Shape(3,4))```          |
| ```np.random.randint(0, 10, [3, 4])```                   | ```NC::Random<int>::randInt(NC::Shape(3,4),0,10)```      |
| ```np.random.rand(3, 4)```                               | ```NC::Random<double>::randn(NC::Shape(3,4))```          |
| ```np.random.choice(a, 3)```                             | ```NC::Random<dtype>::choice(a, 3)```                    |

### CONCATENATION
Many ways to concatenate `NdArray` are available.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.stack([a, b, c], axis=0)```                        | ```NC::stack({a, b, c}, NC::Axis::ROW)```                |
| ```np.vstack([a, b, c])```                               | ```NC::vstack({a, b, c})```                              |
| ```np.hstack([a, b, c])```                               | ```NC::hstack({a, b, c})```                              |
| ```np.append(a, b, axis=1)```	                           | ```NC::append(a, b, NC::Axis::COL)```                    |

### DIAGONAL, TRIANGULAR, AND FLIP
The following return new `NdArray`s.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.diagonal(a)```                                     | ```NC::diagonal(a)```                                    |
| ```np.triu(a)```                                         | ```NC::triu(a)```                                        |
| ```np.tril(a)```                                         | ```NC::tril(a)```                                        |
| ```np.flip(a, axis=0)```	                               | ```NC::flip(a, NC::Axis::ROW)```                         |
| ```np.flipud(a)```                                       | ```NC::flipud(a)```                                      |
| ```np.fliplr(a)```	                                   | ```NC::fliplr(a)```                                      |

### ITERATION
**NumpCpp** follows the idioms of the C++ STL providing iterator pairs to iterate on arrays in different fashions.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```for value in a"""                                     | ```for(auto it = a.begin(); a < a.end(); ++it)```        |
|                                                          | ```for(auto& value : a)```                               |

### LOGICAL
Logical FUNCTIONS in **NumpCpp** behave the same as **NumPy**.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.where(a > 5)```                                    | ```NC::where(a > 5)```                                   |
| ```np.where(a > 5, a, b)```                              | ```NC::where(a > 5, a, b)```                             |
| ```np.any(a)```                                          | ```NC::any(a)```                                         |
| ```np.all(a)```	                                       | ```NC::all(a)```                                         |
| ```np.logical_and(a, b)```                               | ```NC::logical_and(a, b)```                              |
| ```np.logical_or(a, b)```	                               | ```NC::logical_or(a, b)```                               |
| ```np.isclose(a, b)```                                   | ```NC::isclose(a, b)```                                  |
| ```np.allclose(a, b)```		                           | ```NC::allclose(a, b)```                                 |

### COMPARISONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.equal(a, b)```                                     | ```NC:::equal(a, b)```                                   |
| ```np.not_equal(a, b)```                                 | ```NC::not_equal(a, b)```                                |
| ```np.nonzero(a)```                                      | ```NC::nonzero(a)```                                     |

### MINIMUM, MAXIMUM, SORTING

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.min(a)```                                          | ```NC::min(a)```                                         |
| ```np.max(a)```                                          | ```NC::max(a)```                                         |
| ```np.argmin(a)```                                       | ```NC::argmin(a)```                                      |
| ```np.argmax(a)```	                                   | ```NC::argmax(a)```                                      |
| ```np.sort(a, axis=0)```                                 | ```NC::sort(a, NC::Axis::ROW)```                         |
| ```np.argsort(a, axis=1)```                              | ```NC::argsort(a, NC::Axis::COL)```                      |
| ```np.unique(a)```                                       | ```NC::unique(a)```                                      |
| ```np.setdiff1d(a, b)```		                           | ```NC::setdiff1d(a, b)```                                |
| ```np.diff(a)```		                                   | ```NC::diff(a)```                                        |

### REDUCERS
Reducers accumulate values of `NdArray`s along specified axes. When no axis is specified, values are accumulated along all axes.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.sum(a)```                                          | ```NC::sum(a)```                                         |
| ```np.sum(a, axis=0)```                                  | ```NC::sum(a, NC::Axis::ROW)```                          |
| ```np.prod(a)```                                         | ```NC::prod(a)```                                        |
| ```np.prod(a, axis=0)```	                               | ```NC::prod(a, NC::Axis::ROW)```                         |
| ```np.mean(a)```                                         | ```NC::mean(a)```                                        |
| ```np.mean(a, axis=0)```                                 | ```NC::mean(a, NC::Axis::ROW)```                         |
| ```np.count_nonzero(a)```                                | ```NC::count_nonzero(a)```                               |
| ```np.count_nonzero(a, axis=0)```		                   | ```NC::count_nonzero(a, NC::Axis::ROW)```                |

### I/O
Print and file output methods.  All **NumpCpp** classes support a `print()` method and `<<` stream operators.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| print(a)                                                 | ```a.print()```                                          |
|                                                          | ```std::cout << a```                                     |
| ```a.tofile(filename, sep=’\n’)```                       | ````a.tofile(filename, sep=’\n’)```                      |
| ```np.fromfile(filename, sep=’\n’)```	                   | ```NC::fromfile(filename, sep=’\n’)```                   |
| ```np.dump(a, filename)```                               | ```NC::dump(a, filename)```                              |
| ```np.load(filename)```                                  | ```NC::load(filename)```                                 |

### MATHEMATICAL FUNCTIONS
**NumpCpp** universal functions are provided for a large set number of mathematical functions.

#### BASIC FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.absolute(a)```                                     | ```NC::absolute(a)```                                    |
| ```np.sign(a)```                                         | ```NC::sign(a)```                                        |
| ```np.remainder(a, b)```                                 | ```NC::remainder(a, b)```                                |
| ```np.clip(a, min, max)```                               | ```NC::clip(a, min, max)```                              |
| ```np.interp(x, xp, fp)```                               | ```NC::interp(x, xp, fp)```                              |

#### EXPONENTIAL FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.exp(a)```                                          | ```NC::exp(a)```                                         |
| ```np.expm1(a)```	                                       | ```NC::expm1(a)```                                       |
| ```np.log(a)```                                          | ```NC::log(a)```                                         |
| ```np.log1p(a)```                                        | ```NC::log1p(a)```                                       |

#### POWER FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.power(a, p)```                                     | ```NC::power(a)```                                       |
| ```np.sqrt(a)```	                                       | ```NC::sqrt(a)```                                        |
| ```np.square(a)```                                       | ```NC::square(a)```                                      |
| ```np.cbrt(a)```                                         | ```NC::cbrt(a)```                                        |

#### TRIGONOMETRIC FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.sin(a)```                                          | ```NC::sin(a)```                                         |
| ```np.cos(a)```	                                         | ```NC::cos(a)```                                       |
| ```np.tan(a)```                                          | ```NC::tan(a)```                                         |

#### HYPERBOLIC FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.sinh(a)```                                         | ```NC::sinh(a)```                                        |
| ```np.cosh(a)```	                                       | ```NC::cosh(a)```                                        |
| ```np.tanh(a)```                                         | ```NC::tanh(a)```                                        |

#### CLASSIFICATION FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.isnan(a)```                                        | ```NC::isnan(a)```                                       |
| ```np.isinf(a)```	                                       | ```NC::isinf(a)```                                       |
| ```np.isfinite(a)```                                     | ```NC::isfinite(a)```                                    |

#### LINEAR ALGEBRA

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.linalg.norm(a)```                                  | ```NC::norm(a)```                                        |
| ```np.dot(a, b)```	                                   | ```NC::dot(a, b)```                                      |
| ```np.linalg.det(a)```                                   | ```NC::Linalg::det(a)```                                 |
| ```np.linalg.inv(a)```                                   | ```NC::Linalg::inv(a)```                                 |
| ```np.linalg.lstsq(a, b)```	                           | ```NC::Linalg::lstsq(a, b)```                            |
| ```np.linalg.matrix_power(a, 3)```                       | ```NC::Linalg::matrix_power(a, 3)```                     |
| ```Np.linalg..multi_dot(a, b, c)```                      | ```NC::Linalg::multi_dot({a, b, c})```                   |
| ```np.linalg.svd(a)```	                               | ```NC::Linalg::svd(a)```                                 |
