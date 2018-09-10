# NumCpp: A Templatized Header Only C++ Implementation of the [Python NumPy Library](http://www.numpy.org/)

## From NumPy To NumCpp – A Quick Start Guide
This quick start guide is meant as a very brief overview of some of the things
that can be done with **NumCpp**.  For a full breakdown of everything available
in the **NumCpp** library please visit the [Full Documentation](https://dpilger26.github.io/NumCpp).

```c++
namespace nc = NumCpp;
```

### CONTAINERS
The main data structure in **NumpCpp** is the `NdArray`.  It is inherently a 2D array class, with 1D arrays being implemented as 1xN arrays.  There is also a `DataCube` class that is provided as a convenience container for storing an array of 2D `NdArray`s, but it has limited usefulness past a simple container.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```a = np.array([[3, 4], [5, 6]])```                     | ```nc::NdArray<int> a{{3, 4}, {5, 6}}```                 |
| ```a.reshape([3, 4])```                                  | ```a.reshape(3, 4)```                                    |
| ```a.astype(np.double)```                                | ```a.astype<double>()```                                 |

### INITIALIZERS
Many initializer functions are provided that return `NdArray`s for common needs.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.linspace(1, 10, 2)```	                             | ```nc::Methods<int>::linspace(1, 10, 2)```               |
| ```np.arange(3, 7)```                                    | ```nc::Methods<int>::arrange(3, 7)```                    |
| ```np.eye(4)```                                          | ```nc::Methods<int>::eye(4)```                           |
| ```np.zeros([3, 4])```                                   | ```nc::Methods<int>::zeros(3, 4)```                      |
| ```np.ones([3, 4])```                                    | ```nc::Methods<int>::ones(3, 4)```                       |
| ```np.nans([3, 4])```                                    | ```nc::Methods<int>::nans(3, 4)```                       |
| ```np.empty([3, 4])```                                   | ```nc::Methods<int>::empty(3, 4)```                      |

### SLICING/BROADCASTING
**NumpCpp** offers **NumPy** style slicing and broadcasting.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```a[2, 3]```	                                           | ```a(2, 3)```                                            |
| ```a[2:5, 5:8]```                                        | ```a(nc::Slice(2, 5), nc::Slice(5, 8))```                |
| ```a[:, 7]```                                            | ```a(nc::Slice::all(), 7)```                             |
| ```a[a > 5]```                                           | ```a[a > 5]```                                           |
| ```a[a > 5] = 0```                                       | ```a.put(a > 5, 0)```                                    |

### RANDOM
The random module provides simple ways to create random arrays.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.random.seed(666)```	                               | ```nc::Random<>::seed(666)```                            |
| ```np.random.randn(3, 4)```                              | ```nc::Random<double>::randn(nc::Shape(3,4))```          |
| ```np.random.randint(0, 10, [3, 4])```                   | ```nc::Random<int>::randInt(nc::Shape(3,4),0,10)```      |
| ```np.random.rand(3, 4)```                               | ```nc::Random<double>::randn(nc::Shape(3,4))```          |
| ```np.random.choice(a, 3)```                             | ```nc::Random<dtype>::choice(a, 3)```                    |

### CONCATENATION
Many ways to concatenate `NdArray` are available.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.stack([a, b, c], axis=0)```                        | ```nc::Methods<dtype>::stack({a, b, c}, nc::Axis::ROW)```|
| ```np.vstack([a, b, c])```                               | ```nc::Methods<dtype>::vstack({a, b, c})```              |
| ```np.hstack([a, b, c])```                               | ```nc::Methods<dtype>::hstack({a, b, c})```              |
| ```np.append(a, b, axis=1)```	                           | ```nc::Methods<dtype>::append(a, b, nc::Axis::COL)```    |

### DIAGONAL, TRIANGULAR, AND FLIP
The following return new `NdArray`s.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.diagonal(a)```                                     | ```nc::Methods<dtype>::diagonal(a)```                    |
| ```np.triu(a)```                                         | ```nc::Methods<dtype>::triu(a)```                        |
| ```np.tril(a)```                                         | ```nc::Methods<dtype>::tril(a)```                        |
| ```np.flip(a, axis=0)```	                               | ```nc::Methods<dtype>::flip(a, nc::Axis::ROW)```         |
| ```np.flipud(a)```                                       | ```nc::Methods<dtype>::flipud(a)```                      |
| ```np.fliplr(a)```	                                     | ```nc::Methods<dtype>::fliplr(a)```                      |

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
| ```np.where(a > 5)```                                    | ```nc::Methods<dtype>::where(a > 5)```                   |
| ```np.where(a > 5, a, b)```                              | ```nc::Methods<dtype>::where(a > 5, a, b)```             |
| ```np.any(a)```                                          | ```nc::Methods<dtype>::any(a)```                         |
| ```np.all(a)```	                                         | ```nc::Methods<dtype>::all(a)```                         |
| ```np.logical_and(a, b)```                               | ```nc::Methods<dtype>::logical_and(a, b)```              |
| ```np.logical_or(a, b)```	                               | ```nc::Methods<dtype>::logical_or(a, b)```               |
| ```np.isclose(a, b)```                                   | ```nc::Methods<dtype>::isclose(a, b)```                  |
| ```np.allclose(a, b)```		                               | ```nc::Methods<dtype>::allclose(a, b)```                 |

### COMPARISONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.equal(a, b)```                                     | ```nc::Methods<dtype>::equal(a, b)```                    |
| ```np.not_equal(a, b)```                                 | ```nc::Methods<dtype>::not_equal(a, b)```                |
| ```np.nonzero(a)```                                      | ```nc::Methods<dtype>::nonzero(a)```                     |

### MINIMUM, MAXIMUM, SORTING

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.min(a)```                                          | ```nc::Methods<dtype>::min(a)```                         |
| ```np.max(a)```                                          | ```nc::Methods<dtype>::max(a)```                         |
| ```np.argmin(a)```                                       | ```nc::Methods<dtype>::argmin(a)```                      |
| ```np.argmax(a)```	                                     | ```nc::Methods<dtype>::argmax(a)```                      |
| ```np.sort(a, axis=0)```                                 | ```nc::Methods<dtype>::sort(a, nc::Axis::ROW)```         |
| ```np.argsort(a, axis=1)```                              | ```nc::Methods<dtype>::argsort(a, nc::Axis::COL)```      |
| ```np.unique(a)```                                       | ```nc::Methods<dtype>::unique(a)```                      |
| ```np.setdiff1d(a, b)```		                             | ```nc::Methods<dtype>::setdiff1d(a, b)```                |
| ```np.diff(a)```		                                     | ```nc::Methods<dtype>::diff(a)```                        |

### REDUCERS
Reducers accumulate values of `NdArray`s along specified axes. When no axis is specified, values are accumulated along all axes.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.sum(a)```                                          | ```nc::Methods<dtype>::sum(a)```                         |
| ```np.sum(a, axis=0)```                                  | ```nc::Methods<dtype>::sum(a, nc::Axis::ROW)```          |
| ```np.prod(a)```                                         | ```nc::Methods<dtype>::prod(a)```                        |
| ```np.prod(a, axis=0)```	                               | ```nc::Methods<dtype>::prod(a, nc::Axis::ROW)```         |
| ```np.mean(a)```                                         | ```nc::Methods<dtype>::mean(a)```                        |
| ```np.mean(a, axis=0)```                                 | ```nc::Methods<dtype>::mean(a, nc::Axis::ROW)```         |
| ```np.count_nonzero(a)```                                | ```nc::Methods<dtype>::count_nonzero(a)```               |
| ```np.count_nonzero(a, axis=0)```		                     | ```nc::Methods<dtype>::count_nonzero(a, nc::Axis::ROW)```|

### I/O
Print and file output methods.  All **NumpCpp** classes support a `print()` method and `<<` stream operators.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| print(a)                                                 | ```a.print()```                                          |
|                                                          | ```std::cout << a```                                     |
| ```a.tofile(filename, sep=’\n’)```                       | 	```a.tofile(filename, sep=’\n’)```                      |
| ```np.fromfile(filename, sep=’\n’)```	                   | ```nc::Methods<dtype>::fromfile(filename, sep=’\n’)```   |
| ```np.dump(a, filename)```                               | ```nc::Methods<dtype>::dump(a, filename)```              |
| ```np.load(filename)```                                  | ```nc::Methods<dtype>::load(filename)```                 |

### MATHEMATICAL FUNCTIONS
**NumpCpp** universal functions are provided for a large set number of mathematical functions.

#### BASIC FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.absolute(a)```                                     | ```nc::Methods<dtype>::absolute(a)```                    |
| ```np.sign(a)```                                         | ```nc::Methods<dtype>::sign(a)```                        |
| ```np.remainder(a, b)```                                 | ```nc::Methods<dtype>::remainder(a, b)```                |
| ```np.clip(a, min, max)```                               | ```nc::Methods<dtype>::clip(a, min, max)```              |
| ```np.interp(x, xp, fp)```                               | ```nc::Methods<dtype>::interp(x, xp, fp)```              |

#### EXPONENTIAL FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.exp(a)```                                          | ```nc::Methods<dtype>::exp(a)```                         |
| ```np.expm1(a)```	                                       | ```nc::Methods<dtype>::expm1(a)```                       |
| ```np.log(a)```                                          | ```nc::Methods<dtype>::log(a)```                         |
| ```np.log1p(a)```                                        | ```nc::Methods<dtype>::log1p(a)```                       |

#### POWER FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.power(a, p)```                                     | ```nc::Methods<dtype>::power(a)```                       |
| ```np.sqrt(a)```	                                       | ```nc::Methods<dtype>::sqrt(a)```                        |
| ```np.square(a)```                                       | ```nc::Methods<dtype>::square(a)```                      |
| ```np.cbrt(a)```                                         | ```nc::Methods<dtype>::cbrt(a)```                        |

#### TRIGONOMETRIC FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.sin(a)```                                          | ```nc::Methods<dtype>::sin(a)```                         |
| ```np.cos(a)```	                                         | ```nc::Methods<dtype>::cos(a)```                         |
| ```np.tan(a)```                                          | ```nc::Methods<dtype>::tan(a)```                         |

#### HYPERBOLIC FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.sinh(a)```                                         | ```nc::Methods<dtype>::sinh(a)```                        |
| ```np.cosh(a)```	                                       | ```nc::Methods<dtype>::cosh(a)```                        |
| ```np.tanh(a)```                                         | ```nc::Methods<dtype>::tanh(a)```                        |

#### CLASSIFICATION FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.isnan(a)```                                        | ```nc::Methods<dtype>::isnan(a)```                       |
| ```np.isinf(a)```	                                       | ```nc::Methods<dtype>::isinf(a)```                       |
| ```np.isfinite(a)```                                     | ```nc::Methods<dtype>::isfinite(a)```                    |

#### LINEAR ALGEBRA

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.linalg.norm(a)```                                  | ```nc::Methods<dtype>::norm(a)```                        |
| ```np.dot(a, b)```	                                     | ```nc::Methods<dtype>::dot(a, b)```                      |
| ```np.linalg.det(a)```                                   | ```nc::Linalg<dtype>::det(a)```                          |
| ```np.linalg.inv(a)```                                   | ```nc::Linalg<dtype>::inv(a)```                          |
| ```np.linalg.lstsq(a, b)```	                             | ```nc::Linalg<dtype>::lstsq(a, b)```                     |
| ```np.linalg.matrix_power(a, 3)```                       | ```nc::Linalg<dtype>::matrix_power(a, 3)```              |
| ```Np.linalg..multi_dot(a, b, c)```                      | ```nc::Linalg<dtype>::multi_dot({a, b, c})```            |
| ```np.linalg.svd(a)```	                                 | ```nc::Linalg<dtype>::svd(a)```                          |
