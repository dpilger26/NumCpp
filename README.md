# NumCpp
## C++ implementation of the [Python NumPy Library](http://www.numpy.org/)

### [Full Documentation](https://dpilger26.github.io/NumCpp)

## From NumPy To NumCpp – A Quick Start Guide
```c++
namespace nc = NumCpp;
```

### CONTAINERS
The main data structure in **NumpCpp** is the `NdArray`.  It is inherently a 2D array class, with 1D arrays being implemented as 1xN arrays.  There is also a `DataCube` class that is provided as a convenience container for storing an array of 2D `NdArray`s, but it has limited usefulness past a simple container.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| a = np.array([[3, 4], [5, 6]])                           | nc::NdArray&lt;int&gt; a{{3, 4}, {5, 6}}                 |
| a.reshape([3, 4])                                        | a.reshape(3, 4)                                          |
| a.astype(np.double)                                      | a.astype&lt;double&gt;()                                 |

### INITIALIZERS
Many initializer FUNCTIONS are provided that return `NdArray`s for common needs.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.linspace(1, 10, 2)	                                   | nc::Methods&lt;int&gt;::linspace(1, 10, 2)               |
| np.arange(3, 7)                                          | nc::Methods&lt;int&gt;::arrange(3, 7)                    |
| np.eye(4)                                                | nc::Methods&lt;int&gt;::eye(4)                           |
| np.zeros([3, 4])                                         | nc::Methods&lt;int&gt;::zeros(3, 4)                      |
| np.ones([3, 4])                                          | nc::Methods&lt;int&gt;::ones(3, 4)                       |
| np.nans([3, 4])                                          | nc::Methods&lt;int&gt;::nans(3, 4)                       |
| np.empty([3, 4])                                         | nc::Methods&lt;int&gt;::empty(3, 4)                      |

### SLICING/BROADCASTING
**NumpCpp** offers **NumPy** style slicing and broadcasting.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| a[2, 3]	                                                 | a(2, 3)                                                  |
| a[2:5, 5:8]                                              | a(nc::Slice(2, 5), nc::Slice(5, 8))                      |
| a[:, 7]                                                  | a(nc::Slice::all(), 7)                                   |
| a[a &gt; 5]                                              | a[a &gt; 5]                                              |
| a[a &gt; 5] = 0                                          | a.put(a &gt; 5, 0)                                       |

### RANDOM
The random module provides simple ways to create random arrays.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.random.seed(666)	                                     | nc::Random&lt;&gt;::seed(666)                            |
| np.random.randn(3, 4)                                    | nc::Random&lt;double&gt;::randn(nc::Shape(3,4))          |
| np.random.randint(0, 10, [3, 4])                         | nc::Random&lt;int&gt;::randInt(nc::Shape(3,4),0,10)      |
| np.random.rand(3, 4)                                     | nc::Random&lt;double&gt;::randn(nc::Shape(3,4))          |
| np.random.choice(a, 3)                                   | nc::Random&lt;dtype&gt;::choice(a, 3)                    |

### COncATENATION
Many ways to concatenate `NdArray` are available.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.stack([a, b, c], axis=0)                              | nc::Methods&lt;dtype&gt;::stack({a,b,c},nc::Axis::ROW)   |
| np.vstack([a, b, c])                                     | nc::Methods&lt;dtype&gt;::vstack({a, b, c})              |
| np.hstack([a, b, c])                                     | nc::Methods&lt;dtype&gt;::hstack({a, b, c})              |
| np.append(a, b, axis=1)	                                 | nc::Methods&lt;dtype&gt;::append(a,b,nc::Axis::COL)      |

### DIAGONAL, TRIANGULAR, AND FLIP
The following return new `NdArray`s.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.diagonal(a)                                           | nc::Methods&lt;dtype&gt;::diagonal(a)                    |
| np.triu(a)                                               | nc::Methods&lt;dtype&gt;::triu(a)                        |
| np.tril(a)                                               | nc::Methods&lt;dtype&gt;::tril(a)                        |
| np.flip(a, axis=0)	                                     | nc::Methods&lt;dtype&gt;::flip(a, nc::Axis::ROW)         |
| np.flipud(a)                                             | nc::Methods&lt;dtype&gt;::flipud(a)                      |
| np.fliplr(a)	                                           | nc::Methods&lt;dtype&gt;::fliplr(a)                      |

### ITERATION
**NumpCpp** follows the idioms of the C++ STL providing iterator pairs to iterate on arrays in different fashions.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| for value in a                                           | for(auto it = a.begin(); a &lt; a.end(); ++it)           |
|                                                          | for(auto& value : a)                                     |

### LOGICAL
Logical FUNCTIONS in **NumpCpp** behave the same as **NumPy**.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.where(a &gt; 5)                                       | nc::Methods&lt;dtype&gt;::where(a &gt; 5)                |
| np.where(a &gt; 5, a, b)                                 | nc::Methods&lt;dtype&gt;::where(a &gt; 5, a, b)          |
| np.any(a)                                                | nc::Methods&lt;dtype&gt;::any(a)                         |
| np.all(a)	                                               | nc::Methods&lt;dtype&gt;::all(a)                         |
| np.logical_and(a, b)                                     | nc::Methods&lt;dtype&gt;::logical_and(a, b)              |
| np.logical_or(a, b)	                                     | nc::Methods&lt;dtype&gt;::logical_or(a, b)               |
| np.isclose(a, b)                                         | nc::Methods&lt;dtype&gt;::isclose(a, b)                  |
| np.allclose(a, b)		                                     | nc::Methods&lt;dtype&gt;::allclose(a, b)                 |

### COMPARISONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.equal(a, b)                                           | nc::Methods&lt;dtype&gt;::equal(a, b)                    |
| np.not_equal(a, b)                                       | nc::Methods&lt;dtype&gt;::not_equal(a, b)                |
| np.nonzero(a)                                            | nc::Methods&lt;dtype&gt;::nonzero(a)                     |

### MINIMUM, MAXIMUM, SORTING

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.min(a)                                                | nc::Methods&lt;dtype&gt;::min(a)                         |
| np.max(a)                                                | nc::Methods&lt;dtype&gt;::max(a)                         |
| np.argmin(a)                                             | nc::Methods&lt;dtype&gt;::argmin(a)                      |
| np.argmax(a)	                                           | nc::Methods&lt;dtype&gt;::argmax(a)                      |
| np.sort(a, axis=0)                                       | nc::Methods&lt;dtype&gt;::sort(a, nc::Axis::ROW)         |
| np.argsort(a, axis=1)                                    | nc::Methods&lt;dtype&gt;::argsort(a, nc::Axis::COL)      |
| np.unique(a)                                             | nc::Methods&lt;dtype&gt;::unique(a)                      |
| np.setdiff1d(a, b)		                                   | nc::Methods&lt;dtype&gt;::setdiff1d(a, b)                |
| np.diff(a)		                                           | nc::Methods&lt;dtype&gt;::diff(a)                        |

### REDUCERS
Reducers accumulate values of `NdArray`s along specified axes. When no axis is specified, values are accumulated along all axes.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.sum(a)                                                | nc::Methods&lt;dtype&gt;::sum(a)                         |
| np.sum(a, axis=0)                                        | nc::Methods&lt;dtype&gt;::sum(a, nc::Axis::ROW)          |
| np.prod(a)                                               | nc::Methods&lt;dtype&gt;::prod(a)                        |
| np.prod(a, axis=0)	                                     | nc::Methods&lt;dtype&gt;::prod(a, nc::Axis::ROW)         |
| np.mean(a)                                               | nc::Methods&lt;dtype&gt;::mean(a)                        |
| np.mean(a, axis=0)                                       | nc::Methods&lt;dtype&gt;::mean(a, nc::Axis::ROW)         |
| np.count_nonzero(a)                                      | nc::Methods&lt;dtype&gt;::count_nonzero(a)               |
| np.count_nonzero(a, axis=0)		                           | nc::Methods&lt;dtype&gt;::count_nonzero(a,nc::Axis::ROW) |

### I/O
Print and file output methods.  All **NumpCpp** classes support a `print()` method and `&lt;&lt;` stream operators.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| print(a)                                                 | a.print()                                                |
|                                                          | std::cout &lt;&lt; a                                     |
| a.tofile(filename, sep=’\n’)                             | 	a.tofile(filename, sep=’\n’)                            |
| np.fromfile(filename, sep=’\n’)	                         | nc::Methods&lt;dtype&gt;::fromfile(filename, sep=’\n’)   |
| np.dump(a, filename)                                     | nc::Methods&lt;dtype&gt;::dump(a, filename)              |
| np.load(filename)                                        | nc::Methods&lt;dtype&gt;::load(filename)                 |

### MATHEMATICAL FUNCTIONS
**NumpCpp** universal FUNCTIONS are provided for a large set number of mathematical FUNCTIONS.

#### BASIC FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.absolute(a)                                           | nc::Methods&lt;dtype&gt;::absolute(a)                    |
| np.sign(a)                                               | nc::Methods&lt;dtype&gt;::sign(a)                        |
| np.remainder(a, b)                                       | nc::Methods&lt;dtype&gt;::remainder(a, b)                |
| np.clip(a, min, max)                                     | nc::Methods&lt;dtype&gt;::clip(a, min, max)              |
| np.interp(x, xp, fp)                                     | nc::Methods&lt;dtype&gt;::interp(x, xp, fp)              |

#### EXPONENTIAL FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.exp(a)                                                | nc::Methods&lt;dtype&gt;::exp(a)                         |
| np.expm1(a)	                                             | nc::Methods&lt;dtype&gt;::expm1(a)                       |
| np.log(a)                                                | nc::Methods&lt;dtype&gt;::log(a)                         |
| np.log1p(a)                                              | nc::Methods&lt;dtype&gt;::log1p(a)                       |

#### POWER FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.power(a, p)                                           | nc::Methods&lt;dtype&gt;::power(a)                       |
| np.sqrt(a)	                                             | nc::Methods&lt;dtype&gt;::sqrt(a)                        |
| np.square(a)                                             | nc::Methods&lt;dtype&gt;::square(a)                      |
| np.cbrt(a)                                               | nc::Methods&lt;dtype&gt;::cbrt(a)                        |

#### TRIGONOMETRIC FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.sin(a)                                                | nc::Methods&lt;dtype&gt;::sin(a)                         |
| np.cos(a)	                                               | nc::Methods&lt;dtype&gt;::cos(a)                         |
| np.tan(a)                                                | nc::Methods&lt;dtype&gt;::tan(a)                         |

#### HYPERBOLIC FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.sinh(a)                                               | nc::Methods&lt;dtype&gt;::sinh(a)                        |
| np.cosh(a)	                                             | nc::Methods&lt;dtype&gt;::cosh(a)                        |
| np.tanh(a)                                               | nc::Methods&lt;dtype&gt;::tanh(a)                        |

#### CLASSIFICATION FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.isnan(a)                                              | nc::Methods&lt;dtype&gt;::isnan(a)                       |
| np.isinf(a)	                                             | nc::Methods&lt;dtype&gt;::isinf(a)                       |
| np.isfinite(a)                                           | nc::Methods&lt;dtype&gt;::isfinite(a)                    |

#### LINEAR ALGEBRA

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.linalg.norm(a)                                        | nc::Methods&lt;dtype&gt;::norm(a)                        |
| np.dot(a, b)	                                           | nc::Methods&lt;dtype&gt;::dot(a, b)                      |
| np.linalg.det(a)                                         | nc::Linalg&lt;dtype&gt;::det(a)                          |
| np.linalg.inv(a)                                         | nc::Linalg&lt;dtype&gt;::inv(a)                          |
| np.linalg.lstsq(a, b)	                                   | nc::Linalg&lt;dtype&gt;::lstsq(a, b)                     |
| np.linalg.matrix_power(a, 3)                             | nc::Linalg&lt;dtype&gt;::matrix_power(a, 3)              |
| Np.linalg..multi_dot(a, b, c)                            | nc::Linalg&lt;dtype&gt;::multi_dot({a, b, c})            |
| np.linalg.svd(a)	                                       | nc::Linalg&lt;dtype&gt;::svd(a)                          |
