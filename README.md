# NumCpp
## C++ implementation of the [Python NumPy Library](http://www.numpy.org/)

### [Full Documentation](https://dpilger26.github.io/NumCpp)

## From NumPy To NumCpp – A Quick Start Guide
```c++
namespace NC = NumCpp;
```

### CONTAINERS
The main data structure in **NumpCpp** is the `NdArray`.  It is inherently a 2D array class, with 1D arrays being implemented as 1xN arrays.  There is also a `DataCube` class that is provided as a convenience container for storing an array of 2D `NdArray`s, but it has limited usefulness past a simple container.

| **NumPy****                                              | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| a = np.array([[3, 4], [5, 6]])                           | NC::`NdArray`&lt;int&gt; a{{3, 4}, {5, 6}}                 |
| a.reshape([3, 4])                                        | a.reshape(3, 4)                                          |
| a.astype(np.double)                                      | a.astype&lt;double&gt;()                                 |

### INITIALIZERS
Many initializer functions are provided that return `NdArray`s for common needs.

| **NumPy****                                              | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.linspace(1, 10, 2)	                                   | NC::Methods&lt;int&gt;::linspace(1, 10, 2)               |
| np.arange(3, 7)                                          | NC::Methods&lt;int&gt;::arrange(3, 7)                    |
| np.eye(4)                                                | NC::Methods&lt;int&gt;::eye(4)                           |
| np.zeros([3, 4])                                         | NC::Methods&lt;int&gt;::zeros(3, 4)                      |
| np.ones([3, 4])                                          | NC::Methods&lt;int&gt;::ones(3, 4)                       |
| np.nans([3, 4])                                          | NC::Methods&lt;int&gt;::nans(3, 4)                       |
| np.empty([3, 4])                                         | NC::Methods&lt;int&gt;::empty(3, 4)                      |

### SLICING/BROADCASTING
**NumpCpp** offers **NumPy** style slicing and broadcasting.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| a[2, 3]	                                                 | a(2, 3)                                                  |
| a[2:5, 5:8]                                              | a(NC::Slice(2, 5), NC::Slice(5, 8))                      |
| a[:, 7]                                                  | a(NC::Slice::all(), 7)                                   |
| a[a &gt; 5]                                              | a[a &gt; 5]                                              |
| a[a &gt; 5] = 0                                          | a.put(a &gt; 5, 0)                                       |

### RANDOM
The random module provides simple ways to create random arrays.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.random.seed(666)	                                     | NC::Random&lt;&gt;::seed(666)                            |
| np.random.randn(3, 4)                                    | NC::Random&lt;double&gt;::randn(nc::Shape(3,4))          |
| np.random.randint(0, 10, [3, 4])                         | NC::Random&lt;int&gt;::randInt(nc::Shape(3,4),0,10)      |
| np.random.rand(3, 4)                                     | NC::Random&lt;double&gt;::randn(nc::Shape(3,4))          |
| np.random.choice(a, 3)                                   | NC::Random&lt;dtype&gt;::choice(a, 3)                    |

### CONCATENATION
Many ways to concatenate `NdArray` are available.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.stack([a, b, c], axis=0)                              | NC::Methods&lt;dtype&gt;::stack({a,b,c},NC::Axis::ROW)   |
| np.vstack([a, b, c])                                     | NC::Methods&lt;dtype&gt;::vstack({a, b, c})              |
| np.hstack([a, b, c])                                     | NC::Methods&lt;dtype&gt;::hstack({a, b, c})              |
| np.append(a, b, axis=1)	                                 | NC::Methods&lt;dtype&gt;::append(a,b,NC::Axis::COL)      |

### DIAGONAL, TRIANGULAR, AND FLIP
The following return new `NdArray`s.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.diagonal(a)                                           | NC::Methods&lt;dtype&gt;::diagonal(a)                    |
| np.triu(a)                                               | NC::Methods&lt;dtype&gt;::triu(a)                        |
| np.tril(a)                                               | NC::Methods&lt;dtype&gt;::tril(a)                        |
| np.flip(a, axis=0)	                                     | NC::Methods&lt;dtype&gt;::flip(a, NC::Axis::ROW)         |
| np.flipud(a)                                             | NC::Methods&lt;dtype&gt;::flipud(a)                      |
| np.fliplr(a)	                                           | NC::Methods&lt;dtype&gt;::fliplr(a)                      |

### ITERATION
**NumpCpp** follows the idioms of the C++ STL providing iterator pairs to iterate on arrays in different fashions.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| for value in a                                           | for(auto it = a.begin(); a &lt; a.end(); ++it)           |
|                                                          | for(auto& value : a)                                     |

### LOGICAL
Logical functions in **NumpCpp** behave the same as **NumPy**.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.where(a &gt; 5)                                       | NC::Methods&lt;dtype&gt;::where(a &gt; 5)                |
| np.where(a &gt; 5, a, b)                                 | NC::Methods&lt;dtype&gt;::where(a &gt; 5, a, b)          |
| np.any(a)                                                | NC::Methods&lt;dtype&gt;::any(a)                         |
| np.all(a)	                                               | NC::Methods&lt;dtype&gt;::all(a)                         |
| np.logical_and(a, b)                                     | NC::Methods&lt;dtype&gt;::logical_and(a, b)              |
| np.logical_or(a, b)	                                     | NC::Methods&lt;dtype&gt;::logical_or(a, b)               |
| np.isclose(a, b)                                         | NC::Methods&lt;dtype&gt;::isclose(a, b)                  |
| np.allclose(a, b)		                                     | NC::Methods&lt;dtype&gt;::allclose(a, b)                 |

### COMPARISONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.equal(a, b)                                           | NC::Methods&lt;dtype&gt;::equal(a, b)                    |
| np.not_equal(a, b)                                       | NC::Methods&lt;dtype&gt;::not_equal(a, b)                |
| np.nonzero(a)                                            | NC::Methods&lt;dtype&gt;::nonzero(a)                     |

### MINIMUM, MAXIMUM, SORTING

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.min(a)                                                | NC::Methods&lt;dtype&gt;::min(a)                         |
| np.max(a)                                                | NC::Methods&lt;dtype&gt;::max(a)                         |
| np.argmin(a)                                             | NC::Methods&lt;dtype&gt;::argmin(a)                      |
| np.argmax(a)	                                           | NC::Methods&lt;dtype&gt;::argmax(a)                      |
| np.sort(a, axis=0)                                       | NC::Methods&lt;dtype&gt;::sort(a, NC::Axis::ROW)         |
| np.argsort(a, axis=1)                                    | NC::Methods&lt;dtype&gt;::argsort(a, NC::Axis::COL)      |
| np.unique(a)                                             | NC::Methods&lt;dtype&gt;::unique(a)                      |
| np.setdiff1d(a, b)		                                   | NC::Methods&lt;dtype&gt;::setdiff1d(a, b)                |
| np.diff(a)		                                           | NC::Methods&lt;dtype&gt;::diff(a)                        |

### REDUCERS
Reducers accumulate values of `NdArray`s along specified axes. When no axis is specified, values are accumulated along all axes.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.sum(a)                                                | NC::Methods&lt;dtype&gt;::sum(a)                         |
| np.sum(a, axis=0)                                        | NC::Methods&lt;dtype&gt;::sum(a, NC::Axis::ROW)          |
| np.prod(a)                                               | NC::Methods&lt;dtype&gt;::prod(a)                        |
| np.prod(a, axis=0)	                                     | NC::Methods&lt;dtype&gt;::prod(a, NC::Axis::ROW)         |
| np.mean(a)                                               | NC::Methods&lt;dtype&gt;::mean(a)                        |
| np.mean(a, axis=0)                                       | NC::Methods&lt;dtype&gt;::mean(a, NC::Axis::ROW)         |
| np.count_nonzero(a)                                      | NC::Methods&lt;dtype&gt;::count_nonzero(a)               |
| np.count_nonzero(a, axis=0)		                           | NC::Methods&lt;dtype&gt;::count_nonzero(a,NC::Axis::ROW) |

### I/O
Print and file output methods.  All **NumpCpp** classes support a print() method and &lt;&lt; stream operators.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| print(a)                                                 | a.print()                                                |
|                                                          | std::cout &lt;&lt; a                                     |
| a.tofile(filename, sep=’\n’)                             | 	a.tofile(filename, sep=’\n’)                            |
| np.fromfile(filename, sep=’\n’)	                         | NC::Methods&lt;dtype&gt;::fromfile(filename, sep=’\n’)   |
| np.dump(a, filename)                                     | NC::Methods&lt;dtype&gt;::dump(a, filename)              |
| np.load(filename)                                        | NC::Methods&lt;dtype&gt;::load(filename)                 |

### MATHEMATICAL FUNCTIONS
**NumpCpp** universal functions are provided for a large set number of mathematical functions.
BASIC FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.absolute(a)                                           | NC::Methods&lt;dtype&gt;::absolute(a)                    |
| np.sign(a)                                               | NC::Methods&lt;dtype&gt;::sign(a)                        |
| np.remainder(a, b)                                       | NC::Methods&lt;dtype&gt;::remainder(a, b)                |
| np.clip(a, min, max)                                     | NC::Methods&lt;dtype&gt;::clip(a, min, max)              |
| np.interp(x, xp, fp)                                     | NC::Methods&lt;dtype&gt;::interp(x, xp, fp)              |

### EXPONENTIAL FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.exp(a)                                                | NC::Methods&lt;dtype&gt;::exp(a)                         |
| np.expm1(a)	                                             | NC::Methods&lt;dtype&gt;::expm1(a)                       |
| np.log(a)                                                | NC::Methods&lt;dtype&gt;::log(a)                         |
| np.log1p(a)                                              | NC::Methods&lt;dtype&gt;::log1p(a)                       |

### POWER FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.power(a, p)                                           | NC::Methods&lt;dtype&gt;::power(a)                       |
| np.sqrt(a)	                                             | NC::Methods&lt;dtype&gt;::sqrt(a)                        |
| np.square(a)                                             | NC::Methods&lt;dtype&gt;::square(a)                      |
| np.cbrt(a)                                               | NC::Methods&lt;dtype&gt;::cbrt(a)                        |

### TRIGONOMETRIC FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.sin(a)                                                | NC::Methods&lt;dtype&gt;::sin(a)                         |
| np.cos(a)	                                               | NC::Methods&lt;dtype&gt;::cos(a)                         |
| np.tan(a)                                                | NC::Methods&lt;dtype&gt;::tan(a)                         |

### HYPERBOLIC FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.sinh(a)                                               | NC::Methods&lt;dtype&gt;::sinh(a)                        |
| np.cosh(a)	                                             | NC::Methods&lt;dtype&gt;::cosh(a)                        |
| np.tanh(a)                                               | NC::Methods&lt;dtype&gt;::tanh(a)                        |

### CLASSIFICATION FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.isnan(a)                                              | NC::Methods&lt;dtype&gt;::isnan(a)                       |
| np.isinf(a)	                                             | NC::Methods&lt;dtype&gt;::isinf(a)                       |
| np.isfinite(a)                                           | NC::Methods&lt;dtype&gt;::isfinite(a)                    |

### LINEAR ALGEBRA

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| np.linalg.norm(a)                                        | NC::Methods&lt;dtype&gt;::norm(a)                        |
| np.dot(a, b)	                                           | NC::Methods&lt;dtype&gt;::dot(a, b)                      |
| np.linalg.det(a)                                         | NC::Linalg&lt;dtype&gt;::det(a)                          |
| np.linalg.inv(a)                                         | NC::Linalg&lt;dtype&gt;::inv(a)                          |
| np.linalg.lstsq(a, b)	                                   | NC::Linalg&lt;dtype&gt;::lstsq(a, b)                     |
| np.linalg.matrix_power(a, 3)                             | NC::Linalg&lt;dtype&gt;::matrix_power(a, 3)              |
| Np.linalg..multi_dot(a, b, c)                            | NC::Linalg&lt;dtype&gt;::multi_dot({a, b, c})            |
| np.linalg.svd(a)	                                       | NC::Linalg&lt;dtype&gt;::svd(a)                          |
