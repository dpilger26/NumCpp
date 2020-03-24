![NumCpp logo](https://github.com/dpilger26/NumCpp/raw/master/docs/logo/NumCppLogo.png)

<!-- [![GitHub watchers](https://img.shields.io/github/watchers/dpilger26/NumCpp.svg?style=social&label=Watch&maxAge=2592000)](https://GitHub.com/dpilger26/NumCpp/watchers/)
[![GitHub stars](https://img.shields.io/github/stars/dpilger26/NumCpp.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/dpilger26/NumCpp/stargazers/)
[![GitHub forks](https://img.shields.io/github/forks/dpilger26/NumCpp.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/dpilger26/NumCpp/network/)
[![HitCount](http://hits.dwyl.com/dpilger26/NumCpp.svg)](http://hits.dwyl.com/dpilger26/NumCpp.svg) -->

# NumCpp: A Templatized Header Only C++ Implementation of the [Python NumPy Library](http://www.numpy.org)

## Author: David Pilger dpilger26@gmail.com

## Version: 1.3

## License [![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

Copyright 2020 David Pilger

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files(the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Testing

**C++ Standards:** C++11, C++14, and C++17  
**Compilers:** VS 2017/2019, GCC 7.4.0/8.3/9.1, and Clang 6.0/8.0/9.0  
**Boost Versions:** 1.68 and 1.70  

## [Documentation](https://dpilger26.github.io/NumCpp)

## [GitHub](https://github.com/dpilger26/NumCpp)

## [Installation](https://dpilger26.github.io/NumCpp/doxygen/html/md__c_1__users_pilgeda__documents__git_hub__num_cpp_install__r_e_a_d_m_e.html)

## From NumPy To NumCpp – A Quick Start Guide

This quick start guide is meant as a very brief overview of some of the things
that can be done with **NumCpp**.  For a full breakdown of everything available
in the **NumCpp** library please visit the [Full Documentation](https://dpilger26.github.io/NumCpp).

### CONTAINERS

The main data structure in **NumCpp** is the `NdArray`.  It is inherently a 2D array class, with 1D arrays being implemented as 1xN arrays.  There is also a `DataCube` class that is provided as a convenience container for storing an array of 2D `NdArray`s, but it has limited usefulness past a simple container.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```a = np.array([[1, 2], [3, 4], [5, 6]])```             | ```nc::NdArray<int> a = { {1, 2}, {3, 4}, {5, 6} }```    |
| ```a.reshape([2, 3])```                                  | ```a.reshape(2, 3)```                                    |
| ```a.astype(np.double)```                                | ```a.astype<double>()```                                 |

### INITIALIZERS

Many initializer functions are provided that return `NdArray`s for common needs.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.linspace(1, 10, 5)```                              | ```nc::linspace<dtype>(1, 10, 5)```                      |
| ```np.arange(3, 7)```                                    | ```nc::arange<dtype>(3, 7)```                            |
| ```np.eye(4)```                                          | ```nc::eye<dtype>(4)```                                  |
| ```np.zeros([3, 4])```                                   | ```nc::zeros<dtype>(3, 4)```                             |
|                                                          | ```nc::NdArray<dtype>(3, 4) a = 0```                     |
| ```np.ones([3, 4])```                                    | ```nc::ones<dtype>(3, 4)```                              |
|                                                          | ```nc::NdArray<dtype>(3, 4) a = 1```                     |
| ```np.nans([3, 4])```                                    | ```nc::nans(3, 4)```                                     |
|                                                          | ```nc::NdArray<double>(3, 4) a = nc::constants::nan```   |
| ```np.empty([3, 4])```                                   | ```nc::empty<dtype>(3, 4)```                             |
|                                                          | ```nc::NdArray<dtype>(3, 4) a```                         |

### SLICING/BROADCASTING

**NumCpp** offers **NumPy** style slicing and broadcasting.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```a[2, 3]```                                            | ```a(2, 3)```                                            |
| ```a[2:5, 5:8]```                                        | ```a(nc::Slice(2, 5), nc::Slice(5, 8))```                |
|                                                          | ```a({2, 5}, {5, 8})```                                  |
| ```a[:, 7]```                                            | ```a(a.rSlice(), 7)```                                   |
| ```a[a > 5]```                                           | ```a[a > 50]```                                          |
| ```a[a > 5] = 0```                                       | ```a.putMask(a > 50, 666)```                             |

### RANDOM

The random module provides simple ways to create random arrays.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.random.seed(666)```                                | ```nc::random::seed(666)```                              |
| ```np.random.randn(3, 4)```                              | ```nc::random::randn<double>(nc::Shape(3,4))```          |
|                                                          | ```nc::random::randn<double>({3, 4})```                  |
| ```np.random.randint(0, 10, [3, 4])```                   | ```nc::random::randInt<int>(nc::Shape(3,4),0,10)```      |
|                                                          | ```nc::random::randInt<int>({3, 4},0,10)```              |
| ```np.random.rand(3, 4)```                               | ```nc::random::rand<double>(nc::Shape(3,4))```           |
|                                                          | ```nc::random::rand<double>({3, 4})```                   |
| ```np.random.choice(a, 3)```                             | ```nc::random::choice(a, 3)```                           |

### CONCATENATION

Many ways to concatenate `NdArray` are available.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.stack([a, b, c], axis=0)```                        | ```nc::stack({a, b, c}, nc::Axis::ROW)```                |
| ```np.vstack([a, b, c])```                               | ```nc::vstack({a, b, c})```                              |
| ```np.hstack([a, b, c])```                               | ```nc::hstack({a, b, c})```                              |
| ```np.append(a, b, axis=1)```                            | ```nc::append(a, b, nc::Axis::COL)```                    |

### DIAGONAL, TRIANGULAR, AND FLIP

The following return new `NdArray`s.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.diagonal(a)```                                     | ```nc::diagonal(a)```                                    |
| ```np.triu(a)```                                         | ```nc::triu(a)```                                        |
| ```np.tril(a)```                                         | ```nc::tril(a)```                                        |
| ```np.flip(a, axis=0)```                                 | ```nc::flip(a, nc::Axis::ROW)```                         |
| ```np.flipud(a)```                                       | ```nc::flipud(a)```                                      |
| ```np.fliplr(a)```                                       | ```nc::fliplr(a)```                                      |

### ITERATION

**NumCpp** follows the idioms of the C++ STL providing iterator pairs to iterate on arrays in different fashions.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```for value in a```                                     | ```for(auto it = a.begin(); it < a.end(); ++it)```       |
|                                                          | ```for(auto& value : a)```                               |

### LOGICAL

Logical FUNCTIONS in **NumCpp** behave the same as **NumPy**.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.where(a > 5, a, b)```                              | ```nc::where(a > 5, a, b)```                             |
| ```np.any(a)```                                          | ```nc::any(a)```                                         |
| ```np.all(a)```                                          | ```nc::all(a)```                                         |
| ```np.logical_and(a, b)```                               | ```nc::logical_and(a, b)```                              |
| ```np.logical_or(a, b)```                                | ```nc::logical_or(a, b)```                               |
| ```np.isclose(a, b)```                                   | ```nc::isclose(a, b)```                                  |
| ```np.allclose(a, b)```                                  | ```nc::allclose(a, b)```                                 |

### COMPARISONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.equal(a, b)```                                     | ```nc::equal(a, b)```                                    |
|                                                          | ```a == b```                                             |
| ```np.not_equal(a, b)```                                 | ```nc::not_equal(a, b)```                                |
|                                                          | ```a != b```                                             |
| ```rows, cols = np.nonzero(a)```                         | ```auto [rows, cols] = nc::nonzero(a)```                 |

### MINIMUM, MAXIMUM, SORTING

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.min(a)```                                          | ```nc::min(a)```                                         |
| ```np.max(a)```                                          | ```nc::max(a)```                                         |
| ```np.argmin(a)```                                       | ```nc::argmin(a)```                                      |
| ```np.argmax(a)```                                       | ```nc::argmax(a)```                                      |
| ```np.sort(a, axis=0)```                                 | ```nc::sort(a, nc::Axis::ROW)```                         |
| ```np.argsort(a, axis=1)```                              | ```nc::argsort(a, nc::Axis::COL)```                      |
| ```np.unique(a)```                                       | ```nc::unique(a)```                                      |
| ```np.setdiff1d(a, b)```                                 | ```nc::setdiff1d(a, b)```                                |
| ```np.diff(a)```                                         | ```nc::diff(a)```                                        |

### REDUCERS

Reducers accumulate values of `NdArray`s along specified axes. When no axis is specified, values are accumulated along all axes.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.sum(a)```                                          | ```nc::sum(a)```                                         |
| ```np.sum(a, axis=0)```                                  | ```nc::sum(a, nc::Axis::ROW)```                          |
| ```np.prod(a)```                                         | ```nc::prod(a)```                                        |
| ```np.prod(a, axis=0)```                                 | ```nc::prod(a, nc::Axis::ROW)```                         |
| ```np.mean(a)```                                         | ```nc::mean(a)```                                        |
| ```np.mean(a, axis=0)```                                 | ```nc::mean(a, nc::Axis::ROW)```                         |
| ```np.count_nonzero(a)```                                | ```nc::count_nonzero(a)```                               |
| ```np.count_nonzero(a, axis=0)```                        | ```nc::count_nonzero(a, nc::Axis::ROW)```                |

### I/O

Print and file output methods.  All **NumCpp** classes support a `print()` method and `<<` stream operators.

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```print(a)```                                           | ```a.print()```                                          |
|                                                          | ```std::cout << a```                                     |
| ```a.tofile(filename, sep=’\n’)```                       | ```a.tofile(filename, "\n")```                           |
| ```np.fromfile(filename, sep=’\n’)```                    | ```nc::fromfile<dtype>(filename, "\n")```                |
| ```np.dump(a, filename)```                               | ```nc::dump(a, filename)```                              |
| ```np.load(filename)```                                  | ```nc::load<dtype>(filename)```                          |

### MATHEMATICAL FUNCTIONS

**NumCpp** universal functions are provided for a large set number of mathematical functions.

#### BASIC FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.abs(a)```                                          | ```nc::abs(a)```                                         |
| ```np.sign(a)```                                         | ```nc::sign(a)```                                        |
| ```np.remainder(a, b)```                                 | ```nc::remainder(a, b)```                                |
| ```np.clip(a, 3, 8)```                                   | ```nc::clip(a, 3, 8)```                                  |
| ```np.interp(x, xp, fp)```                               | ```nc::interp(x, xp, fp)```                              |

#### EXPONENTIAL FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.exp(a)```                                          | ```nc::exp(a)```                                         |
| ```np.expm1(a)```                                        | ```nc::expm1(a)```                                       |
| ```np.log(a)```                                          | ```nc::log(a)```                                         |
| ```np.log1p(a)```                                        | ```nc::log1p(a)```                                       |

#### POWER FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.power(a, 4)```                                     | ```nc::power(a, 4)```                                    |
| ```np.sqrt(a)```                                         | ```nc::sqrt(a)```                                        |
| ```np.square(a)```                                       | ```nc::square(a)```                                      |
| ```np.cbrt(a)```                                         | ```nc::cbrt(a)```                                        |

#### TRIGONOMETRIC FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.sin(a)```                                          | ```nc::sin(a)```                                         |
| ```np.cos(a)```                                          | ```nc::cos(a)```                                         |
| ```np.tan(a)```                                          | ```nc::tan(a)```                                         |

#### HYPERBOLIC FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.sinh(a)```                                         | ```nc::sinh(a)```                                        |
| ```np.cosh(a)```                                         | ```nc::cosh(a)```                                        |
| ```np.tanh(a)```                                         | ```nc::tanh(a)```                                        |

#### CLASSIFICATION FUNCTIONS

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.isnan(a)```                                        | ```nc::isnan(a)```                                       |
| ```np.isinf(a)```                                        | ```nc::isinf(a)```                                       |

#### LINEAR ALGEBRA

| **NumPy**                                                | **NumCpp**                                               |
|:--------------------------------------------------------:|:--------------------------------------------------------:|
| ```np.linalg.norm(a)```                                  | ```nc::norm(a)```                                        |
| ```np.dot(a, b)```                                       | ```nc::dot(a, b)```                                      |
| ```np.linalg.det(a)```                                   | ```nc::linalg::det(a)```                                 |
| ```np.linalg.inv(a)```                                   | ```nc::linalg::inv(a)```                                 |
| ```np.linalg.lstsq(a, b)```                              | ```nc::linalg::lstsq(a, b)```                            |
| ```np.linalg.matrix_power(a, 3)```                       | ```nc::linalg::matrix_power(a, 3)```                     |
| ```Np.linalg.multi_dot(a, b, c)```                       | ```nc::linalg::multi_dot({a, b, c})```                   |
| ```np.linalg.svd(a)```                                   | ```nc::linalg::svd(a)```                                 |
