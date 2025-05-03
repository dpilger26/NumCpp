# Release Notes

## Version 2.14.1

* made behavior of `linalg::lstsq` match NumPy when inputting a multidimensional `b` array for **Issue #229**

## Version 2.14.0

* added `argpartition()` for **Issue #225** <https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html>

## Version 2.13.0

* added `putAlongAxis()` for **Issue #217** <https://numpy.org/doc/stable/reference/generated/numpy.put_along_axis.html>
* fixed random seeding for **Issue #218**
* fixed error in `fromfile()` that limited filesize to 4GB for **Issue #219**

## Version 2.12.1

* updated TRUE/FALSE enum fields to YES/NO to deconflict with other libraries terrible macro practice of #defining TRUE/FALSE

## Version 2.12.0

* added `searchsorted()` for **Issue #202** <https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html>
* added `meanFilter` to `Filters`
* added `complementaryMeanFilter` to `Filters`

## Version 2.11.0

* fixed Issue #191
* fixed Issue #194
* fixed Issue #196
* added `rows` and `columns` methods to `NdArray`
* added `wrap` and `wrap2Pi` functions
* added `normalize` function for `NdArray`s
* added `Logger` and `BinaryLogger` classes for logging data
* added `coordinates::Cartesian` class
* added `coordinates::reference_frames` and `coordinates::transforms` namespaces for converting between various coordinate systems
* various improvements and bug fixes

## Version 2.10.1

* fixed an error in installed cmake target when using `NUMCPP_NO_USE_BOOST`

## Version 2.10.0

* added broadcasting for all `NdArray` operators for **Issue #147** and **Issue #174**
* added broadcasting for `minimum` and `maximum` functions for **Issue #74**
* added broadcasting for:
  * `fmin`
  * `fmax`
  * `fmod`
  * `hypot`
  * `logical_and`
  * `logical_or`
  * `logical_xor`
  * `remainder`
* added `insert` function for **Issue #170** <https://numpy.org/doc/stable/reference/generated/numpy.insert.html>
* fixed **Issue #177**: slice and put with various integer index types
* additional `NdArray` access operator overloads and `at` overloads
* additional `put` overloads to `NdArray`
* added `dimSize` method to `NdArray`
* added `timeit` function
* added overload of `hypot` for 3 `NdArray`s
* various performance improvements and bug fixes

## Version 2.9.0

* dropped support of C++14 standard, now requires C++17 or greater
* added a new `DateTime` class for working with date times
* added `split` for **Issue #168** <https://numpy.org/doc/stable/reference/generated/numpy.split.html>
* added `hsplit` for **Issue #168** <https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html>
* added `vsplit` for **Issue #168** <https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html>
* added `digitize` <https://numpy.org/doc/stable/reference/generated/numpy.digitize.html>
* added `fromfunction` <https://numpy.org/doc/stable/reference/generated/numpy.fromfunction.html>
* added `fromstring` <https://numpy.org/doc/stable/reference/generated/numpy.fromstring.html>
* added `logaddexp` <https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html>
* added `logaddexp2` <https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html>
* added `packbits` <https://numpy.org/doc/stable/reference/generated/numpy.packbits.html>
* added `unpackbits` <https://numpy.org/doc/stable/reference/generated/numpy.unpackbits.html>
* added `vander` <https://numpy.org/doc/stable/reference/generated/numpy.vander.html>
* added `rollRotation`, `pitchRotation`, and `yawRotation` methods to `Quaternion`
* added `NOMINMAX` compiler define to CMake target to deconflict with Windows builds

## Version 2.8.0

* fixed error in `inv` when a zero was on the diagnol for **Issue #132**
* fixed **Issue #140**
* added `bit_count`, Computes the number of 1-bits in the absolute value of the input
* added `swapRows` and `swapCols` methods to `NdArray`, and also free functions
* added `pinv`, <https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html>
* added `take` for **Issue #149**, <https://numpy.org/doc/stable/reference/generated/numpy.take.html>
* new non-global `random` interface
* added ellipse metrics to `imageProcessing::Centroids`
* documentation updates
* various minor performance improvements and bug fixes

## Version 2.7.0

* added `bartlett`, <https://numpy.org/doc/stable/reference/generated/numpy.bartlett.html>
* added `blackman`, <https://numpy.org/doc/stable/reference/generated/numpy.blackman.html>
* added `corrcoef`, <https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html>
* added `cov`, <https://numpy.org/doc/stable/reference/generated/numpy.cov.html>
* added `cov_inv`, the inverse covariance matrix, aka the concentration matrix
* added `extract`, <https://numpy.org/doc/stable/reference/generated/numpy.extract.html>
* added `geomspace`, <https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html>
* added `hamming`, <https://numpy.org/doc/stable/reference/generated/numpy.hamming.html>
* added `hanning`, <https://numpy.org/doc/stable/reference/generated/numpy.hanning.html>
* added `inner`, <https://numpy.org/doc/stable/reference/generated/numpy.inner.html>
* added `isneginf`, <https://numpy.org/doc/stable/reference/generated/numpy.isneginf.html>
* added `isposinf`, <https://numpy.org/doc/stable/reference/generated/numpy.isposinf.html>
* added `kaiser`, <https://numpy.org/doc/stable/reference/generated/numpy.kaiser.html>
* added `logb`, logarithm of with an arbitrary base b
* added `logspace`, <https://numpy.org/doc/stable/reference/generated/numpy.logspace.html>
* added `nth_root`, the nth root of a value
* added `place`, <https://numpy.org/doc/stable/reference/generated/numpy.place.html>
* added `select` function, <https://numpy.org/doc/stable/reference/generated/numpy.select.html>
* `fmod` and the modulus `%` operator now work with float dtypes
* added Hamming EDAC (Error Dectection and Correction) `encode` and `decode` functions, <https://en.wikipedia.org/wiki/Hamming_code>
* various minor performance improvements and bug fixes

## Version 2.6.2

* `tofile` and `fromfile` will now work for generic struct dtypes

## Version 2.6.1

* Added more delimiter support to `fromfile` method

## Version 2.6.0

* Added `linalg::solve`

## Version 2.5.1

* Made behavior of `interp` function consistent with `NumPy` when passing in non-sorted data

## Version 2.5.0

* Added additional `NdArray` slice overloads
* Removed `NO_MULTITHREAD` compiler flag and replaced with `NUMCPP_USE_MULTITHREAD` so that single threaded is now the default
* renamed `NO_USE_BOOST` compiler flag to `NUMCPP_NO_USE_BOOST`
* renamed `INCLUDE_BOOST_PYTHON_INTERFACE` compiler flat to `NUMCPP_INCLUDE_BOOST_PYTHON_INTERFACE`
* renamed `INCLUDE_PYBIND_PYTHON_INTERFACE` compiler flag to `NUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE`

## Version 2.4.2

* Fixed a type error with `percentile` and `nanpercentile`
* Updated doxygen API css

## Version 2.4.1

* Fixed a build error for multiply defined symbols of `isLittleEndian`

## Version 2.4.0

* Compile with `NO_USE_BOOST` definition to remove the Boost libraries as a dependency, with reduced functionality:
  * `gcd` with a pair of values (still available using a C++17 compliant compiler)
  * `gcd` array
  * `lcm` with a pair of values (still available using a C++17 compliant compiler)
  * `lcm` array
  * `polynomial::chebyshev_t`
  * `polynomial::chebyshev_u`
  * `polynomial::hermite` (still available using a C++17 compliant compiler)
  * `polynomial::laguerre` (still available using a C++17 compliant compiler)
  * `polynomial::legendre_p` (still available using a C++17 compliant compiler)
  * `polynomial::legendre_q`
  * `polynomial::spherical_harmonic`
  * `random::beta`
  * `random::laplace`
  * `random::nonCentralChiSquared`
  * `random::triangle`
  * `random::uniformOnSphere`
  * `special::airy_ai`
  * `special::airy_ai_prime`
  * `special::airy_bi`
  * `special::airy_bi_prime`
  * `special::bernoulli`
  * `special::bessel_in` (still available using a C++17 compliant compiler)
  * `special::bessel_in_prime`
  * `special::bessel_jn` (still available using a C++17 compliant compiler)
  * `special::bessel_jn_prime`
  * `special::bessel_kn` (still available using a C++17 compliant compiler)
  * `special::bessel_kn_prime`
  * `special::bessel_yn` (still available using a C++17 compliant compiler)
  * `special::bessel_yn_prime`
  * `special::beta` (still available using a C++17 compliant compiler)
  * `special::cyclic_hankel_1`
  * `special::cyclic_hankel_2`
  * `special::digamma`
  * `special::erf`
  * `special::erf_inv`
  * `special::erfc`
  * `special::erfc_inv`
  * `special::gamma`
  * `special::gamma1pm1`
  * `special::log_gamma`
  * `special::polygamma`
  * `special::prime`
  * `special::riemann_zeta` (still available using a C++17 compliant compiler)
  * `special::spherical_bessel_jn` (still available using a C++17 compliant compiler)
  * `special::spherical_bessel_yn` (still available using a C++17 compliant compiler)
  * `special::spherical_hankel_1`
  * `special::spherical_hankel_2`
  * `special::trigamma`
* Added `replace` option into `random::choice`
* Added `nan_to_num` function
* Added complete and incomplete elliptical integrals of the first, second, and third kind to `special` namespace (requires either Boost or C++17 compliant compiler)
* Added exponential integral to `special` namespace (requires either Boost or C++17 compliant compiler)
* Added `NO_MULTITHREAD` compile definition to turn off algorithm multithreading from compliant compilers

## Version 2.3.1

* Added option for user defined bin edges in `histogram()` function

## Version 2.3.0

* Added slicing to `DataCube` class  

## Version 2.2.0

* Added additional `where()` overloads to match NumPy functionality  

## Version 2.1.0

* Improved installation and usage with CMake find_package support
* Various minor improvements

## Version 2.0.0

* Dropped support of C++11, now requires a C++14 or higher compiler
* Added support for `std::complex<T>`, closing **Issue #58**
* Added more `NdArray` constructors for STL containers including `std::vector<std::vector<T>>`, closing **Issue #59**
* Added `polyfit` routine inline with Numpy `polyfit`, closing **Issue #61**
* Added ability to use `NdArray` as container for generic structs
* Non-linear least squares fitting using Gauss-Newton
* Root finding routines
* Numerical integration routines
* `lu_decomposition` and `pivotLU_decomposition` added to `Linalg` namespace
* New STL iterators added to `NdArray`
  * `iterator`
  * `const_iterator`
  * `reverse_iterator`
  * `const_reverse_iterator`
  * `column_iterator`
  * `const_column_iterator`
  * `reverse_column_iterator`
  * `const_reverse_column_iterator`
* Added `rodriguesRotation` and `wahbasProblem` to `Rotations` namespace
* Various efficiency and/or bug fixes
