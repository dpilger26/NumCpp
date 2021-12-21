# Release Notes

## Version 2.6.3

* added `select` function
* `fmod` and the modulus `%` operator now work with float dtypes
* minor performance improvements

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
* Added support for `std::complex<T>`, closing __Issue #58__
* Added more `NdArray` constructors for STL containers including `std::vector<std::vector<T>>`, closing __Issue #59__
* Added `polyfit` routine inline with Numpy `polyfit`, closing __Issue #61__
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
