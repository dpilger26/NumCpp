# Release Notes

## Version 2.4.0

* Added `replace` option into `random::choice`
* Added `nan_to_num` function

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