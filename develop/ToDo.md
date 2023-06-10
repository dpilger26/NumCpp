# TODO

## Version 2.11.0

* `stack`/`vstack`/`hstack`/`row_stack`/`col_stack` should take in a `std::vector`, ISSUE #196
* incorporate new utilities under `/develop`
  * reference_frames
  * transforms
    * combine RA/Dec/Coordinate into a single file called Celestial.hpp
    * unit tests for new constructors for Coordinate
    * tests for new Vec3 overload
    * tests for new normalize function
    * tests for new NdArray rows(), and columns() methods
  * Logger
  * BinaryLogger
* mixing of floats and ints in NdArray operators?

* run clang-tidy
* run cppcheck
* rebuild documentation
* merge to main / tag / create release
