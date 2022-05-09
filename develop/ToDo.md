# TODO

* update run all script for new linux env and windows env
* test building all with NumCpp with -DNO_USE_BOOST option in cmake configure

* integrate improved threshold and cluster code

* new non-global random interface
* address open issues
  * 132: fix `inv()`
    * <https://www.geeksforgeeks.org/adjoint-inverse-matrix/>
  * 137: add `fft()`
    * <https://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B>
    * <https://www.oreilly.com/library/view/c-cookbook/0596007612/ch11s18.html>
  * 139: strided functions
  * 140: Template Bool -> BoolType
  * 143: eigen values and vectors
* add doxygen docstring descriptions of all inputs and outputs of functions/methods
* regenerate docs
* redo all CI/CD pipelines for new compilers supported
