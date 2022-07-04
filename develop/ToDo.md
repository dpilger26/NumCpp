# TODO

* modify cmake to build NumCppPy into pytest directory
  * modify pytests to not need to add path

* split ndarray access operators into private non-error checking and public error checking
  * all internal "safe" operations can use the non-error checking versions

* update run all script for new linux env and windows env
* test building all with NumCpp with -DNO_USE_BOOST option in cmake configure

* integrate improved threshold and cluster code

* address open issues
  * 137: add `fft()`
    * <https://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B>
    * <https://www.oreilly.com/library/view/c-cookbook/0596007612/ch11s18.html>
  * 139: strided functions
  * 143: eigen values and vectors
* add doxygen docstring descriptions of all inputs and outputs of functions/methods
* regenerate docs
* redo all CI/CD pipelines for new compilers supported
