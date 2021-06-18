# Compiler Flags

* `NUMCPP_NO_USE_BOOST`: disables all **NumCpp** features that require the **Boost** libraries as a dependency.  When this compiler flag is defined **NumCpp** will have no external dependancies and is completely standalone
* `NUMCPP_USE_MULTITHREAD`: enables STL parallel execution policies throughout the library.  Using multi-threaded algorithms can have negative performace impact for "small" array operations and should usually only be used when dealing with large array operations.  Benchmarking should be performed with your system and build tools to determine which works best for your setup and application
* `NUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE`: includes the **PyBind11** Python interface helper functions
* `NUMCPP_INCLUDE_BOOST_PYTHON_INTERFACE`: includes the **Boost** Python interface helper functions
