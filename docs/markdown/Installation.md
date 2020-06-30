# Installation

## 1. Clone the NumCpp repository from [GitHub](https://github.com/dpilger26/NumCpp)

**NUMCPP_REPO_PATH** = path to clone repository

```console
cd <NUMCPP_REPO_PATH>
git clone https://github.com/dpilger26/NumCpp.git
```

## 2. Build the install products using CMake

```console
cd NumCpp
mkdir build
cd build
cmake ..
```

## 3. Install the includes and CMake target files

On Linux run the following command with `sudo`.  On Windows, you may need to open a cmd prompt or PowerShell with admin privileges.

```console
cmake --build . --target install
```

## Alternative

NumCpp is a header only library so you can of course simply add the NumCpp include directory (wherever it resides) to your build system's include directories and build that way.
