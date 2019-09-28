# Installation

## 1. Clone the NumCpp repository from [GitHub](https://github.com/dpilger26/NumCpp)

**NUMCPP_PATH** = path to clone repository

```console
cd <NUMCPP_PATH>  
git clone https://github.com/dpilger26/NumCpp.git
```

## 2. Add the include files

### Linux

```console
cd <NUMCPP_PATH>/NumCpp/install
mkdir build
cd build
cmake ..
make install
```

### Windows

#### Visual Studio

```
1. Project -> Properties -> Configuration Properties -> C/C++ -> General
2. Additional Include Directories: <NUMCPP_PATH>/NumCpp/include
```

#### Cygwin/MinGW

```console
cd <NUMCPP_PATH>/NumCpp/install
mkdir build
cd build
cmake ..
make install
```
