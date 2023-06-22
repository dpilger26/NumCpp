# Building

## A NumCpp "Hello World" example

This example assumes you have followed the steps for installing NumCpp on your system.  You will also need to have CMake installed.

### 1. Source File

**main.cpp**

```cpp
#include "NumCpp.hpp"

#include <cstdlib>
#include <iostream>

int main()
{
    auto a = nc::random::randInt<int>({10, 10}, 0, 100);
    std::cout << a;

    return EXIT_SUCCESS;
}
```

### 2. CMakeLists.txt file

```cmake
cmake_minimum_required(VERSION 3.20)

project("HelloWorld" CXX)

add_executable(${PROJECT_NAME} main.cpp)

find_package(NumCpp 2.12.0 REQUIRED)
target_link_libraries(${PROJECT_NAME}
    NumCpp::NumCpp
)
```

Alternative using cmake fetch content

```cmake
cmake_minimum_required(VERSION 3.20)

project("HelloWorld" CXX)

add_executable(${PROJECT_NAME} main.cpp)

include(FetchContent)
FetchContent_Declare(NumCpp
        GIT_REPOSITORY https://github.com/dpilger26/NumCpp
        GIT_TAG Version_2.10.2)
FetchContent_MakeAvailable(NumCpp)

target_link_libraries(${PROJECT_NAME}
    NumCpp::NumCpp
)
```

### 3. Build

**SRC_DIRECTORY** = directory containing `main.cpp` and `CMakeLists.txt` files

```console
>> cd <SRC_DIRECTORY>
>> mkdir build
>> cd build
>> cmake ..
>> cmake --build . --config Release
```

### 4. Run

#### Linux

```console
>> ./HelloWorld
```

#### Windows

```console
>> HelloWorld.exe
```

### Alternative

**NumCpp** is a header only library so you can of course simply add the NumCpp include directory to your build system's include directories and build that way.  However, `find_package(NumCpp)` takes care of finding and linking in the **Boost** headers automatically, so if you add the NumCpp headers manually you will need to manually include the **Boost** headers as well.
