#!/bin/sh
# Builds the unit tests with the different compilers and C++ standards and runs them

update_compiler()
{
    COMPILER=$1
    VERSION=$2
    echo "Building with $COMPILER $VERSION..."
    sudo update-alternatives --set $COMPILER /usr/bin/$COMPILER-$VERSION
}

build_tests()
{
    COMPILER=$1
    CXX_STANDARD=$2
    rm -rf build
    mkdir build
    cd build
    cmake -DCMAKE_CXX_COMPILER=$COMPILER -DCMAKE_CXX_STANDARD=$CXX_STANDARD ..
    make
    cd ..
}

run_pytest()
{
    pytest ../pytest
}

test_standards()
{
    COMPILER=$1
    build_tests $COMPILER 14
    run_pytest
    build_tests $COMPILER 17
    run_pytest
    build_tests $COMPILER 20
    run_pytest
}

run_test()
{
    COMPILER=$1
    VERSION=$2
    update_compiler $COMPILER $VERSION
    test_standards $COMPILER
}

run_all_tests()
{
    run_test g++ 6
    run_test g++ 7
    run_test g++ 8
    run_test g++ 9
    run_test clang++ 9
    run_test clang++ 10
}

# Main script body
cd ../src
run_all_tests
