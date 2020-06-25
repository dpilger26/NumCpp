#!/bin/sh
# Builds the examples with the different compilers and C++ standards

update_compiler()
{
    COMPILER=$1
    VERSION=$2
    echo "Building with $COMPILER $VERSION..."
    sudo update-alternatives --set $COMPILER /usr/bin/$COMPILER-$VERSION
}

build_tests()
{
    ROOT_DIR=$1
    COMPILER=$2
    CXX_STANDARD=$3
    CURRENT_DIR=$(pwd)
    cd $ROOT_DIR
    rm -rf build
    mkdir build
    cd build
    cmake -DCMAKE_CXX_COMPILER=$COMPILER -DCMAKE_CXX_STANDARD=$CXX_STANDARD ..
    make
    cd $CURRENT_DIR
}

test_standards()
{
    EXAMPLE_DIR=$1
    COMPILER=$2
    build_tests $EXAMPLE_DIR $COMPILER 14
    build_tests $EXAMPLE_DIR $COMPILER 17
    build_tests $EXAMPLE_DIR $COMPILER 20
}

run_test()
{
    EXAMPLE_DIR=$1
    COMPILER=$2
    VERSION=$3
    update_compiler $COMPILER $VERSION
    test_standards $EXAMPLE_DIR $COMPILER
}

run_all_compilers()
{
    EXAMPLE_DIR=$1
    run_test $EXAMPLE_DIR g++ 6
    run_test $EXAMPLE_DIR g++ 7
    run_test $EXAMPLE_DIR g++ 8
    run_test $EXAMPLE_DIR g++ 9
    run_test $EXAMPLE_DIR clang++ 9
    run_test $EXAMPLE_DIR clang++ 10
}

run_all_examples()
{
    EXAMPLE_ROOT_DIR=$1
    run_all_compilers "$EXAMPLE_ROOT_DIR/GaussNewtonNlls"
    run_all_compilers "$EXAMPLE_ROOT_DIR/InterfaceWithEigen"
    run_all_compilers "$EXAMPLE_ROOT_DIR/InterfaceWithOpenCV"
    run_all_compilers "$EXAMPLE_ROOT_DIR/ReadMe"
}

# Main script body
EXAMPLE_PATH=$( realpath "../../examples" )
run_all_examples $EXAMPLE_PATH
