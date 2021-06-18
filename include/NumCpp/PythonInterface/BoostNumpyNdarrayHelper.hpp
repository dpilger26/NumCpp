/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2021 David Pilger
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy of this
/// software and associated documentation files(the "Software"), to deal in the Software
/// without restriction, including without limitation the rights to use, copy, modify,
/// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
/// permit persons to whom the Software is furnished to do so, subject to the following
/// conditions :
///
/// The above copyright notice and this permission notice shall be included in all copies
/// or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
/// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
/// PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
/// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
/// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
/// DEALINGS IN THE SOFTWARE.
///
/// Description
/// A wrapper class for interacting with the boost numpy arrays
///
#pragma once

#if defined(NUMCPP_INCLUDE_BOOST_PYTHON_INTERFACE) && !defined(NUMCPP_NO_USE_BOOST)

#include "NumCpp/Core/Internal/StlAlgorithms.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/Utils/num2str.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "boost/python.hpp"
#include "boost/python/numpy.hpp"

namespace nc
{
    namespace boostPythonInterface
    {
        //================================================================================
        ///						Helper class for ndarray
        template<typename dtype>
        class BoostNdarrayHelper
        {
        public:
            //================================================================================
            ///						C or Fortran ordering from python
            enum class Order { F, C };

            //============================================================================
            ///						Constructor
            ///
            /// @param      inArray:  ndarray
            ///
            explicit BoostNdarrayHelper(const boost::python::numpy::ndarray& inArray) :
                theArray_(inArray.astype(boost::python::numpy::dtype::get_builtin<dtype>())),
                numDimensions_(static_cast<uint8>(inArray.get_nd())),
                shape_(numDimensions_),
                strides_(numDimensions_),
                order_(Order::C)

            {
                Py_intptr_t const * shapePtr = inArray.get_shape();
                for (uint8 i = 0; i < numDimensions_; ++i)
                {
                    strides_[i] = static_cast<uint32>(theArray_.strides(i));
                    shape_[i] = shapePtr[i];
                }

                if (numDimensions_ > 1 && inArray.strides(0) < inArray.strides(1))
                {
                    order_ = Order::F;
                }
            }

            //============================================================================
            ///						Constructor
            ///
            /// @param      inShape
            ///
            explicit BoostNdarrayHelper(boost::python::tuple inShape) :
                theArray_(boost::python::numpy::zeros(inShape, boost::python::numpy::dtype::get_builtin<dtype>())),
                numDimensions_(static_cast<uint8>(theArray_.get_nd())),
                shape_(numDimensions_),
                strides_(numDimensions_),
                order_(Order::C)
            {
                Py_intptr_t const * shapePtr = theArray_.get_shape();
                for (uint8 i = 0; i < numDimensions_; ++i)
                {
                    strides_[i] = static_cast<uint32>(theArray_.strides(i));
                    shape_[i] = shapePtr[i];
                }

                if (numDimensions_ > 1 && theArray_.strides(0) < theArray_.strides(1))
                {
                    order_ = Order::F;
                }

            }


            //============================================================================
            ///						Returns the internaly held ndarray
            ///
            /// @return     reference to the held ndarray
            ///
            const boost::python::numpy::ndarray& getArray() noexcept 
            {
                return theArray_;
            }

            //============================================================================
            ///						Returns the internaly held ndarray as a numpy matrix
            ///
            /// @return     matrix
            ///
            boost::python::numpy::matrix getArrayAsMatrix()
            {
                return boost::python::numpy::matrix(theArray_);
            }

            //============================================================================
            ///						Returns the number of dimensions of the array
            ///
            /// @return     num dimensions
            ///
            uint8 numDimensions() noexcept 
            {
                return numDimensions_;
            }

            //============================================================================
            ///						Returns the shape of the array
            ///
            /// @return     vector
            ///
            const std::vector<Py_intptr_t>& shape() noexcept 
            {
                return shape_;
            }

            //============================================================================
            ///						Returns the size of the array
            ///
            /// @return     size
            ///
            uint32 size()
            {
                uint32 theSize = 1;
                for (auto dimSize : shape_)
                {
                    theSize *= static_cast<uint32>(dimSize);
                }
                return theSize;
            }


            //============================================================================
            ///						Returns the strides of the array
            ///
            /// @return     vector
            ///
            const std::vector<uint32>& strides() 
            {
                return strides_;
            }

            //============================================================================
            ///						Returns the memory order of the array (C or Fortran)
            ///
            /// @return     Order
            ///
            Order order() 
            {
                return order_;
            }

            //============================================================================
            ///						Returns if the shapes of the two array helpers are equal
            ///
            /// @param      otherNdarrayHelper
            ///
            /// @return     boolean
            ///
            bool shapeEqual(BoostNdarrayHelper& otherNdarrayHelper)
            {
                if (shape_.size() != otherNdarrayHelper.shape_.size())
                {
                    return false;
                }

                return stl_algorithms::equal(shape_.begin(), shape_.end(), otherNdarrayHelper.shape_.begin());
            }

            //============================================================================
            ///						1D access operator
            ///
            /// @param      index
            ///
            /// @return     dtype
            ///
            dtype& operator()(uint32 index)
            {
                checkIndices1D(index);

                return *reinterpret_cast<dtype*>(theArray_.get_data() + strides_.front() * index);
            }

            //============================================================================
            ///						2D access operator
            ///
            /// @param      index1
            /// @param      index2
            ///
            /// @return     dtype
            ///
            dtype& operator()(uint32 index1, uint32 index2)
            {
                checkIndices2D(index1, index2);

                return *reinterpret_cast<dtype*>(theArray_.get_data() + strides_.front() * index1 + strides_[1] * index2);
            }

            //============================================================================
            ///						Prints a 1D array
            ///
            void printArray1D()
            {
                printf("array = \n");
                if (numDimensions_ != 1)
                {
                    std::cout << "printArray1D can only be used on a 1D array." << std::endl;
                    return;
                }

                for (int32 i = 0; i < shape_.front(); ++i)
                {
                    printf("\t%f\n", operator()(i));
                }
            }

            //============================================================================
            ///						Prints a 2D array
            ///
            void printArray2D()
            {
                printf("array = \n");
                if (numDimensions_ != 2)
                {
                    std::cout << "printArray2D can only be used on a 2D array." << std::endl;
                    return;
                }

                for (int32 index1 = 0; index1 < shape_.front(); ++index1)
                {
                    for (int32 index2 = 0; index2 < shape_.back(); ++index2)
                    {
                        printf("\t%f", operator()(index1, index2));
                    }
                    printf("\n");
                }
            }

        private:
            //====================================Attributes==============================
            boost::python::numpy::ndarray	theArray_;
            uint8							numDimensions_;
            std::vector<Py_intptr_t>		shape_;
            std::vector<uint32>				strides_;
            Order   						order_;

            //============================================================================
            ///						Generic check of input indices
            ///
            /// @param      indices
            ///
            void checkIndicesGeneric(boost::python::tuple indices)
            {
                if (boost::python::len(indices) != numDimensions_)
                {
                    std::string errStr = "Error: BoostNdarrayHelper::checkIndicesGeneric: Array has " + utils::num2str(numDimensions_);
                    errStr += " dimensions, you asked for " + utils::num2str(static_cast<int>(boost::python::len(indices))) + "!";
                    PyErr_SetString(PyExc_RuntimeError, errStr.c_str());
                }

                for (int i = 0; i < numDimensions_; ++i)
                {
                    int index = boost::python::extract<int>(indices[i]);
                    if (index > shape_[i])
                    {
                        std::string errStr = "Error: BoostNdarrayHelper::checkIndicesGeneric: Input index [" + utils::num2str(index);
                        errStr += "] is larger than the size of the array [" + utils::num2str(shape_[i]) + "].";
                        PyErr_SetString(PyExc_RuntimeError, errStr.c_str());
                    }
                }
            }

            //============================================================================
            ///						Checks 1D input indices
            ///
            /// @param      index
            ///
            void checkIndices1D(uint32 index)
            {
                boost::python::tuple indices = boost::python::make_tuple(index);
                checkIndicesGeneric(indices);
            }

            //============================================================================
            ///						Checks 2D input indices
            ///
            /// @param      index1
            /// @param		index2
            ///
            void checkIndices2D(uint32 index1, uint32 index2)
            {
                boost::python::tuple indices = boost::python::make_tuple(index1, index2);
                checkIndicesGeneric(indices);
            }
        };
    }  // namespace boostPythonInterface
} // namespace nc

#endif // #if defined(NUMCPP_INCLUDE_BOOST_PYTHON_INTERFACE) && !defined(NUMCPP_NO_USE_BOOST)
