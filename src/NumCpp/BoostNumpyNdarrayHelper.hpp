/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
/// @version 1.0
///
/// @section LICENSE
/// Copyright 2018 David Pilger
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
/// @section DESCRIPTION
/// A module for interacting with the boost numpy arrays
///
#pragma once

#include<NumCpp/NdArray.hpp>
#include<NumCpp/Types.hpp>

#include<cmath>
#include<vector>
#include<iostream>
#include<string>
#include<stdexcept>
#include<utility>

#include"boost/python.hpp"
#include"boost/python/numpy.hpp"

namespace NumCpp
{
    //================================================================================
    ///						C or Fortran ordering from python
    struct Order { enum Type { F, C }; };

    //================================================================================
    ///						Helper class for ndarray
    class BoostNdarrayHelper
    {
    private:
        //====================================Attributes==============================
        boost::python::numpy::ndarray	theArray_;
        uint8							numDimensions_;
        std::vector<Py_intptr_t>		shape_;
        std::vector<uint32>				strides_;
        Order::Type						order_;

        //============================================================================
        ///						Generic check of input indices
        ///		
        /// @param      tuple of indices
        ///
        /// @return     None
        ///
        void checkIndicesGeneric(boost::python::tuple indices)
        {
            if (boost::python::len(indices) != numDimensions_)
            {
                std::string errorString = "Error: Array has " + std::to_string(numDimensions_) + " dimensions, you asked for " + std::to_string(static_cast<int>(boost::python::len(indices))) + "!";
                PyErr_SetString(PyExc_RuntimeError, errorString.c_str());
            }

            for (int i = 0; i < numDimensions_; ++i)
            {
                int index = boost::python::extract<int>(indices[i]);
                if (index > shape_[i])
                {
                    std::string errorString = "Error: Input index [" + std::to_string(index) + "] is larger than the size of the array [" + std::to_string(shape_[i]) + "].";
                    PyErr_SetString(PyExc_RuntimeError, errorString.c_str());
                }
            }
        }

        //============================================================================
        ///						Checks 1D input indices
        ///		
        /// @param      index
        ///
        /// @return     None
        ///
        void checkIndices1D(uint32 index1)
        {
            boost::python::tuple indices = boost::python::make_tuple(index1);
            checkIndicesGeneric(indices);
        }

        //============================================================================
        ///						Checks 2D input indices
        ///		
        /// @param      index1
        /// @param		index2
        ///
        /// @return     None
        ///
        void checkIndices2D(uint32 index1, uint32 index2)
        {
            boost::python::tuple indices = boost::python::make_tuple(index1, index2);
            checkIndicesGeneric(indices);
        }

        //============================================================================
        ///						Checks 3D input indices
        ///		
        /// @param      index1
        /// @param      index2
        /// @param      index3
        ///
        /// @return     None
        ///
        void checkIndices3D(uint32 index1, uint32 index2, uint32 index3)
        {
            boost::python::tuple indices = boost::python::make_tuple(index1, index2, index3);
            checkIndicesGeneric(indices);
        }

    public:
        //============================================================================
        ///						Constructor
        ///		
        /// @param      pointer to an ndarray
        ///
        /// @return     None
        ///
        BoostNdarrayHelper(boost::python::numpy::ndarray* inArray) :
            theArray_(inArray->astype(boost::python::numpy::dtype::get_builtin<double>())),
            numDimensions_(static_cast<uint8>(inArray->get_nd())),
            order_(Order::C)

        {
            Py_intptr_t const * shapePtr = inArray->get_shape();
            for (uint8 i = 0; i < numDimensions_; ++i)
            {
                strides_.push_back(static_cast<uint32>(theArray_.strides(i)));
                shape_.push_back(shapePtr[i]);
            }

            if (numDimensions_ > 1 && inArray->strides(0) < inArray->strides(1))
            {
                order_ = Order::F;
            }
        }

        //============================================================================
        ///						Constructor
        ///		
        /// @param      pointer to an ndarray
        ///
        /// @return     None
        ///
        BoostNdarrayHelper(boost::python::tuple inShape) :
            theArray_(boost::python::numpy::zeros(inShape, boost::python::numpy::dtype::get_builtin<double>()))
        {
            BoostNdarrayHelper newArrayHelper(&theArray_);
            numDimensions_ = newArrayHelper.numDimensions();
            shape_ = newArrayHelper.shape();
            strides_ = newArrayHelper.strides();
            order_ = newArrayHelper.order();
        }


        //============================================================================
        ///						Returns the internaly held ndarray
        ///		
        /// @param      None
        ///
        /// @return     pointer to an ndarray
        ///
        const boost::python::numpy::ndarray* getArray()
        {
            return &theArray_;
        }

        //============================================================================
        ///						Returns the internaly held ndarray as a numpy matrix
        ///		
        /// @param      None
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
        /// @param      None
        ///
        /// @return     num dimensions
        ///
        uint8 numDimensions()
        {
            return numDimensions_;
        }

        //============================================================================
        ///						Returns the shape of the array
        ///		
        /// @param      None
        ///
        /// @return     vector
        ///
        const std::vector<Py_intptr_t>& shape()
        {
            return shape_;
        }

        //============================================================================
        ///						Returns the size of the array
        ///		
        /// @param      None
        ///
        /// @return     size
        ///
        uint32 size()
        {
            uint32 theSize = 1;
            for (uint8 dim = 0; dim < numDimensions_; ++dim)
            {
                theSize *= static_cast<uint32>(shape_[dim]);
            }
            return theSize;
        }


        //============================================================================
        ///						Returns the strides of the array
        ///		
        /// @param      None
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
        /// @param      None
        ///
        /// @return     Order
        ///
        Order::Type order()
        {
            return order_;
        }

        //============================================================================
        ///						Returns if the shapes of the two array helpers are equal
        ///		
        /// @param      None
        ///
        /// @return     boolean
        ///
        bool shapeEqual(BoostNdarrayHelper& otherNdarrayHelper)
        {
            if (shape_.size() != otherNdarrayHelper.shape().size())
            {
                return false;
            }

            for (uint32 i = 0; i < shape_.size(); ++i)
            {
                if (shape_[i] != otherNdarrayHelper.shape()[i])
                {
                    return false;
                }
            }
            return true;
        }

        //============================================================================
        ///						1D access operator
        ///		
        /// @param      None
        ///
        /// @return     double
        ///
        double& operator()(uint32 index)
        {
            checkIndices1D(index);

            return *reinterpret_cast<double*>(theArray_.get_data() + strides_[0] * index);
        }

        //============================================================================
        ///						2D access operator
        ///		
        /// @param      None
        ///
        /// @return     double
        ///
        double& operator()(uint32 index1, uint32 index2)
        {
            checkIndices2D(index1, index2);
            return *reinterpret_cast<double*>(theArray_.get_data() + strides_[0] * index1 + strides_[1] * index2);
        }

        //============================================================================
        ///						3D access operator
        ///		
        /// @param      None
        ///
        /// @return     double
        ///
        double& operator()(uint32 index1, uint32 index2, uint32 index3)
        {
            checkIndices3D(index1, index2, index3);

            return *reinterpret_cast<double*>(theArray_.get_data() + strides_[0] * index1 + strides_[1] * index2 + strides_[2] * index3);
        }

        //============================================================================
        ///						Prints a 1D array
        ///		
        /// @param      None
        ///
        /// @return     None
        ///
        void printArray1D()
        {
            printf("array = \n");
            if (numDimensions_ > 1)
            {
                std::cout << "printArray1D can only be used on a 1D array." << std::endl;
                return;
            }

            for (int32 i = 0; i < shape_[0]; ++i)
            {
                printf("\t%f\n", this->operator()(i));
            }
        }

        //============================================================================
        ///						Prints a 2D array
        ///		
        /// @param      None
        ///
        /// @return     None
        ///
        void printArray2D()
        {
            printf("array = \n");
            if (numDimensions_ > 2)
            {
                std::cout << "printArray2D can only be used on a 2D array." << std::endl;
                return;
            }

            for (int32 index1 = 0; index1 < shape_[0]; ++index1)
            {
                for (int32 index2 = 0; index2 < shape_[1]; ++index2)
                {
                    printf("\t%f", this->operator()(index1, index2));
                }
                printf("\n");
            }
        }

        //============================================================================
        ///						Prints a 3D array
        ///		
        /// @param      None
        ///
        /// @return     None
        ///
        void printArray3D()
        {
            printf("array = \n");
            if (numDimensions_ > 2)
            {
                std::cout << "printArray3D can only be used on a 3D array." << std::endl;
                return;
            }

            for (int32 index1 = 0; index1 < shape_[0]; ++index1)
            {
                for (int32 index2 = 0; index2 < shape_[1]; ++index2)
                {
                    for (int32 index3 = 0; index3 < shape_[2]; ++index3)
                    {
                        printf("\t%f", this->operator()(index1, index2, index3));
                    }
                }
                printf("\n");
            }
            printf("\n");
        }
    }; // class ndarrayHelper

    //============================================================================
    ///						Converts from a boost ndarray to a NumCpp NdArray<T>
    ///		
    /// @param      ndarray
    ///
    /// @return     NdArray<T>
    ///
    template<typename dtype>
    NdArray<dtype> boostToNumC(boost::python::numpy::ndarray& inArray)
    {
        BoostNdarrayHelper helper(&inArray);
        if (helper.numDimensions() > 2)
        {
            throw std::runtime_error("ERROR: Can only convert 1 and 2 dimensional arrays.");
        }

        NumCpp::Shape arrayShape;
        if (helper.numDimensions() == 1)
        {
            arrayShape.rows = 1;
            arrayShape.cols = static_cast<uint32>(helper.shape()[0]);

            NdArray<dtype> returnArray(arrayShape);
            for (uint32 i = 0; i < arrayShape.size(); ++i)
            {
                returnArray[i] = static_cast<dtype>(helper(i));
            }

            return std::move(returnArray);
        }
        else
        {
            arrayShape.rows = static_cast<uint32>(helper.shape()[0]);
            arrayShape.cols = static_cast<uint32>(helper.shape()[1]);

            NdArray<dtype> returnArray(arrayShape);
            uint32 i = 0;
            for (uint32 row = 0; row < arrayShape.rows; ++row)
            {
                for (uint32 col = 0; col < arrayShape.cols; ++col)
                {
                    returnArray[i++] = static_cast<dtype>(helper(row, col));
                }
            }

            return std::move(returnArray);
        }
    }

    //============================================================================
    ///						Converts from a NumCpp NdArray<T> to a boost ndarray
    ///		
    /// @param      NdArray<T>
    ///
    /// @return     ndarray
    ///
    template<typename dtype>
    boost::python::numpy::ndarray numCToBoost(const NdArray<dtype>& inArray)
    {
        Shape inShape = inArray.shape();
        boost::python::tuple shape = boost::python::make_tuple(inShape.rows, inShape.cols);
        BoostNdarrayHelper newNdArrayHelper(shape);

        for (uint32 row = 0; row < inShape.rows; ++row)
        {
            for (uint32 col = 0; col < inShape.cols; ++col)
            {
                newNdArrayHelper(row, col) = static_cast<double>(inArray(row, col));
            }
        }
        return *(newNdArrayHelper.getArray());
    }
}
