/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// @section License
/// Copyright 2020 David Pilger
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
/// @section Description
/// Convience container for holding a uniform array of NdArrays
///
#pragma once

#include "NumCpp/Core/Internal/Filesystem.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

#include <deque>
#include <limits>
#include <string>

namespace nc
{
    //================================================================================
    ///						Convenience container for holding a uniform array of NdArrays
    template<typename dtype>
    class DataCube
    {
    public:
        //================================Typedefs==================================
        using iterator = typename std::deque<NdArray<dtype> >::iterator;
        using const_iterator = typename std::deque<NdArray<dtype> >::const_iterator;

        //============================================================================
        ///						Default Constructor
        ///
        DataCube() = default;

        //============================================================================
        ///						Constructor, preallocates to the input size
        ///
        /// @param      inSize
        ///
        explicit DataCube(uint32 inSize) :
            cube_(inSize)
        {};

        //============================================================================
        ///						Access method, with bounds checking
        ///
        /// @param      inIndex
        ///
        /// @return     NdArray
        ///
        NdArray<dtype>& at(uint32 inIndex)
        {
            return cube_.at(inIndex);
        }

        //============================================================================
        ///						Const access method, with bounds checking
        ///
        /// @param      inIndex
        ///
        /// @return     NdArray
        ///
        const NdArray<dtype>& at(uint32 inIndex) const
        {
            return cube_.at(inIndex);
        }

        //============================================================================
        ///						Returns a reference to the last element of the array
        ///
        /// @return     NdArray&
        ///
        NdArray<dtype>& back() noexcept 
        {
            return cube_.back();
        }

        //============================================================================
        ///						Returns an iterator to the beginning of the container
        ///
        /// @return     iterator
        ///
        iterator begin() noexcept
        {
            return cube_.begin();
        }

        //============================================================================
        ///						Returns a const_iterator to the beginning of the container
        ///
        /// @return     const_iterator
        ///
        const_iterator cbegin() const noexcept
        {
            return cube_.cbegin();
        }

        //============================================================================
        ///						Outputs the DataCube as a .bin file
        ///
        /// @param      inFilename
        ///
        void dump(const std::string& inFilename) const
        {
            filesystem::File f(inFilename);
            if (!f.hasExt())
            {
                f.withExt("bin");
            }

            std::ofstream ofile(f.fullName().c_str(), std::ios::binary);
            if (!ofile.good())
            {
                THROW_RUNTIME_ERROR("Could not open the input file:\n\t" + inFilename);
            }

            for (auto& ndarray : cube_)
            {
                ofile.write(reinterpret_cast<const char*>(ndarray.data()), ndarray.size() * sizeof(dtype));
            }

            ofile.close();
        }

        //============================================================================
        ///						Tests whether or not the container is empty
        ///
        /// @return     bool
        ///
        bool isempty() noexcept 
        {
            return cube_.empty();
        }

        //============================================================================
        ///						Returns an iterator to 1 past the end of the container
        ///
        /// @return     iterator
        ///
        iterator end() noexcept
        {
            return cube_.end();
        }

        //============================================================================
        ///						Returns a const_iterator to 1 past the end of the container
        ///
        /// @return     const_iterator
        ///
        const_iterator cend() const noexcept
        {
            return cube_.cend();
        }

        //============================================================================
        ///						returns a reference to the first element of the array
        ///
        /// @return     NdArray&
        ///
        NdArray<dtype>& front() noexcept 
        {
            return cube_.front();
        }

        //============================================================================
        ///						returns the number shape of the element arrays
        ///
        /// @return     Shape
        ///
        const Shape& shape() const noexcept 
        {
            return elementShape_;
        }

        //============================================================================
        ///						Returns the size of the container array
        ///
        /// @return     size
        ///
        uint32 size() const noexcept 
        {
            return static_cast<uint32>(cube_.size());
        }

        //============================================================================
        ///						Removes the last element in the container
        ///
        void pop_back() noexcept 
        {
            cube_.pop_back();
        }

        //============================================================================
        ///						Removes the first element in the container
        ///
        void pop_front() noexcept 
        {
            cube_.pop_front();
        }

        //============================================================================
        ///						Adds a new element at the end of the container
        ///
        /// @param      inArray
        ///
        void push_back(const NdArray<dtype>& inArray)
        {
            const Shape inputShape = inArray.shape();

            if (elementShape_.rows == 0 && elementShape_.cols == 0)
            {
                // initialize to the first input array size
                elementShape_.rows = inputShape.rows;
                elementShape_.cols = inputShape.cols;
            }

            if (inputShape != elementShape_)
            {
                THROW_INVALID_ARGUMENT_ERROR("element arrays must all be the same shape");
            }

            cube_.push_back(inArray);
        }

        //============================================================================
        ///						Adds a new element at the beginning of the container
        ///
        /// @param      inArray
        ///
        void push_front(const NdArray<dtype>& inArray)
        {
            const Shape inputShape = inArray.shape();

            if (elementShape_.rows == 0 && elementShape_.cols == 0)
            {
                // initialize to the first input array size
                elementShape_.rows = inputShape.rows;
                elementShape_.cols = inputShape.cols;
            }

            if (inputShape != elementShape_)
            {
                THROW_INVALID_ARGUMENT_ERROR("element arrays must all be the same shape.");
            }

            cube_.push_front(inArray);
        }

        //============================================================================
        ///						Access operator, no bounds checking
        ///
        /// @param      inIndex
        ///
        /// @return     NdArray
        ///
        NdArray<dtype>& operator[](uint32 inIndex) noexcept 
        {
            return cube_[inIndex];
        }

        //============================================================================
        ///						Const access operator, no bounds checking
        ///
        /// @param      inIndex
        ///
        /// @return     NdArray
        ///
        const NdArray<dtype>& operator[](uint32 inIndex) const noexcept
        {
            return cube_[inIndex];
        }

    private:
        //================================Attributes==================================
        std::deque<NdArray<dtype> >     cube_{};
        Shape                           elementShape_{ 0, 0 };
    };
}  // namespace nc
