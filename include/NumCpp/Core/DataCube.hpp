/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2025 David Pilger
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
/// Convience container for holding a uniform array of NdArrays
///
#pragma once

#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "NumCpp/Core/Internal/Error.hpp"
#include "NumCpp/Core/Shape.hpp"
#include "NumCpp/Core/Slice.hpp"
#include "NumCpp/Core/Types.hpp"
#include "NumCpp/NdArray.hpp"

namespace nc
{
    //================================================================================
    /// Convenience container for holding a uniform array of NdArrays
    template<typename dtype>
    class DataCube
    {
    public:
        //================================Typedefs==================================
        using iterator       = typename std::deque<NdArray<dtype>>::iterator;
        using const_iterator = typename std::deque<NdArray<dtype>>::const_iterator;

        //============================================================================
        /// Default Constructor
        ///
        DataCube() = default;

        //============================================================================
        /// Constructor, preallocates to the input size
        ///
        /// @param inSize
        ///
        explicit DataCube(uint32 inSize)
        {
            cube_.reserve(inSize);
        }

        //============================================================================
        /// Access method, with bounds checking. Returns the 2d z "slice" element of the cube.
        ///
        /// @param inIndex
        ///
        /// @return NdArray
        ///
        NdArray<dtype>& at(uint32 inIndex)
        {
            return cube_.at(inIndex);
        }

        //============================================================================
        /// Const access method, with bounds checking. Returns the 2d z "slice" element of the cube.
        ///
        /// @param inIndex
        ///
        /// @return NdArray
        ///
        [[nodiscard]] const NdArray<dtype>& at(uint32 inIndex) const
        {
            return cube_.at(inIndex);
        }

        //============================================================================
        /// Returns a reference to the last 2d "slice" of the cube in the z-axis
        ///
        /// @return NdArray&
        ///
        NdArray<dtype>& back() noexcept
        {
            return cube_.back();
        }

        //============================================================================
        /// Returns an iterator to the first 2d z "slice" of the cube.
        ///
        /// @return iterator
        ///
        [[nodiscard]] iterator begin() noexcept
        {
            return cube_.begin();
        }

        //============================================================================
        /// Returns an const_iterator to the first 2d z "slice" of the cube.
        ///
        /// @return const_iterator
        ///
        [[nodiscard]] const_iterator begin() const noexcept
        {
            return cube_.cbegin();
        }

        //============================================================================
        /// Returns an const_iterator to the first 2d z "slice" of the cube.
        ///
        /// @return const_iterator
        ///
        [[nodiscard]] const_iterator cbegin() const noexcept
        {
            return cube_.cbegin();
        }

        //============================================================================
        /// Outputs the DataCube as a .bin file
        ///
        /// @param inFilename
        ///
        void dump(const std::string& inFilename) const
        {
            std::filesystem::path f(inFilename);
            if (!f.has_extension())
            {
                f.replace_extension("bin");
            }

            std::ofstream ofile(f.c_str(), std::ios::binary);
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
        /// Tests whether or not the container is empty
        ///
        /// @return bool
        ///
        bool isempty() noexcept
        {
            return cube_.empty();
        }

        //============================================================================
        /// Returns an iterator to 1 past the last 2d z "slice" of the cube.
        ///
        /// @return iterator
        ///
        [[nodiscard]] iterator end() noexcept
        {
            return cube_.end();
        }

        //============================================================================
        /// Returns an const_iterator to 1 past the last 2d z "slice" of the cube.
        ///
        /// @return const_iterator
        ///
        [[nodiscard]] const_iterator end() const noexcept
        {
            return cube_.cend();
        }

        //============================================================================
        /// Returns an const_iterator to 1 past the last 2d z "slice" of the cube.
        ///
        /// @return const_iterator
        ///
        [[nodiscard]] const_iterator cend() const noexcept
        {
            return cube_.cend();
        }

        //============================================================================
        /// Returns a reference to the front 2d "slice" of the cube in the z-axis
        ///
        /// @return NdArray&
        ///
        NdArray<dtype>& front() noexcept
        {
            return cube_.front();
        }

        //============================================================================
        /// Returns the x/y shape of the cube
        ///
        /// @return Shape
        ///
        [[nodiscard]] const Shape& shape() const noexcept
        {
            return elementShape_;
        }

        //============================================================================
        /// Returns the size of the z-axis of the cube
        ///
        /// @return size
        ///
        [[nodiscard]] uint32 sizeZ() const noexcept
        {
            return static_cast<uint32>(cube_.size());
        }

        //============================================================================
        /// Removes the last z "slice" of the cube
        ///
        void pop_back() noexcept
        {
            cube_.pop_back();
        }

        //============================================================================
        /// Adds a new z "slice" to the end of the cube
        ///
        /// @param inArray
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
        /// Slices the z dimension of the cube
        ///
        /// @param inIndex: the flattend 2d index (row, col) to slice
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<dtype> sliceZAll(int32 inIndex) const
        {
            if (inIndex < 0)
            {
                inIndex += elementShape_.size();
            }

            NdArray<dtype> returnArray(1, sizeZ());

            for (uint32 i = 0; i < sizeZ(); ++i)
            {
                returnArray[i] = cube_[i][inIndex];
            }

            return returnArray;
        }

        //============================================================================
        /// Slices the z dimension of the cube
        ///
        /// @param inIndex: the flattend 2d index (row, col) to slice
        /// @param inSliceZ: the slice dimensions of the z-axis
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<dtype> sliceZ(int32 inIndex, Slice inSliceZ) const
        {
            if (inIndex < 0)
            {
                inIndex += elementShape_.size();
            }

            NdArray<dtype> returnArray(1, inSliceZ.numElements(sizeZ()));

            uint32 idx = 0;
            for (int32 i = inSliceZ.start; i < inSliceZ.stop; i += inSliceZ.step)
            {
                returnArray[idx++] = cube_[i][inIndex];
            }

            return returnArray;
        }

        //============================================================================
        /// Slices the z dimension of the cube with NO bounds checking
        ///
        /// @param inRow
        /// @param inCol
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<dtype> sliceZAll(int32 inRow, int32 inCol) const
        {
            if (inRow < 0)
            {
                inRow += elementShape_.rows;
            }

            if (inCol < 0)
            {
                inCol += elementShape_.cols;
            }

            NdArray<dtype> returnArray(1, sizeZ());

            for (uint32 i = 0; i < sizeZ(); ++i)
            {
                returnArray[i] = cube_[i](inRow, inCol);
            }

            return returnArray;
        }

        //============================================================================
        /// Slices the z dimension of the cube with NO bounds checking
        ///
        /// @param inRow
        /// @param inCol
        /// @param inSliceZ: the slice dimensions of the z-axis
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<dtype> sliceZ(int32 inRow, int32 inCol, Slice inSliceZ) const
        {
            if (inRow < 0)
            {
                inRow += elementShape_.rows;
            }

            if (inCol < 0)
            {
                inCol += elementShape_.cols;
            }

            NdArray<dtype> returnArray(1, inSliceZ.numElements(sizeZ()));

            uint32 idx = 0;
            for (int32 i = inSliceZ.start; i < inSliceZ.stop; i += inSliceZ.step)
            {
                returnArray[idx++] = cube_[i](inRow, inCol);
            }

            return returnArray;
        }

        //============================================================================
        /// Slices the z dimension of the cube with NO bounds checking
        ///
        /// @param inRow
        /// @param inCol
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<dtype> sliceZAll(Slice inRow, int32 inCol) const
        {
            if (inCol < 0)
            {
                inCol += elementShape_.cols;
            }

            NdArray<dtype> returnArray(inRow.numElements(elementShape_.rows), sizeZ());
            for (uint32 i = 0; i < sizeZ(); ++i)
            {
                returnArray.put(returnArray.rSlice(), i, cube_[i](inRow, inCol));
            }

            return returnArray;
        }

        //============================================================================
        /// Slices the z dimension of the cube with NO bounds checking
        ///
        /// @param inRow
        /// @param inCol
        /// @param inSliceZ: the slice dimensions of the z-axis
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<dtype> sliceZ(Slice inRow, int32 inCol, Slice inSliceZ) const
        {
            if (inCol < 0)
            {
                inCol += elementShape_.cols;
            }

            NdArray<dtype> returnArray(inRow.numElements(elementShape_.rows), inSliceZ.numElements(sizeZ()));
            uint32         idx = 0;
            for (int32 i = inSliceZ.start; i < inSliceZ.stop; i += inSliceZ.step)
            {
                returnArray.put(returnArray.rSlice(), idx++, cube_[i](inRow, inCol));
            }

            return returnArray;
        }

        //============================================================================
        /// Slices the z dimension of the cube with NO bounds checking
        ///
        /// @param inRow
        /// @param inCol
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<dtype> sliceZAll(int32 inRow, Slice inCol) const
        {
            if (inRow < 0)
            {
                inRow += elementShape_.rows;
            }

            NdArray<dtype> returnArray(inCol.numElements(elementShape_.cols), sizeZ());
            for (uint32 i = 0; i < sizeZ(); ++i)
            {
                returnArray.put(returnArray.rSlice(), i, cube_[i](inRow, inCol));
            }

            return returnArray;
        }

        //============================================================================
        /// Slices the z dimension of the cube with NO bounds checking
        ///
        /// @param inRow
        /// @param inCol
        /// @param inSliceZ: the slice dimensions of the z-axis
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<dtype> sliceZ(int32 inRow, Slice inCol, Slice inSliceZ) const
        {
            if (inRow < 0)
            {
                inRow += elementShape_.rows;
            }

            NdArray<dtype> returnArray(inCol.numElements(elementShape_.cols), inSliceZ.numElements(sizeZ()));
            uint32         idx = 0;
            for (int32 i = inSliceZ.start; i < inSliceZ.stop; i += inSliceZ.step)
            {
                returnArray.put(returnArray.rSlice(), idx++, cube_[i](inRow, inCol));
            }

            return returnArray;
        }

        //============================================================================
        /// Slices the z dimension of the cube with NO bounds checking
        ///
        /// @param inRow
        /// @param inCol
        /// @return DataCube
        ///
        DataCube<dtype> sliceZAll(Slice inRow, Slice inCol) const
        {
            DataCube<dtype> returnCube(sizeZ());
            for (uint32 i = 0; i < sizeZ(); ++i)
            {
                returnCube.push_back(cube_[i](inRow, inCol));
            }

            return returnCube;
        }

        //============================================================================
        /// Slices the z dimension of the cube with NO bounds checking
        ///
        /// @param inRow
        /// @param inCol
        /// @param inSliceZ: the slice dimensions of the z-axis
        /// @return DataCube
        ///
        DataCube<dtype> sliceZ(Slice inRow, Slice inCol, Slice inSliceZ) const
        {
            DataCube<dtype> returnCube(inSliceZ.numElements(sizeZ()));
            for (int32 i = inSliceZ.start; i < inSliceZ.stop; i += inSliceZ.step)
            {
                returnCube.push_back(cube_[i](inRow, inCol));
            }

            return returnCube;
        }

        //============================================================================
        /// Slices the z dimension of the cube with bounds checking
        ///
        /// @param inIndex: the flattend 2d index (row, col) to slice
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<dtype> sliceZAllat(int32 inIndex) const
        {
            if (inIndex < 0)
            {
                inIndex += elementShape_.size();
            }

            if (static_cast<uint32>(inIndex) >= elementShape_.size())
            {
                THROW_INVALID_ARGUMENT_ERROR("inIndex exceeds matrix dimensions.");
            }

            return sliceZAll(inIndex);
        }

        //============================================================================
        /// Slices the z dimension of the cube with bounds checking
        ///
        /// @param inIndex: the flattend 2d index (row, col) to slice
        /// @param inSliceZ: the slice dimensions of the z-axis
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<dtype> sliceZat(int32 inIndex, Slice inSliceZ) const
        {
            if (inIndex < 0)
            {
                inIndex += elementShape_.size();
            }

            if (static_cast<uint32>(inIndex) >= elementShape_.size())
            {
                THROW_INVALID_ARGUMENT_ERROR("inIndex exceeds matrix dimensions.");
            }

            auto numElements = inSliceZ.numElements(sizeZ());
            if (numElements > sizeZ())
            {
                THROW_INVALID_ARGUMENT_ERROR("inIndex exceeds matrix dimensions.");
            }

            return sliceZ(inIndex, inSliceZ);
        }

        //============================================================================
        /// Slices the z dimension of the cube with bounds checking
        ///
        /// @param inRow
        /// @param inCol
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<dtype> sliceZAllat(int32 inRow, int32 inCol) const
        {
            if (inRow < 0)
            {
                inRow += elementShape_.rows;
            }

            if (inCol < 0)
            {
                inCol += elementShape_.cols;
            }

            if (static_cast<uint32>(inRow) >= elementShape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("inRow exceeds matrix dimensions.");
            }

            if (static_cast<uint32>(inCol) >= elementShape_.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("inCol exceeds matrix dimensions.");
            }

            return sliceZAll(inRow, inCol);
        }

        //============================================================================
        /// Slices the z dimension of the cube with bounds checking
        ///
        /// @param inRow
        /// @param inCol
        /// @param inSliceZ: the slice dimensions of the z-axis
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<dtype> sliceZat(int32 inRow, int32 inCol, Slice inSliceZ) const
        {
            if (inRow < 0)
            {
                inRow += elementShape_.rows;
            }

            if (inCol < 0)
            {
                inCol += elementShape_.cols;
            }

            if (static_cast<uint32>(inRow) >= elementShape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("inRow exceeds matrix dimensions.");
            }
            if (static_cast<uint32>(inCol) >= elementShape_.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("inCol exceeds matrix dimensions.");
            }

            auto numElements = inSliceZ.numElements(sizeZ());
            if (numElements > sizeZ())
            {
                THROW_INVALID_ARGUMENT_ERROR("Index exceeds matrix dimensions.");
            }

            return sliceZ(inRow, inCol, inSliceZ);
        }

        //============================================================================
        /// Slices the z dimension of the cube with bounds checking
        ///
        /// @param inRow
        /// @param inCol
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<dtype> sliceZAllat(Slice inRow, int32 inCol) const
        {
            auto numRows = inRow.numElements(elementShape_.rows);
            if (numRows > elementShape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("inRow exceeds matrix dimensions.");
            }

            if (inCol < 0)
            {
                inCol += elementShape_.cols;
            }

            if (static_cast<uint32>(inCol) >= elementShape_.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("inCol exceeds matrix dimensions.");
            }

            return sliceZAll(inRow, inCol);
        }

        //============================================================================
        /// Slices the z dimension of the cube with bounds checking
        ///
        /// @param inRow
        /// @param inCol
        /// @param inSliceZ: the slice dimensions of the z-axis
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<dtype> sliceZat(Slice inRow, int32 inCol, Slice inSliceZ) const
        {
            auto numRows = inRow.numElements(elementShape_.rows);
            if (numRows > elementShape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("inRow exceeds matrix dimensions.");
            }

            if (inCol < 0)
            {
                inCol += elementShape_.cols;
            }

            if (static_cast<uint32>(inCol) >= elementShape_.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("inCol exceeds matrix dimensions.");
            }

            auto numElements = inSliceZ.numElements(sizeZ());
            if (numElements > sizeZ())
            {
                THROW_INVALID_ARGUMENT_ERROR("Index exceeds matrix dimensions.");
            }

            return sliceZ(inRow, inCol, inSliceZ);
        }

        //============================================================================
        /// Slices the z dimension of the cube with bounds checking
        ///
        /// @param inRow
        /// @param inCol
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<dtype> sliceZAllat(int32 inRow, Slice inCol) const
        {
            auto numCols = inCol.numElements(elementShape_.cols);
            if (numCols > elementShape_.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("inCol exceeds matrix dimensions.");
            }

            if (inRow < 0)
            {
                inRow += elementShape_.rows;
            }

            if (static_cast<uint32>(inRow) >= elementShape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("inRow exceeds matrix dimensions.");
            }

            return sliceZAll(inRow, inCol);
        }

        //============================================================================
        /// Slices the z dimension of the cube with bounds checking
        ///
        /// @param inRow
        /// @param inCol
        /// @param inSliceZ: the slice dimensions of the z-axis
        /// @return NdArray
        ///
        [[nodiscard]] NdArray<dtype> sliceZat(int32 inRow, Slice inCol, Slice inSliceZ) const
        {
            auto numCols = inCol.numElements(elementShape_.cols);
            if (numCols > elementShape_.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("inCol exceeds matrix dimensions.");
            }

            if (inRow < 0)
            {
                inRow += elementShape_.rows;
            }

            if (static_cast<uint32>(inRow) >= elementShape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("inRow exceeds matrix dimensions.");
            }

            auto numElements = inSliceZ.numElements(sizeZ());
            if (numElements > sizeZ())
            {
                THROW_INVALID_ARGUMENT_ERROR("Index exceeds matrix dimensions.");
            }

            return sliceZ(inRow, inCol, inSliceZ);
        }

        //============================================================================
        /// Slices the z dimension of the cube with bounds checking
        ///
        /// @param inRow
        /// @param inCol
        /// @return DataCube
        ///
        DataCube<dtype> sliceZAllat(Slice inRow, Slice inCol) const
        {
            if (inRow.numElements(elementShape_.rows) > elementShape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("inRow exceeds matrix dimensions.");
            }

            if (inCol.numElements(elementShape_.cols) > elementShape_.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("inCol exceeds matrix dimensions.");
            }

            return sliceZAll(inRow, inCol);
        }

        //============================================================================
        /// Slices the z dimension of the cube with bounds checking
        ///
        /// @param inRow
        /// @param inCol
        /// @param inSliceZ: the slice dimensions of the z-axis
        /// @return DataCube
        ///
        DataCube<dtype> sliceZat(Slice inRow, Slice inCol, Slice inSliceZ) const
        {
            if (inRow.numElements(elementShape_.rows) > elementShape_.rows)
            {
                THROW_INVALID_ARGUMENT_ERROR("inRow exceeds matrix dimensions.");
            }

            if (inCol.numElements(elementShape_.cols) > elementShape_.cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("inCol exceeds matrix dimensions.");
            }

            auto numElements = inSliceZ.numElements(sizeZ());
            if (numElements > sizeZ())
            {
                THROW_INVALID_ARGUMENT_ERROR("Index exceeds matrix dimensions.");
            }

            return sliceZ(inRow, inCol, inSliceZ);
        }

        //============================================================================
        /// Access operator, no bounds checking.  Returns the 2d z "slice" element of the cube.
        ///
        /// @param inIndex
        ///
        /// @return NdArray
        ///
        NdArray<dtype>& operator[](uint32 inIndex) noexcept
        {
            return cube_[inIndex];
        }

        //============================================================================
        /// Const access operator, no bounds checking. Returns the 2d z "slice" element of the cube.
        ///
        /// @param inIndex
        ///
        /// @return NdArray
        ///
        const NdArray<dtype>& operator[](uint32 inIndex) const noexcept
        {
            return cube_[inIndex];
        }

    private:
        //================================Attributes==================================
        std::vector<NdArray<dtype>> cube_{};
        Shape                       elementShape_{ 0, 0 };
    };
} // namespace nc
