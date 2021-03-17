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
/// Custom iterators for the NdArray class
///
#pragma once

#include "NdArrayIterators.hpp"
#include "Types.hpp"
#include "TypeTraits.hpp"
#include "Utils.hpp"

#include "NumCpp.hpp"

#include <array>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace nc_develop
{
    //================================================================================
    // Class Description:
    ///	The main work horse of the NumCpp library
    template<typename dtype, class Allocator = std::allocator<dtype>,
        std::enable_if_t<is_valid_dtype_v<dtype>, int> = 0, // DP NOTE: remove nc:: when integrating
        std::enable_if_t<std::is_same_v<dtype, typename Allocator::value_type>, int> = 0>
    class NdArray
    {
    private:
        using AllocType     = typename std::allocator_traits<Allocator>::template rebind_alloc<dtype>;
        using AllocTraits   = std::allocator_traits<AllocType>;

        struct PrivateTag {};
        struct ErrorCheckingTag {};
        struct NoErrorCheckingTag {};

    public:
        //Type alliases=============================================================
        using value_type                = dtype;
        using allocator_type            = Allocator;
        using pointer                   = typename AllocTraits::pointer;
        using const_pointer             = typename AllocTraits::const_pointer;
        using reference                 = dtype&;
        using const_reference           = const dtype&;
        using size_type                 = std::size_t;
        using difference_type           = typename AllocTraits::difference_type;
        using shared_ptr                = std::shared_ptr<dtype>;
        using const_shared_ptr          = const shared_ptr;

        using iterator                  = NdArrayIterator<dtype, pointer, difference_type>;
        using const_iterator            = NdArrayConstIterator<dtype, const_pointer, difference_type>;
        using reverse_iterator          = std::reverse_iterator<iterator>;
        using const_reverse_iterator    = std::reverse_iterator<const_iterator>;

        //============================================================================
        // Method Description:
        ///	Defualt Constructor
        ///
        NdArray() = default;

        //============================================================================
        // Method Description:
        ///	Constructor, constructs an empty array of the input shape
        /// @param dimSizes: the sizes of the dimensions of the array
        ///
        template<typename ... DimSizes,
            std::enable_if_t<greater_than_zero_v<sizeof...(DimSizes)>, int> = 0,
            std::enable_if_t<all_convertable_v<size_type, DimSizes...>, int> = 0>
            NdArray(DimSizes ... dimSizes) :
            NdArray((... * dimSizes), { static_cast<size_type>(dimSizes)... })
        {}

        //============================================================================
        // Method Description:
        ///	Constructor, constructs an empty array of the input shape
        /// @param shape: container with the sizes of the dimensions of the array
        ///
        template<typename ContainerType,
            std::enable_if_t<is_conforming_container_v<ContainerType>, int> = 0>
            NdArray(const ContainerType& shape) :
            NdArray(std::accumulate(shape.begin(), shape.end(), size_type{ 1 }, std::multiplies<size_type>()), 
                shape_t{shape.begin(), shape.end()})
        {}

        //============================================================================
        // Method Description:
        ///	1D Array Constructor
        /// @param listArray
        ///
        NdArray(std::initializer_list<dtype> listArray) : 
            NdArray(listArray.size(), { listArray.size() })
        {
            // DP NOTE: remove nc:: when integrating
            nc::stl_algorithms::copy(listArray.begin(), listArray.end(), begin());
        }

        //============================================================================
        // Method Description:
        ///	2D Array Constructor
        /// @param listArray
        ///
        NdArray(std::initializer_list<std::initializer_list<dtype>> listArray)
        {
            const auto dim0Size = listArray.size();
            size_type dim1Size = 0;

            for (const auto& dim1List : listArray)
            {
                if (dim1Size == 0)
                {
                    dim1Size = static_cast<size_type>(dim1List.size());
                }
                else if (dim1List.size() != dim1Size)
                {
                    THROW_INVALID_ARGUMENT_ERROR("All rows of the initializer listArray must have the same number of elements");
                }
            }

            allocator_.setSize(dim0Size * dim1Size);

            shape_.reserve(2);
            shape_.push_back(dim0Size);
            shape_.push_back(dim1Size);

            constructStridesAndAllocateData();

            auto iter = begin();
            for (const auto& dim1List : listArray)
            {
                // DP NOTE: remove nc:: when integrating
                nc::stl_algorithms::copy(dim1List.begin(), dim1List.end(), iter);
                iter += strides_.front();
            }
        }

        //============================================================================
        // Method Description:
        ///	3D Array Constructor
        /// @param listArray
        ///
        NdArray(std::initializer_list<std::initializer_list<std::initializer_list<dtype>>> listArray)
        {
            const auto dim0Size = listArray.size();
            size_type dim1Size = 0;
            size_type dim2Size = 0;

            for (const auto& dim1List : listArray)
            {
                if (dim1Size == 0)
                {
                    dim1Size = static_cast<size_type>(dim1List.size());
                }
                else if (dim1List.size() != dim1Size)
                {
                    THROW_INVALID_ARGUMENT_ERROR("All rows of the initializer listArray must have the same number of elements");
                }

                for (const auto& dim2List : dim1List)
                {
                    if (dim2Size == 0)
                    {
                        dim2Size = static_cast<size_type>(dim2List.size());
                    }
                    else if (dim2List.size() != dim2Size)
                    {
                        THROW_INVALID_ARGUMENT_ERROR("All columns of the initializer listArray must have the same number of elements");
                    }
                }
            }

            allocator_.setSize(dim0Size * dim1Size * dim2Size);

            shape_.reserve(3);
            shape_.push_back(dim0Size);
            shape_.push_back(dim1Size);
            shape_.push_back(dim2Size);

            constructStridesAndAllocateData();

            auto iter = begin();
            for (const auto& dim1List : listArray)
            {
                for (const auto& dim2List : dim1List)
                {
                    // DP NOTE: remove nc:: when integrating
                    nc::stl_algorithms::copy(dim2List.begin(), dim2List.end(), iter);
                    iter += strides_[1];
                }
            }
        }

        ~NdArray() = default;

        reference operator[](size_type index) noexcept
        {
            return data_.get()[index];
        }

        const_reference operator[](size_type index) const noexcept 
        {
            return data_.get()[index];
        }

        template<typename ... Indices,
            std::enable_if_t<greater_than_zero_v<sizeof...(Indices)>, int> = 0,
            std::enable_if_t<all_convertable_v<size_type, Indices...>, int> = 0>
        reference operator()(Indices ... indices)
        {
            return operator[](flatIndex(NoErrorCheckingTag{}, indices...));
        }

        template<typename ... Indices,
            std::enable_if_t<greater_than_zero_v<sizeof...(Indices)>, int> = 0,
            std::enable_if_t<all_convertable_v<size_type, Indices...>, int> = 0>
        const_reference operator()(Indices ... indices) const
        {
            return operator[](flatIndex(NoErrorCheckingTag{}, indices...));
        }

        template<typename ... Indices,
            std::enable_if_t<greater_than_zero_v<sizeof...(Indices)>, int> = 0,
            std::enable_if_t<all_convertable_v<size_type, Indices...>, int> = 0>
            reference at(Indices ... indices)
        {
            return operator[](flatIndex(ErrorCheckingTag{}, indices...));
        }

        template<typename ... Indices,
            std::enable_if_t<greater_than_zero_v<sizeof...(Indices)>, int> = 0,
            std::enable_if_t<all_convertable_v<size_type, Indices...>, int> = 0>
            const_reference at(Indices ... indices) const
        {
            return operator[](flatIndex(ErrorCheckingTag{}, indices...));
        }

        // DP NOTE: on operator()
        // DP NOTE: this class is responsible for supplying the correct position of the begining and end pointers to the
        //          iterator classes
        // DP NOTE: this class is responsible for multiplying the strides by the slice steps before passing to the 
        //          iterator classes.
        // DP NOTE: if number of indices < number of dimensions then return new array.  Memory slice instead of copy would be nice... but how?
        //          use non-owning shell functionality, would require reference counting though so that underlying data doesn't go out of scope?
        // DP NOTE: if number of indices > number of dimensions then throw exception
        // DP NOTE: if number of indices == number of dimensions then return reference scaler



        //============================================================================
        // Method Description:
        ///	iterator to the beginning of the flattened array
        /// @return iterator
        ///
        iterator begin() noexcept 
        {
            return iterator(data_);
        }

        //============================================================================
        // Method Description:
        ///	const iterator to the beginning of the flattened array
        /// @return const_iterator
        ///
        const_iterator begin() const noexcept 
        {
            return cbegin();
        }

        //============================================================================
        // Method Description:
        ///	const iterator to the beginning of the flattened array
        ///
        /// @return const_iterator
        ///
        const_iterator cbegin() const noexcept 
        {
            return const_iterator(data_);
        }

        //============================================================================
        // Method Description:
        ///	iterator to 1 past the end of the flattened array
        /// @return iterator
        ///
        iterator end() noexcept 
        {
            return begin() += size();
        }

        //============================================================================
        // Method Description:
        ///	const iterator to 1 past the end of the flattened array
        /// @return const_iterator
        ///
        const_iterator end() const noexcept 
        {
            return cend();
        }

        //============================================================================
        // Method Description:
        ///	const iterator to 1 past the end of the flattened array
        /// @return const_iterator
        ///
        const_iterator cend() const noexcept 
        {
            return cbegin() += size();
        }

        size_type size() const noexcept
        {
            return allocator_.size();
        }

        shape_t shape() const noexcept
        {
            return shape_;
        }

        strides_t strides() const noexcept
        {
            auto strides = strides_;
            std::for_each(strides.begin(), strides.end(),
                [](auto& value) -> void
                {
                    value *= sizeof(dtype);
                }
            );
            return strides;
        }

        size_type ndim() const noexcept
        {
            return shape_.size();
        }

        constexpr size_type itemSize() const noexcept
        {
            return sizeof(dtype);
        }

        template<typename ... DimSizes,
            std::enable_if_t<greater_than_zero_v<sizeof...(DimSizes)>, int> = 0,
            std::enable_if_t<all_convertable_v<size_type, DimSizes...>, int> = 0>
        void reshape(DimSizes ... dimSizes)
        {
            reshape({ static_cast<size_type>(dimSizes)... }, PrivateTag{});
        }

        template<typename ContainerType,
            std::enable_if_t<is_conforming_container_v<ContainerType>, int> = 0>
        void reshape(const ContainerType& shape)
        {
            reshape(shape_t{shape.begin(), shape.end()}, PrivateTag{});
        }

        void printInfo() const
        {
            std::cout << "size_   : " << allocator_.size() << '\n';
            std::cout << "shape_  : ";
            utils::printContainer(shape_);
            std::cout << "strides_: ";
            utils::printContainer(strides_);
        }

    private:
        NdArray(size_type size, shape_t&& shape) :
            allocator_(allocator_type{}, size),
            shape_(std::move(shape))
        {
            constructStridesAndAllocateData();
        }

        template<typename ErrorChecking, typename ... Indices,
            std::enable_if_t<greater_than_zero_v<sizeof...(Indices)>, int> = 0,
            std::enable_if_t<all_convertable_v<size_type, Indices...>, int> = 0>
        size_type flatIndex(ErrorChecking errorChecking, Indices ... indices) const
        {
            return flatIndex(errorChecking, std::array<size_type, (sizeof...(indices))>{ static_cast<size_type>(indices)... });
        }

        template<typename ErrorChecking, typename ContainerType,
            std::enable_if_t<is_conforming_container_v<ContainerType>, int> = 0>
        size_type flatIndex(ErrorChecking, const ContainerType&& indices) const
        {
            size_type flatIndex = indices.front();
            if (indices.size() > 1)
            {
                if (indices.size() != shape_.size())
                {
                    std::string errStr = "input number of indices " + std::to_string(indices.size());
                    errStr += " is different than the number of dimensions " + std::to_string(shape_.size());
                    throw std::invalid_argument(errStr);
                }

                flatIndex *= strides_.front();
                flatIndex += std::inner_product(indices.begin() + 1, indices.end(), strides_.begin() + 1, size_type{ 0 });
            }

            if constexpr (std::is_same_v<ErrorChecking, ErrorCheckingTag>)
            {
                if (flatIndex >= allocator_.size())
                {
                    std::string errStr = "invalid index " + std::to_string(flatIndex) + " for array of size " + 
                        std::to_string(allocator_.size());
                    throw std::invalid_argument(errStr);
                }
            }

            return flatIndex;
        }

        void reshape(shape_t&& shape, PrivateTag)
        {
            const auto newSize = std::accumulate(shape.begin(), shape.end(), size_type{ 1 }, std::multiplies<size_type>());
            if (newSize != allocator_.size())
            {
                std::string errStr = "cannot reshape array of size " + std::to_string(allocator_.size()) + 
                    " into shape " + utils::stringifyContainer(shape);
                throw std::invalid_argument(errStr);
            }

            shape_ = shape;
            constructStrides();
        }

        void allocateData()
        {
            data_.reset(allocator_.allocate(), allocator_);

#ifdef DEBUG
            // DP NOTE: for testing, remove later
            std::iota(data_.get(), data_.get() + allocator_.size(), dtype{ 0 });
#endif
        }

        void constructStrides()
        {
            strides_.reserve(shape_.size());
            auto stride = allocator_.size();
            for (const auto dimSize : shape_)
            {
                stride /= dimSize; // integer division
                strides_.push_back(stride);
            }
        }

        void constructStridesAndAllocateData()
        {
            constructStrides();
            allocateData();
        }

        class AllocatorDeleter
        {
        public:
            AllocatorDeleter(allocator_type&& allocator, size_type size) : 
                allocator_(std::move(allocator)),
                size_(size)
            {}

            size_type size() const noexcept
            {
                return size_;
            }

            void setSize(size_type size) noexcept
            {
                size_ = size;
            }

            pointer allocate()
            {
                return allocator_.allocate(size_);
            }

            void operator()(pointer ptr) noexcept
            {
                allocator_.deallocate(ptr, size_);
            }

        private:
            allocator_type  allocator_{};
            size_type       size_{0};
        };

        AllocatorDeleter        allocator_{ allocator_type{}, 0 };
        shape_t                 shape_{};
        strides_t               strides_{};
        shared_ptr              data_{ nullptr };
    };
}
