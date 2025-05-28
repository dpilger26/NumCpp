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
/// Binary data logger
///
#pragma once

#ifndef NUMCPP_NO_USE_BOOST

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>

#include "boost/algorithm/string.hpp"
#include "boost/core/demangle.hpp"

namespace nc::logger
{
    namespace detail
    {
        namespace type_traits
        {
            /**
             * @brief type trait to check if a type has a serialize method with the correct signature
             */
            template<typename DataType>
            using serialize_t = decltype(std::declval<DataType>().serialize());

            /**
             * @brief type trait to check if a type has a serialize method with the correct signature
             */
            template<typename DataType, typename = std::void_t<>>
            class has_serialize : std::false_type
            {
            public:
                static constexpr bool value = false;
            };

            /**
             * @brief type trait to check if a type has a serialize method with the correct signature
             */
            template<typename DataType>
            class has_serialize<DataType,
                                std::void_t<std::enable_if_t<std::is_same_v<serialize_t<DataType>, std::string>, int>>>
            {
            public:
                static constexpr bool value = true;
            };

            /**
             * @brief type trait to check if a type has a serialize method with the correct signature
             */
            template<typename DataType>
            inline constexpr bool has_serialize_v = has_serialize<DataType>::value;
        } // namespace type_traits

        /**
         * @brief Binary Logger
         */
        template<typename DataType>
        class BinaryDataLogger
        {
        public:
            using value_type      = DataType;
            using const_pointer   = const DataType* const;
            using const_reference = const DataType&;

            // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, hicpp-avoid-c-arrays, modernize-avoid-c-arrays)
            static constexpr char LOG_EXT[]                      = ".log";
            static constexpr auto DATA_ELEMENT_SIZE              = sizeof(value_type);
            static constexpr auto DATE_TYPE_HAS_SERIALIZE_METHOD = type_traits::has_serialize_v<value_type>;

            /**
             * @brief Default constructor
             */
            BinaryDataLogger() { }

            /**
             * @brief Constructor
             * @param outputDir: the directory to place the output log
             */
            explicit BinaryDataLogger(std::filesystem::path outputDir)
            {
                setOutputDir(outputDir);
            }

            /**
             * @brief Constructor
             * @param outputDir: the directory to place the output log
             */
            explicit BinaryDataLogger(std::string_view outputDir) :
                BinaryDataLogger(std::filesystem::path(outputDir))
            {
            }

            /**
             * @brief Destructor
             */
            ~BinaryDataLogger()
            {
                ofile_.close();
            }

            // explicitly delete
            BinaryDataLogger(const BinaryDataLogger&)            = delete;
            BinaryDataLogger(BinaryDataLogger&&)                 = delete;
            BinaryDataLogger& operator=(const BinaryDataLogger&) = delete;
            BinaryDataLogger& operator=(BinaryDataLogger&&)      = delete;

            /**
             * @brief The log file path
             */
            const std::filesystem::path& filepath() const noexcept
            {
                return filepath_;
            }

            /**
             * @brief Sets the output log directory.
             *
             * @param outputDir: the output directory
             */
            void setOutputDir(std::filesystem::path outputDir)
            {
                if (std::filesystem::is_directory(outputDir))
                {
                    auto dataTypeName = boost::core::demangle(typeid(DataType).name());
                    boost::algorithm::replace_all(dataTypeName, "::", "_");
                    boost::algorithm::replace_all(dataTypeName, "<", "_");
                    boost::algorithm::replace_all(dataTypeName, ">", "_");
                    const auto filename = std::filesystem::path(dataTypeName).replace_extension(LOG_EXT);
                    filepath_           = std::filesystem::canonical(outputDir) / filename;

                    ofile_ = std::ofstream(filepath_.c_str(), std::ios::out | std::ios::binary);
                    if (!ofile_.good())
                    {
                        throw std::runtime_error("Unable to open the log file:\n\t" + filepath_.string());
                    }
                }
                else
                {
                    throw std::runtime_error("The provided output log directory is not valid:\n\t" +
                                             outputDir.string());
                }
            }

            /**
             * @brief Sets the output log directory.
             *
             * @param outputDir: the output directory
             */
            void setOutputDir(std::string_view outputDir)
            {
                setOutputDir(std::filesystem::path(outputDir));
            }

            /**
             * @brief Enable the logger
             */
            void enable() noexcept
            {
                enabled_ = true;
            }

            /**
             * @brief Disable the logger
             */
            void disable() noexcept
            {
                enabled_ = false;
            }

            /**
             * @brief Checks whether logger is enabled
             */
            bool isEnabled() noexcept
            {
                return enabled_;
            }

            /**
             * @brief Force a flush of the output stream
             */
            void flush()
            {
                ofile_.flush();
            }

            /**
             * @brief Logs the data element
             * @param dataElement: the data element
             */
            void log(const_reference dataElement)
            {
                if (!enabled_)
                {
                    return;
                }

                if (filepath_.empty())
                {
                    throw std::runtime_error("The output log directory does not exist");
                }

                if constexpr (DATE_TYPE_HAS_SERIALIZE_METHOD)
                {
                    const auto serializedData = dataElement.serialize();
                    ofile_.write(serializedData.data(), serializedData.size());
                }
                else
                {
                    ofile_.write(reinterpret_cast<const char*>(&dataElement), DATA_ELEMENT_SIZE);
                }
            }

            /**
             * @brief Logs the data elements
             * @param dataElements: the data element pointer
             * @param numElements: the number of data elements to log
             */
            void log(const_pointer dataElements, std::size_t numElements)
            {
                if (!enabled_)
                {
                    return;
                }

                if (filepath_.empty())
                {
                    throw std::runtime_error("The output log directory does not exist");
                }

                std::for_each(dataElements,
                              dataElements + numElements,
                              [this](const_reference dataElement) { log(dataElement); });
            }

        private:
            std::filesystem::path filepath_{};
            std::ofstream         ofile_;
            bool                  enabled_{ true };
        };
    } // namespace detail

    /**
     * @brief Binary Logger Singleton
     */
    class BinaryLogger
    {
    public:
        // explicitly delete
        BinaryLogger(const BinaryLogger&)            = delete;
        BinaryLogger(BinaryLogger&&)                 = delete;
        BinaryLogger& operator=(const BinaryLogger&) = delete;
        BinaryLogger& operator=(BinaryLogger&&)      = delete;

        /**
         * @brief Singleton instance getter
         *
         * @returns: singleton instance of BinaryLogger
         */
        static BinaryLogger& getInstance() noexcept
        {
            static BinaryLogger binaryLogger;
            return binaryLogger;
        }

        /**
         * @brief Sets the output directory. This should be called BEFORE any type loggers
         *        have been created, and will NOT reset the output directory afterwards!
         *
         * @param outputDir: the output directory
         */
        void setOutputDir(const std::filesystem::path& outputDir)
        {
            if (!std::filesystem::is_directory(outputDir))
            {
                throw std::runtime_error("outputDir does not exist");
            }

            outputDir_ = outputDir;
        }

        /**
         * @brief Sets the output directory. This should be called BEFORE any type loggers
         *        have been created, and will NOT reset the output directory afterwards!
         *
         * @param outputDir: the output directory
         */
        void setOutputDir(std::string_view outputDir)
        {
            setOutputDir(std::filesystem::path(outputDir));
        }

        /**
         * @brief Gets the logger instance for the specific data type
         *
         * @returns data type logger instance
         */
        template<typename DataType>
        detail::BinaryDataLogger<DataType>& getTypeLogger()
        {
            static detail::BinaryDataLogger<DataType> typeLogger(outputDir_);
            return typeLogger;
        }

    private:
        std::filesystem::path outputDir_{ "." };

        /**
         * @brief Constructor
         */
        BinaryLogger() = default;
    };
} // namespace nc::logger

#endif // #ifndef NUMCPP_NO_USE_BOOST
