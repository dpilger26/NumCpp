// UNCLASSIFIED

#ifndef NUMCPP_NO_USE_BOOST

#include "gtest/gtest.h"

#include <chrono>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "NumCpp/Logging/BinaryLogger.hpp"

namespace nc::logger
{
    class BinaryLoggerTestSuite : public ::testing::Test
    {
    };

    class BaseDataType
    {
    public:
        std::size_t SIZE = sizeof(BaseDataType);

        BaseDataType() = default;

        BaseDataType(int a, double b, bool c, char d) noexcept :
            a_(a),
            b_(b),
            c_(c),
            d_(d)
        {
        }

        int& a() noexcept
        {
            return a_;
        }

        double& b() noexcept
        {
            return b_;
        }

        bool& c() noexcept
        {
            return c_;
        }

        char& d() noexcept
        {
            return d_;
        }

    private:
        int    a_{ 666 };
        double b_{ 3.14159 };
        bool   c_{ true };
        char   d_{ 'd' };
    };

    /**
     * @brief test Binary Data Logger constructor
     */
    TEST_F(BinaryLoggerTestSuite, TestBinaryDataLoggerConstructors)
    {
        auto       defaultDataLogger = detail::BinaryDataLogger<BaseDataType>();
        const auto data              = BaseDataType{};
        auto       dataVec           = std::vector<BaseDataType>(10);
        // Filepath is empty; exceptions should be thrown
        ASSERT_TRUE(defaultDataLogger.filepath().empty());
        ASSERT_THROW(defaultDataLogger.log(data), std::runtime_error);
        ASSERT_THROW(defaultDataLogger.log(dataVec.data(), dataVec.size()), std::runtime_error);

        namespace fs                   = std::filesystem;
        const fs::path validPath       = fs::temp_directory_path();
        auto           validDataLogger = detail::BinaryDataLogger<BaseDataType>(validPath);
        // Valid filepath, logging should succeed
        ASSERT_NO_THROW(validDataLogger.log(data));
        ASSERT_NO_THROW(validDataLogger.log(dataVec.data(), dataVec.size()));
        const fs::path invalidPath = "/invalid/path";
        // Invalid filepath, error thrown on instantiation
        ASSERT_THROW(auto invalidDataLogger = detail::BinaryDataLogger<BaseDataType>(invalidPath), std::runtime_error);
    }

    /**
     * @brief test Binary Data Logger with basic type
     */
    TEST_F(BinaryLoggerTestSuite, TestBinaryLoggerBasic)
    {
        auto& binaryLogger = BinaryLogger::getInstance();
        auto& typeLogger   = binaryLogger.template getTypeLogger<BaseDataType>();

        ASSERT_NO_THROW(typeLogger.enable());
        ASSERT_NO_THROW(typeLogger.disable());
        ASSERT_NO_THROW(typeLogger.enable());

        constexpr int NUM_ELEMENTS = 10;

        const auto data = BaseDataType{};
        for (auto i = 0; i < NUM_ELEMENTS; ++i)
        {
            ASSERT_NO_THROW(typeLogger.log(data));
        }

        auto dataVec = std::vector<BaseDataType>(NUM_ELEMENTS);
        ASSERT_NO_THROW(typeLogger.log(dataVec.data(), dataVec.size()));

        ASSERT_TRUE(std::filesystem::is_regular_file(typeLogger.filepath()));
        const auto expectedSizeBytes = sizeof(BaseDataType) * (NUM_ELEMENTS + dataVec.size());
        typeLogger.flush();
        ASSERT_EQ(std::filesystem::file_size(typeLogger.filepath()), expectedSizeBytes);
    }

    template<typename T>
    class SerialDataType
    {
    public:
        using DataType                     = T;
        static constexpr auto DateTypeSize = sizeof(T);

        SerialDataType() = default;

        explicit SerialDataType(std::size_t size) :
            size_(size),
            data_(size)
        {
        }

        [[nodiscard]] std::size_t numBytes() const noexcept
        {
            return sizeof(size_) + DateTypeSize * size_;
        }

        [[nodiscard]] std::string serialize() const
        {
            std::string serialized;
            serialized.resize(numBytes());
            memcpy(serialized.data(), &size_, sizeof(size_));
            memcpy(serialized.data() + sizeof(size_), data_.data(), data_.size() * DateTypeSize);
            return serialized;
        }

    private:
        std::size_t    size_{ 1 };
        std::vector<T> data_{ 1 };
    };

    /**
     * @brief test Binary Data Logger with serialize type
     */
    TEST_F(BinaryLoggerTestSuite, TestBinaryLoggerSerialize)
    {
        using SerialDataTypeDouble = SerialDataType<double>;

        auto& binaryLogger = BinaryLogger::getInstance();
        auto& typeLogger   = binaryLogger.template getTypeLogger<SerialDataTypeDouble>();

        ASSERT_NO_THROW(typeLogger.enable());
        ASSERT_NO_THROW(typeLogger.disable());
        ASSERT_NO_THROW(typeLogger.enable());

        constexpr int NUM_ELEMENTS = 10;

        const auto data = SerialDataTypeDouble{ 10 };
        for (auto i = 0; i < NUM_ELEMENTS; ++i)
        {
            ASSERT_NO_THROW(typeLogger.log(data));
        }

        auto dataVec = std::vector<SerialDataTypeDouble>{};
        for (auto i = 0; i < NUM_ELEMENTS; ++i)
        {
            dataVec.push_back(SerialDataTypeDouble(NUM_ELEMENTS));
        }
        ASSERT_NO_THROW(typeLogger.log(dataVec.data(), dataVec.size()));

        ASSERT_TRUE(std::filesystem::is_regular_file(typeLogger.filepath()));
        const auto expectedSizeBytes = data.numBytes() * (NUM_ELEMENTS + dataVec.size());
        typeLogger.flush();
        ASSERT_EQ(std::filesystem::file_size(typeLogger.filepath()), expectedSizeBytes);
    }
} // namespace nc::logger

#endif // #ifndef NUMCPP_NO_USE_BOOST

// UNCLASSIFIED
