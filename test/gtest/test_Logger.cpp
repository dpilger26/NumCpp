// UNCLASSIFIED

#include "gtest/gtest.h"

#include <filesystem>
#include <string>

#include "NumCpp/Logging/Logger.hpp"

namespace nc::logger
{
    std::string getLogLevelStr(boost::log::trivial::severity_level level)
    {
        std::string logLevelStr{};
        if (level == boost::log::trivial::severity_level::trace)
        {
            logLevelStr = "trace";
        }
        else if (level == boost::log::trivial::severity_level::info)
        {
            logLevelStr = "info";
        }
        else if (level == boost::log::trivial::severity_level::debug)
        {
            logLevelStr = "debug";
        }
        else if (level == boost::log::trivial::severity_level::warning)
        {
            logLevelStr = "warning";
        }
        else if (level == boost::log::trivial::severity_level::error)
        {
            logLevelStr = "error";
        }
        else if (level == boost::log::trivial::severity_level::fatal)
        {
            logLevelStr = "fatal";
        }

        return logLevelStr;
    }

    void logOut(boost::log::trivial::severity_level level)
    {
        setLogLevel(boost::log::trivial::severity_level::trace);
        LOG_TRACE << "Setting log level to '" << getLogLevelStr(level) << "'";
        setLogLevel(level);

        LOG_DEBUG << "This is a debug line";
        LOG_INFO << "This is an info line";
        LOG_WARNING << "This is a warning line";
        LOG_ERROR << "This is an error line";
        LOG_FATAL << "This is a fatal line";
    }

    void logAllSeverities()
    {
        LOG_TRACE << "Logging all severity levels:";
        logOut(boost::log::trivial::severity_level::trace);
        logOut(boost::log::trivial::severity_level::debug);
        logOut(boost::log::trivial::severity_level::info);
        logOut(boost::log::trivial::severity_level::warning);
        logOut(boost::log::trivial::severity_level::error);
        logOut(boost::log::trivial::severity_level::fatal);
    }

    class LoggerTestSuite : public ::testing::Test
    {
    };

    /**
     * @brief test Console Logger
     */
    TEST_F(LoggerTestSuite, TestConsoleLogger)
    {
        logAllSeverities();
        SUCCEED();
    }

    /**
     * @brief test File Logger
     */
    TEST_F(LoggerTestSuite, TestFileLogger)
    {
        namespace fs              = std::filesystem;
        const fs::path tempPath   = fs::temp_directory_path();
        fs::path       loggerFile = tempPath / "TestLog.log";
        addOutputFileLog(loggerFile);

        logAllSeverities();
        ASSERT_TRUE(fs::exists(loggerFile));
        ASSERT_TRUE(fs::is_regular_file(loggerFile));

        fs::path newLogFile = tempPath / "another/dir/newLog.log";
        addOutputFileLog(newLogFile);

        logAllSeverities();
        ASSERT_TRUE(fs::exists(newLogFile));
        ASSERT_TRUE(fs::is_regular_file(newLogFile));
    }
} // namespace nc::logger

// UNCLASSIFIED
