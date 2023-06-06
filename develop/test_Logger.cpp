// UNCLASSIFIED

#include <filesystem>
#include <string>

#include "gtest/gtest.h"
#include "test.h"

#include "ball/common/utils/logger/Logger.h"

namespace ball::common::utils::logger
{
	std::string getLogLevelStr(boost::log::trivial::severity_level level)
	{
		std::string logLevelStr{};
		if(level == boost::log::trivial::severity_level::trace)
		{
			logLevelStr = "trace";
		}
		else if(level == boost::log::trivial::severity_level::info)
		{
			logLevelStr = "info";
		}
		else if(level == boost::log::trivial::severity_level::debug)
		{
			logLevelStr = "debug";
		}
		else if(level == boost::log::trivial::severity_level::warning)
		{
			logLevelStr = "warning";
		}
		else if(level == boost::log::trivial::severity_level::error)
		{
			logLevelStr = "error";
		}
		else if(level == boost::log::trivial::severity_level::fatal)
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

	class LoggerTestSuite : public BaseTest
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
		namespace fs = std::filesystem;
		const fs::path currentPath = fs::current_path();
		fs::path loggerFile = currentPath / "TestLog.log";
		addOutputFileLog(loggerFile);

		logAllSeverities();
		ASSERT_TRUE(fs::exists(loggerFile));
		ASSERT_TRUE(fs::is_regular_file(loggerFile));

		fs::path newLogFile = currentPath / "another/dir/newLog.log";
		addOutputFileLog(newLogFile);

		logAllSeverities();
		ASSERT_TRUE(fs::exists(newLogFile));
		ASSERT_TRUE(fs::is_regular_file(newLogFile));
	}
} // namespace ball::common::utils::logger

// UNCLASSIFIED
