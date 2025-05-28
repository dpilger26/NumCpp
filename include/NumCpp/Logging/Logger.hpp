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
/// Text file loggger
///
#pragma once

#ifndef NUMCPP_NO_USE_BOOST

#include <filesystem>
#include <fstream>
#include <ostream>
#include <sstream>
#include <string>

#include "boost/core/null_deleter.hpp"
#include "boost/log/attributes.hpp"
#include "boost/log/core/core.hpp"
#include "boost/log/expressions.hpp"
#include "boost/log/expressions/formatters/date_time.hpp"
#include "boost/log/sinks/sync_frontend.hpp"
#include "boost/log/sinks/text_ostream_backend.hpp"
#include "boost/log/sources/global_logger_storage.hpp"
#include "boost/log/sources/severity_logger.hpp"
#include "boost/log/support/date_time.hpp"
#include "boost/log/trivial.hpp"
#include "boost/log/utility/manipulators/add_value.hpp"
#include "boost/log/utility/setup/common_attributes.hpp"
#include "boost/make_shared.hpp"
#include "boost/shared_ptr.hpp"

namespace nc::logger
{
    /**
     * @brief Register a global logger
     */
    BOOST_LOG_GLOBAL_LOGGER(fileLogger, boost::log::sources::severity_logger_mt<boost::log::trivial::severity_level>)

    /**
     * Boost log attributes
     */
    BOOST_LOG_ATTRIBUTE_KEYWORD(lineId, "LineID", uint32_t)
    BOOST_LOG_ATTRIBUTE_KEYWORD(timestamp, "TimeStamp", boost::posix_time::ptime)
    BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", boost::log::trivial::severity_level)
    BOOST_LOG_ATTRIBUTE_KEYWORD(filename, "Filename", std::string)
    BOOST_LOG_ATTRIBUTE_KEYWORD(functionName, "FunctionName", std::string)
    BOOST_LOG_ATTRIBUTE_KEYWORD(lineNumber, "LineNumber", uint32_t)

    namespace detail
    {
        // just log messages with severity >= SEVERITY_THRESHOLD are written
        constexpr boost::log::trivial::severity_level INIT_LOGLEVEL{ boost::log::trivial::severity_level::trace };

        // log file extension
        constexpr char OUTPUT_LOG_FILE_EXT[] = "log";

        // typedefs
        using text_sink = boost::log::sinks::synchronous_sink<boost::log::sinks::text_ostream_backend>;

        /**
         * @brief local variables to hold the sink pointers
         */
        inline boost::shared_ptr<text_sink> sinkConsole{};
        inline boost::shared_ptr<text_sink> sinkFile{};

        /**
         * @brief function to define the format of the output
         */
        [[nodiscard]] inline boost::log::formatter createOutputFormat()
        {
            // specify the format of the log message
            constexpr auto        counterWidth = 7;
            boost::log::formatter formatter =
                boost::log::expressions::stream
                << std::setw(counterWidth) << std::setfill('0') << lineId << std::setfill(' ') << " ["
                << boost::log::expressions::format_date_time(timestamp, "%Y-%m-%d %H:%M:%S.%f") << "] "
                << "[" << boost::log::trivial::severity << "] "
                << "[" << filename << ":" << functionName << "():" << lineNumber << "] "
                << boost::log::expressions::smessage;
            return formatter;
        }
    } // namespace detail

    /**
     * @brief Global intializer and constructor for the global logger
     *        Sets the initial sink to console backend and sets the default severity level
     */
    [[nodiscard]] inline BOOST_LOG_GLOBAL_LOGGER_INIT(fileLogger, boost::log::sources::severity_logger_mt)
    {
        boost::log::sources::severity_logger_mt<boost::log::trivial::severity_level> logger;

        // add attributes
        logger.add_attribute("LineID", boost::log::attributes::counter<uint32_t>(1));
        logger.add_attribute("TimeStamp", boost::log::attributes::local_clock());

        detail::sinkConsole = boost::make_shared<detail::text_sink>();

        // add "console" output stream to our sink
        detail::sinkConsole->locked_backend()->add_stream(
            boost::shared_ptr<std::ostream>(&std::clog, boost::null_deleter()));

        // specify the format of the log message
        const auto formatter = detail::createOutputFormat();

        // set the formatting
        detail::sinkConsole->set_formatter(formatter);

        // just log messages with severity >= INIT_LOGLEVEL are written
        detail::sinkConsole->set_filter(severity >= detail::INIT_LOGLEVEL);

        // "register" our sink
        boost::log::core::get()->add_sink(detail::sinkConsole);

        return logger;
    }

    /**
     * @brief Function to add the name of the output log file
     * @note This function will attempt to create any parent directories, if necessary
     * @throws std::runtime_error if it cannot create the appropiate directories
     *
     * @param logFileName  path of log file to write to
     */
    inline void addOutputFileLog(std::filesystem::path logFileName)
    {
        logFileName = std::filesystem::absolute(logFileName.replace_extension(detail::OUTPUT_LOG_FILE_EXT));

        // create the parent directories as needed
        const auto errorCode = [&]
        {
            auto error = std::error_code{};
            std::filesystem::create_directories(logFileName.parent_path(), error);
            return error;
        }();
        if (errorCode)
        {
            auto ss = std::stringstream{};
            ss << "Failed to create " << logFileName << " -- " << errorCode.message();
            throw std::runtime_error{ ss.str() };
        }

        // add a text sink
        detail::sinkFile = boost::make_shared<detail::text_sink>();

        // add a logfile stream to our sink
        detail::sinkFile->locked_backend()->add_stream(boost::make_shared<std::ofstream>(logFileName));

        // specify the format of the log message
        const auto formatter = detail::createOutputFormat();

        // set the formatting
        detail::sinkFile->set_formatter(formatter);

        // just log messages with severity >= INIT_LOGLEVEL are written
        detail::sinkFile->set_filter(severity >= detail::INIT_LOGLEVEL);

        // "register" our sink
        boost::log::core::get()->add_sink(detail::sinkFile);
    }

    /**
     * @brief Function to set the severity level to report back to console and log file
     *
     * @param level: level at which to report the logs
     */
    inline void setLogLevel(boost::log::trivial::severity_level level)
    {
        detail::sinkConsole->set_filter(severity >= level);
        if (detail::sinkFile)
        {
            detail::sinkFile->set_filter(severity >= level);
        }
    }
} // namespace nc::logger

// just a helper macro used by the macros below - don't use it in your code
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define BOOST_LOGGER(severity)                                                                           \
    BOOST_LOG_SEV(nc::logger::fileLogger::get(), boost::log::trivial::severity)                          \
        << boost::log::add_value("Filename", std::filesystem::path(__FILE__).filename().stem().string()) \
        << boost::log::add_value("FunctionName", __FUNCTION__)                                           \
        << boost::log::add_value("LineNumber", uint32_t{ __LINE__ })

// ===== log macros =====
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOG_TRACE BOOST_LOGGER(trace)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOG_DEBUG BOOST_LOGGER(debug)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOG_INFO BOOST_LOGGER(info)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOG_WARNING BOOST_LOGGER(warning)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOG_ERROR BOOST_LOGGER(error)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LOG_FATAL BOOST_LOGGER(fatal)

#endif // #ifndef NUMCPP_NO_USE_BOOST
