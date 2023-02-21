/// @file
/// @author David Pilger <dpilger26@gmail.com>
/// [GitHub Repository](https://github.com/dpilger26/NumCpp)
///
/// License
/// Copyright 2018-2023 David Pilger
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
/// Function profiling/timing
///
#pragma once

#include <chrono>
#include <iostream>
#include <type_traits>

#include "NumCpp/Core/Timer.hpp"
#include "NumCpp/Core/Types.hpp"

namespace nc::utils
{
    namespace timeit_detail
    {
        /// @brief  Result statistics of a timeit run
        template<typename TimeUnit>
        struct Result
        {
            TimeUnit min{};
            TimeUnit max{};
            TimeUnit mean{};
        };

        template<typename TimeUnit>
        std::ostream& operator<<(std::ostream& os, const Result<TimeUnit> result)
        {
            std::string unit{};
            if constexpr (std::is_same<TimeUnit, std::chrono::hours>::value)
            {
                unit = " hours";
            }
            else if constexpr (std::is_same<TimeUnit, std::chrono::minutes>::value)
            {
                unit = " minutes";
            }
            else if constexpr (std::is_same<TimeUnit, std::chrono::seconds>::value)
            {
                unit = " seconds";
            }
            else if constexpr (std::is_same<TimeUnit, std::chrono::milliseconds>::value)
            {
                unit = " milliseconds";
            }
            else if constexpr (std::is_same<TimeUnit, std::chrono::microseconds>::value)
            {
                unit = " microseconds";
            }
            else if constexpr (std::is_same<TimeUnit, std::chrono::nanoseconds>::value)
            {
                unit = " nanoseconds";
            }
            else
            {
                unit = " time units of some sort";
            }

            os << "Timeit results:\n";
            os << "\tmin: " << result.min.count() << unit << "\n";
            os << "\tmax: " << result.max.count() << unit << "\n";
            os << "\tmean: " << result.mean.count() << unit << "\n";

            return os;
        }
    } // namespace timeit_detail

    //============================================================================
    /// Timing of a function
    ///
    /// @param function: the function to time
    /// @param numItererations: number of iterations for the timing statistics
    /// @param args: the arguements that are forwarded to the function input
    ///
    /// @return timing statistics
    ///
    template<typename TimeUnit, typename Function, typename... Args>
    timeit_detail::Result<TimeUnit>
        timeit(uint32 numIterations, bool printResults, Function function, Args&&... args) noexcept
    {
        auto result = timeit_detail::Result<TimeUnit>{};
        auto timer  = Timer<TimeUnit>{};

        for (uint32 i = 0; i < numIterations; ++i)
        {
            if (i == 0)
            {
                result.min = TimeUnit::max();
            }

            timer.tic();

            using ResultType = std::invoke_result_t<Function, Args...>;
            if constexpr (std::is_same_v<ResultType, void>)
            {
                function(std::forward<Args>(args)...);
            }
            else
            {
                [[maybe_unused]] const ResultType functionResult = function(std::forward<Args&>(args)...);
            }

            const auto elapsedTime = timer.toc(false);

            result.mean = result.mean + elapsedTime;
            result.min  = std::min(result.min, elapsedTime);
            result.max  = std::max(result.max, elapsedTime);
        }

        result.mean = result.mean / numIterations;

        if (printResults)
        {
            std::cout << result;
        }

        return result;
    }
} // namespace nc::utils
