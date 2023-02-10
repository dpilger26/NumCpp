#pragma once

#include <chrono>
#include <iostream>

namespace nc
{
    /**
     * @brief Clock Type
     */
    using Clock = std::chrono::system_clock;

    /**
     * @brief Duration Type
     */
    using Duration = std::chrono::nanoseconds;

    /**
     * @brief TimePoint Type
     */
    using TimePoint = std::chrono::time_point<Clock, Duration>;

    /**
     * @brief Output stream operator for the Duration type
     *
     * @param os: the output stream
     * @param duration: the Duration
     * @returns std::ostream
     */
    inline std::ostream& operator<<(std::ostream& os, Duration duration)
    {
        os << duration.count() << " nanoseconds";
        return os;
    }

    /**
     * @brief Output stream operator for the TimePoint type
     *
     * @param os: the output stream
     * @param timepoint: the TimePoint
     * @returns std::ostream
     */
    inline std::ostream& operator<<(std::ostream& os, const TimePoint& timepoint)
    {
        os << timepoint.time_since_epoch() << " nanoseconds since epoch";
        return os;
    }

} // namespace nc
