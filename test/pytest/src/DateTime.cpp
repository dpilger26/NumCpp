#ifndef NUMCPP_NO_USE_BOOST
#include "NumCpp/DateTime.hpp"

#include "BindingsIncludes.hpp"

#include <string>

//================================================================================

void initDateTime(python_interface::module& m)
{
    // Clock
    python_interface::class_<Clock>(m, "Clock")
        .def(python_interface::init<>())
        .def_static("now", []() { return std::chrono::time_point_cast<TimePoint::duration>(Clock::now()); });

    // Duration
    python_interface::class_<Duration>(m, "Duration")
        .def(python_interface::init<>())
        .def(python_interface::init<std::uint64_t>())
        .def("count", &Duration::count)
        .def("min", &Duration::min)
        .def("max", &Duration::max)
        .def("zero", &Duration::zero)
        .def("__eq__", [](const Duration& lhs, const Duration& rhs) { return lhs == rhs; })
        .def("__ne__", [](const Duration& lhs, const Duration& rhs) { return lhs != rhs; })
        .def("__lt__", [](const Duration& lhs, const Duration& rhs) { return lhs < rhs; })
        .def("__le__", [](const Duration& lhs, const Duration& rhs) { return lhs <= rhs; })
        .def("__gt__", [](const Duration& lhs, const Duration& rhs) { return lhs > rhs; })
        .def("__ge__", [](const Duration& lhs, const Duration& rhs) { return lhs >= rhs; })
        .def("__sub__", [](const Duration& lhs, const Duration& rhs) { return lhs - rhs; });

    // TimePoint
    python_interface::class_<TimePoint>(m, "TimePoint")
        .def(python_interface::init<>())
        .def(python_interface::init<Duration>())
        .def("time_since_epoch", &TimePoint::time_since_epoch)
        .def_static("min", &TimePoint::min)
        .def_static("max", &TimePoint::max);

    // DateTime
    python_interface::class_<DateTime>(m, "DateTime")
        .def(python_interface::init<>())
        .def(python_interface::init<TimePoint>())
        .def(python_interface::init<std::string>())
        .def(python_interface::init<int, int, int, int, int, int, double>())
        .def_property("year", &DateTime::year, &DateTime::setYear)
        .def_property("month", &DateTime::month, &DateTime::setMonth)
        .def_property("day", &DateTime::day, &DateTime::setDay)
        .def_property("hour", &DateTime::hour, &DateTime::setHour)
        .def_property("minute", &DateTime::minute, &DateTime::setMinute)
        .def_property("second", &DateTime::second, &DateTime::setSecond)
        .def_property("fractionalSecond", &DateTime::fractionalSecond, &DateTime::setFractionalSecond)
        .def("toTimePoint", &DateTime::toTimePoint)
        .def("toStr", &DateTime::toStr)
        .def("__eq__", [](const DateTime& lhs, const DateTime& rhs) { return lhs == rhs; })
        .def("__ne__", [](const DateTime& lhs, const DateTime& rhs) { return lhs != rhs; })
        .def("__lt__", [](const DateTime& lhs, const DateTime& rhs) { return lhs < rhs; })
        .def("__le__", [](const DateTime& lhs, const DateTime& rhs) { return lhs <= rhs; })
        .def("__gt__", [](const DateTime& lhs, const DateTime& rhs) { return lhs > rhs; })
        .def("__ge__", [](const DateTime& lhs, const DateTime& rhs) { return lhs >= rhs; })
        .def("__sub__", [](const DateTime& lhs, const DateTime& rhs) { return lhs - rhs; })
        .def_static("now", &DateTime::now)
        .def_static("strToTimepoint", &DateTime::strToTimepoint);
}
#endif
