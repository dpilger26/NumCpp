#include "NumCpp/Core/Constants.hpp"

#include "BindingsIncludes.hpp"

#include <version>

#include "NumCpp/Core/Internal/Version.hpp"

//================================================================================

void initConstants(pb11::module &m)
{
#ifdef NUMCPP_NO_USE_BOOST
    m.attr("NUMCPP_NO_USE_BOOST") = true;
#else
    m.attr("NUMCPP_NO_USE_BOOST")        = false;
#endif

#ifdef __cpp_lib_gcd_lcm
    m.attr("STL_GCD_LCM") = true;
#else
    m.attr("STL_GCD_LCM")                = false;
#endif

#ifdef __cpp_lib_clamp
    m.attr("STL_CLAMP") = true;
#else
    m.attr("STL_CLAMP")                  = false;
#endif

#ifdef __cpp_lib_hypot
    m.attr("STL_HYPOT") = true;
#else
    m.attr("STL_HYPOT")                  = false;
#endif

#ifdef __cpp_lib_math_special_functions
    m.attr("STL_SPECIAL_FUNCTIONS") = true;
#else
    m.attr("STL_SPECIAL_FUNCTIONS")      = false;
#endif

#ifdef __cpp_lib_execution
    m.attr("STL_LIB_EXECUTION") = true;
#else
    m.attr("STL_LIB_EXECUTION")          = false;
#endif

#ifdef __cpp_lib_parallel_algorithm
    m.attr("STL_LIB_PARALLEL_ALGORITHM") = true;
#else
    m.attr("STL_LIB_PARALLEL_ALGORITHM") = false;
#endif

    // Version.hpp
    m.attr("VERSION") = VERSION;

    // Constants.hpp
    m.attr("c")                       = constants::c;
    m.attr("e")                       = constants::e;
    m.attr("inf")                     = constants::inf;
    m.attr("pi")                      = constants::pi;
    m.attr("nan")                     = constants::nan;
    m.attr("j")                       = constants::j;
    m.attr("DAYS_PER_WEEK")           = constants::DAYS_PER_WEEK;
    m.attr("MINUTES_PER_HOUR")        = constants::MINUTES_PER_HOUR;
    m.attr("SECONDS_PER_MINUTE")      = constants::SECONDS_PER_MINUTE;
    m.attr("MILLISECONDS_PER_SECOND") = constants::MILLISECONDS_PER_SECOND;
    m.attr("SECONDS_PER_HOUR")        = constants::SECONDS_PER_HOUR;
    m.attr("HOURS_PER_DAY")           = constants::HOURS_PER_DAY;
    m.attr("MINUTES_PER_DAY")         = constants::MINUTES_PER_DAY;
    m.attr("SECONDS_PER_DAY")         = constants::SECONDS_PER_DAY;
    m.attr("MILLISECONDS_PER_DAY")    = constants::MILLISECONDS_PER_DAY;
    m.attr("SECONDS_PER_WEEK")        = constants::SECONDS_PER_WEEK;
}
