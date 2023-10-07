#include "NumCpp/Utils.hpp"

#include "BindingsIncludes.hpp"

//================================================================================

void initUtils(pb11::module &m)
{
    // Utils.hpp
    m.def("num2str", &utils::num2str<double>);
    m.def("sqr", &utils::sqr<double>);
    m.def("cube", &utils::cube<double>);
    m.def("power", &utils::power<double>);
    m.def("power", &utils::power<ComplexDouble>);
    decltype(utils::powerf<double, double>(double{ 0 }, double{ 0 })) (*powerf_double)(double, double) =
        &utils::powerf<double, double>;
    m.def("powerf", powerf_double);
    decltype(utils::powerf<ComplexDouble, ComplexDouble>(ComplexDouble{ 0 }, ComplexDouble{ 0 })) (
        *powerf_complexDouble)(ComplexDouble, ComplexDouble) = &utils::powerf<ComplexDouble, ComplexDouble>;
    m.def("powerf_complex", powerf_complexDouble);

    m.def("num2str", &utils::num2str<int64>);
    m.def("sqr", &utils::sqr<int64>);
    m.def("cube", &utils::cube<int64>);
    m.def("power", &utils::power<int64>);
    decltype(utils::powerf<int64, double>(int64{ 0 }, double{ 0 })) (*powerf_int64)(int64, double) =
        &utils::powerf<int64, double>;
    m.def("powerf", powerf_int64);

    m.def("timeit",
          [](uint32 numIterations, PrintResults printResults, std::function<void(int, int)> function, int v1, int v2)
          { utils::timeit<std::chrono::nanoseconds>(numIterations, printResults, function, v1, v2); });
    m.def("timeit",
          [](uint32 numIterations, PrintResults printResults, std::function<int(int, int)> function, int v1, int v2)
          { utils::timeit<std::chrono::nanoseconds>(numIterations, printResults, function, v1, v2); });
}