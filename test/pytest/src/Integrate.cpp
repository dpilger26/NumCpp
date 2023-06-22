#include "NumCpp/Integrate.hpp"

#include "BindingsIncludes.hpp"

#include "NumCpp/Polynomial/Poly1d.hpp"

//================================================================================

namespace IntegrateInterface
{
    constexpr uint32 NUM_ITERATIONS   = 100;
    constexpr uint32 NUM_SUBDIVISIONS = 10000;

    //================================================================================

    double gauss_legendre(const polynomial::Poly1d<double>& p, double a, double b)
    {
        return integrate::gauss_legendre(a, b, NUM_ITERATIONS, p);
    }

    //================================================================================

    double romberg(const polynomial::Poly1d<double>& p, double a, double b)
    {
        return integrate::romberg(a, b, 10, p);
    }

    //================================================================================

    double simpson(const polynomial::Poly1d<double>& p, double a, double b)
    {
        return integrate::simpson(a, b, NUM_SUBDIVISIONS, p);
    }

    //================================================================================

    double trapazoidal(const polynomial::Poly1d<double>& p, double a, double b)
    {
        return integrate::trapazoidal(a, b, NUM_SUBDIVISIONS, p);
    }
} // namespace IntegrateInterface

//================================================================================

void initIntegrate(pb11::module& m)
{
    // Integrate.hpp
    m.def("integrate_gauss_legendre", &IntegrateInterface::gauss_legendre);
    m.def("integrate_romberg", &IntegrateInterface::romberg);
    m.def("integrate_simpson", &IntegrateInterface::simpson);
    m.def("integrate_trapazoidal", &IntegrateInterface::trapazoidal);
}