#include "NumCpp/Roots.hpp"

#include "BindingsIncludes.hpp"

#include "NumCpp/Polynomial/Poly1d.hpp"

//================================================================================

namespace RootsInterface
{
    constexpr double EPSILON = 1e-10;

    //================================================================================

    double bisection(const polynomial::Poly1d<double>& p, double a, double b)
    {
        auto rootFinder = roots::Bisection(EPSILON, p);
        return rootFinder.solve(a, b);
    }

    //================================================================================

    double brent(const polynomial::Poly1d<double>& p, double a, double b)
    {
        auto rootFinder = roots::Brent(EPSILON, p);
        return rootFinder.solve(a, b);
    }

    //================================================================================

    double dekker(const polynomial::Poly1d<double>& p, double a, double b)
    {
        auto rootFinder = roots::Dekker(EPSILON, p);
        return rootFinder.solve(a, b);
    }

    //================================================================================

    double newton(const polynomial::Poly1d<double>& p, double x)
    {
        auto pPrime     = p.deriv();
        auto rootFinder = roots::Newton(EPSILON, p, pPrime);
        return rootFinder.solve(x);
    }

    //================================================================================

    double secant(const polynomial::Poly1d<double>& p, double a, double b)
    {
        auto rootFinder = roots::Secant(EPSILON, p);
        return rootFinder.solve(a, b);
    }
} // namespace RootsInterface

//================================================================================

void initRoots(python_interface::module& m)
{
    // Roots.hpp
    m.def("bisection_roots", &RootsInterface::bisection);
    m.def("brent_roots", &RootsInterface::brent);
    m.def("dekker_roots", &RootsInterface::dekker);
    m.def("newton_roots", &RootsInterface::newton);
    m.def("secant_roots", &RootsInterface::secant);
}