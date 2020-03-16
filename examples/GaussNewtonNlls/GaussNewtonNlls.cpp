#include "NumCpp.hpp"

#include <cstdlib>
#include <iostream>

using FunctionType = std::function<double(const nc::NdArray<double>&, const nc::NdArray<double>&)>;

void wikipediaExample()
{
    // https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm

    // In a biology experiment studying the relation between substrate concentration [S] and reaction rate in
    // an enzyme-mediated reaction, the data in the following table were obtained.
    nc::NdArray<double> sMeasured = { 0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.74 };
    nc::NdArray<double> rateMeasured = { 0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317 };

    // It is desired to find a curve (model function) of the form
    FunctionType function = [](const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas) -> double
    {
        const double s = coordinates.at(0);
        const double beta1 = betas.at(0);
        const double beta2 = betas.at(1);

        return (beta1 * s) / (beta2 + s);
    };

    // partial derivative of function with respect to beta1
    FunctionType delFdelBeta1 = [](const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas) -> double
    {
        const double s = coordinates.at(0);
        const double beta2 = betas.at(1);

        return s / (beta2 + s);
    };

    // partial derivative of function with respect to beta2
    FunctionType delFdelBeta2 = [](const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas) -> double
    {
        const double s = coordinates.at(0);
        const double beta1 = betas.at(0);
        const double beta2 = betas.at(1);

        return -(beta1 * s) / nc::square(beta2 + s);
    };

    // starting with the initial estimates of beta1Guess and beta2Guess and calculating after 5 iterations
    const nc::uint32 numIterations = 5;
    const double beta1Guess = 0.9;
    const double beta2Guess = 0.2;

    auto [betas, rms] = nc::linalg::gaussNewtonNlls(numIterations, sMeasured.transpose(), rateMeasured,
        function, {delFdelBeta1, delFdelBeta2}, beta1Guess, beta2Guess);

    std::cout << "==========Wikipedia Example==========\n";
    std::cout << "beta values = " << betas;
    std::cout << "RMS = " << rms << '\n';
}

void exponentialExample()
{
    // United States population (in millions) and the corresponding year:
    nc::NdArray<double> year = nc::arange<double>(1.0, 9.0); // just use time points rather than the year
    nc::NdArray<double> population = { 8.3, 11.0, 14.7, 19.7, 26.7, 35.2, 44.4, 55.9 };

    // It is desired to find a curve (model function) of the form
    FunctionType exponentialFunction = [](const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas) -> double
    {
        const double t = coordinates.at(0);
        const double beta1 = betas.at(0);
        const double beta2 = betas.at(1);

        return beta1 * nc::exp(beta2 * t);
    };

    // partial derivative of function with respect to beta1
    FunctionType delFdelBeta1 = [](const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas) -> double
    {
        const double t = coordinates.at(0);
        const double beta2 = betas.at(1);

        return nc::exp(beta2 * t);
    };

    // partial derivative of function with respect to beta2
    FunctionType delFdelBeta2 = [](const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas) -> double
    {
        const double t = coordinates.at(0);
        const double beta1 = betas.at(0);
        const double beta2 = betas.at(1);

        return beta1 * t * nc::exp(beta2 * t);
    };

    // starting with the initial estimates of beta1Guess and beta2Guess and calculating after 5 iterations
    const nc::uint32 numIterations = 5;
    const double beta1Guess = 6.0;
    const double beta2Guess = 0.3;

    auto [betas, rms] = nc::linalg::gaussNewtonNlls(numIterations, year.transpose(), population,
        exponentialFunction, {delFdelBeta1, delFdelBeta2}, beta1Guess, beta2Guess);

    std::cout << "==========Exponential Population Example==========\n";
    std::cout << "beta values = " << betas;
    std::cout << "RMS = " << rms << '\n';
}

void sinusoidalExample()
{
    // Average monthly high temperatures for Baton Rouge, LA:
    nc::NdArray<double> month = nc::arange<double>(1.0, 13.0);
    nc::NdArray<double> temperature = { 61.0, 65.0, 72.0, 78.0, 85.0, 90.0, 92.0, 92.0, 88.0, 81.0, 72.0, 63.0 };

    // It is desired to find a curve (model function) of the form
    FunctionType sinusodialFunction = [](const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas) -> double
    {
        const double t = coordinates.at(0);
        const double beta1 = betas.at(0);
        const double beta2 = betas.at(1);
        const double beta3 = betas.at(2);
        const double beta4 = betas.at(3);

        return beta1 * nc::sin(beta2 * t + beta3) + beta4;
    };

    // partial derivative of function with respect to beta1
    FunctionType delFdelBeta1 = [](const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas) -> double
    {
        const double t = coordinates.at(0);
        const double beta2 = betas.at(1);
        const double beta3 = betas.at(2);

        return nc::sin(beta2 * t + beta3);
    };

    // partial derivative of function with respect to beta2
    FunctionType delFdelBeta2 = [](const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas) -> double
    {
        const double t = coordinates.at(0);
        const double beta1 = betas.at(0);
        const double beta2 = betas.at(1);
        const double beta3 = betas.at(2);

        return beta1 * t * nc::cos(beta2 * t + beta3);
    };

    // partial derivative of function with respect to beta3
    FunctionType delFdelBeta3 = [](const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas) -> double
    {
        const double t = coordinates.at(0);
        const double beta1 = betas.at(0);
        const double beta2 = betas.at(1);
        const double beta3 = betas.at(2);

        return beta1 * nc::cos(beta2 * t + beta3);
    };

    // partial derivative of function with respect to beta4
    FunctionType delFdelBeta4 = [](const nc::NdArray<double>& /*coordinates*/, const nc::NdArray<double>& /*betas*/) -> double
    {
        return 1.0;
    };

    // starting with the initial estimates and calculating after 5 iterations
    const nc::uint32 numIterations = 5;
    const double beta1Guess = 17.0;
    const double beta2Guess = 0.5;
    const double beta3Guess = 10.5;
    const double beta4Guess = 77.0;

    auto [betas, rms] = nc::linalg::gaussNewtonNlls(numIterations, month.transpose(), temperature,
        sinusodialFunction, {delFdelBeta1, delFdelBeta2, delFdelBeta3, delFdelBeta4},
        beta1Guess, beta2Guess, beta3Guess, beta4Guess);

    std::cout << "==========Sinusodial Temperature Example==========\n";
    std::cout << "beta values = " << betas;
    std::cout << "RMS = " << rms << '\n';
}

// https://en.wikipedia.org/wiki/Gaussian_function
double baseGaussianFunction(const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas)
{
    const auto x = coordinates.at(0);
    const auto y = coordinates.at(1);

    const auto a = betas.at(0);
    const auto x0 = betas.at(2);
    const auto y0 = betas.at(3);
    const auto sigmaX = betas.at(4);
    const auto sigmaY = betas.at(5);

    return a * nc::exp(-(nc::square(x - x0) / (2.0 * nc::square(sigmaX)) + nc::square(y - y0) / (2.0 * nc::square(sigmaY))));
};

// It is desired to find a curve (model function) of the form
double gaussianFunction(const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas)
{
    const auto dcOffset = betas.at(1);
    return baseGaussianFunction(coordinates, betas) + dcOffset;
};

void twoDimensionalGaussianExample()
{
    // create some data points to describe a two-dimensional gaussian function
    auto coords = nc::arange<double>(-3.0, 3.1, 0.1);
    auto [y, x] = nc::meshgrid<double>(coords, coords);
    auto coordinates = nc::vstack({x.flatten(), y.flatten()}).transpose();

    // randomize some truth values for the model parameters
    nc::random::seed(666); // set the random seed for repeatability

    // make some random "truth" parameters for generating data
    const auto aTruth = nc::random::randFloat<double>(900.0, 1100.0);
    const auto dcOffsetTruth = nc::random::randFloat<double>(90.0, 100.0);
    const auto x0Truth = nc::random::randFloat<double>(-0.5, 0.5);
    const auto y0Truth = nc::random::randFloat<double>(-0.5, 0.5);
    const auto sigmaXTruth = nc::random::randFloat<double>(0.9, 1.1);
    const auto sigmaYTruth = nc::random::randFloat<double>(0.9, 1.1);

    nc::NdArray<double> betasTruth = { aTruth, dcOffsetTruth, x0Truth, y0Truth, sigmaXTruth, sigmaYTruth };

    // make some "measurements"
    nc::NdArray<double> measurements(1, coordinates.shape().rows);
    const auto cSlice = coordinates.cSlice();
    for (nc::uint32 i = 0; i < measurements.size(); ++i)
    {
        measurements[i] = gaussianFunction(coordinates(i, cSlice), betasTruth);
    }

    // add some noise to the measurements
    auto noise = nc::random::randN<double>(measurements.shape()) * 10.0; // STD = 10 of noise
    measurements += noise;

    // partial derivative of gaussianFunction with respect to a
    FunctionType delFdelA = [](const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas) -> double
    {
        const auto a = betas.at(0);
        return baseGaussianFunction(coordinates, betas) / a;
    };

    // partial derivative of gaussianFunction with respect to dcOffset
    FunctionType delFdelDcOffset = [](const nc::NdArray<double>& /*coordinates*/, const nc::NdArray<double>& /*betas*/) -> double
    {
        return 1.0;
    };

    // partial derivative of gaussianFunction with respect to x0
    FunctionType delFdelX0 = [](const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas) -> double
    {
        const auto x = coordinates.at(0);

        const auto x0 = betas.at(2);
        const auto sigmaX = betas.at(4);

        return (x - x0) / nc::square(sigmaX) * baseGaussianFunction(coordinates, betas);
    };

    // partial derivative of gaussianFunction with respect to y0
    FunctionType delFdelY0 = [](const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas) -> double
    {
        const auto y = coordinates.at(1);

        const auto y0 = betas.at(3);
        const auto sigmaY = betas.at(5);

        return (y - y0) / nc::square(sigmaY) * baseGaussianFunction(coordinates, betas);
    };

    // partial derivative of gaussianFunction with respect to sigmaX
    FunctionType delFdelSigmaX = [](const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas) -> double
    {
        const auto x = coordinates.at(0);

        const auto x0 = betas.at(2);
        const auto sigmaX = betas.at(4);

        return (nc::square(x - x0) / nc::power(sigmaX, 3)) * baseGaussianFunction(coordinates, betas);
    };

    // partial derivative of gaussianFunction with respect to sigmaY
    FunctionType delFdelSigmaY = [](const nc::NdArray<double>& coordinates, const nc::NdArray<double>& betas) -> double
    {
        const auto y = coordinates.at(1);

        const auto y0 = betas.at(2);
        const auto sigmaY = betas.at(4);

        return (nc::square(y - y0) / nc::power(sigmaY, 3)) * baseGaussianFunction(coordinates, betas);
    };

    // starting with the initial estimates of and calculating after 5 iterations
    const nc::uint32 numIterations = 5;
    const double aGuess = aTruth + nc::random::randN<double>() * 5.0;
    const double dcOffsetGuess = dcOffsetTruth + nc::random::randN<double>() * 5.0;
    const double x0Guess = x0Truth + nc::random::randN<double>() * 0.2;
    const double y0Guess = y0Truth + nc::random::randN<double>() * 0.2;
    const double sigmaXGuess = sigmaXTruth + nc::random::randN<double>() * 0.2;
    const double sigmaYGuess = sigmaYTruth + nc::random::randN<double>() * 0.2;

    auto [betas, rms] = nc::linalg::gaussNewtonNlls(numIterations, coordinates, measurements,
        FunctionType(gaussianFunction), {delFdelA, delFdelDcOffset, delFdelX0, delFdelY0, delFdelSigmaX, delFdelSigmaY},
        aGuess, dcOffsetGuess, x0Guess, y0Guess, sigmaXGuess, sigmaYGuess);

    nc::NdArray<double> initialGuess = { aGuess, dcOffsetGuess, x0Guess, y0Guess, sigmaXGuess, sigmaYGuess };

    std::cout << "==========Sinusodial Temperature Example==========\n";
    std::cout << "truth values  = " << betasTruth;
    std::cout << "initial guess = " << initialGuess;
    std::cout << "beta values   = " << betas;
    std::cout << "RMS = " << rms << '\n';
}

int main()
{
    wikipediaExample();
    exponentialExample();
    sinusoidalExample();
    twoDimensionalGaussianExample();

    return EXIT_SUCCESS;
}
