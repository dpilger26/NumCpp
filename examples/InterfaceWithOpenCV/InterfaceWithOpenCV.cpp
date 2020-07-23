#include "NumCpp.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <numeric>

constexpr nc::uint32 NUM_ROWS = 512;
constexpr nc::uint32 NUM_COLS = 512;

int main()
{
    // create a ramp image with NumCpp
    constexpr auto numHalfCols = NUM_COLS / 2; // integer division
    auto ncArray = nc::NdArray<nc::uint8>(NUM_ROWS, NUM_COLS);
    for (nc::uint32 row = 0; row < ncArray.numRows(); ++row)
    {
        auto begin = ncArray.begin(row);
        std::iota(begin, begin + numHalfCols, nc::uint8{0});

        auto rbegin = ncArray.rbegin(row);
        std::iota(rbegin, rbegin + numHalfCols, nc::uint8{0});
    }

    // convert to OpenCV Mat
    auto cvArray = cv::Mat(ncArray.numRows(), ncArray.numCols(), CV_8SC1, ncArray.data());

    // display the OpenCV Mat
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Display window", cvArray); // Show our image inside it.
    cv::waitKey(0); // Wait for a keystroke in the window

    // tranpose the Mat with OpenCV
    auto transposedCvArray = cv::Mat(cvArray.cols, cvArray.rows, CV_8SC1);
    cv::transpose(cvArray, transposedCvArray);

    // display the transposed Mat
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Display window", transposedCvArray); // Show our image inside it.
    cv::waitKey(0); // Wait for a keystroke in the window

    // convert the transposed OpenCV Mat to a NumCpp array
    auto transposedNcArray = nc::NdArray<nc::uint8>(transposedCvArray.data, transposedCvArray.rows, transposedCvArray.cols);

    // make sure the two transposed arrays are the same
    if (nc::array_equal(transposedNcArray, ncArray.transpose()))
    {
        std::cout << "Arrays are equal.\n";
    }
    else
    {
        std::cout << "Arrays are not equal.\n";
    }

    return 0;
}
