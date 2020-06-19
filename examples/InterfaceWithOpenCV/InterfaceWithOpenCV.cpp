#include "NumCpp.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

int main()
{
    // create a random image with NumCpp
    auto ncArray = nc::random::randInt<nc::uint8>({ 500, 500 }, 0, nc::DtypeInfo<nc::uint8>::max());

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
