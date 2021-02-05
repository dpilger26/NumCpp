#include "function.h"

nc::NdArray<double> getRandomArray(nc::uint32 numRows, nc::uint32 numCols)
{
    return nc::random::rand<double>({numRows, numCols});
}