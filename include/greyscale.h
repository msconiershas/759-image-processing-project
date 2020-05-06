
#ifndef GREYSCALE_H__
#define GREYSCALE_H__

#include <string>
#include <cuda_runtime.h> 

uchar4* greyscale(uchar4* img, size_t num_rows, size_t num_cols);

#endif
