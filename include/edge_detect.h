
#ifndef EDGEDETECT_H__
#define EDGEDETECT_H__

#include <string>
#include <cuda_runtime.h> 

uchar4* edge_detect(uchar4* img, size_t num_rows, size_t num_cols);

#endif
