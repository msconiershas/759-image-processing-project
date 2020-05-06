#ifndef LOADSAVEIMAGE_H__
#define LOADSAVEIMAGE_H__

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string>
#include <stdio.h>

void load_img(std::string &filename, uchar4 **imagePtr, size_t *numRows, size_t *numCols);
void save_img(uchar4* image, std::string &output_filename, size_t numRows, size_t numCols);


#endif
