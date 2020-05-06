//#define _glibcxx_use_cxx11_abi 0
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuda.h>

#include "include/blur.h"
#include "include/greyscale.h"
#include "include/edge_detect.h"
#include "include/img_load.h"
#include <string>
using namespace std;

size_t num_rows, num_cols;

int main(int argc, char **argv) {
 
 if(argc < 3) {
    printf("Usage: ./image_proc <image_file> <filter>");
    exit(1);
  } 
  uchar4* h_out = NULL;
  uchar4* h_image, *d_in;
  string arg = string(argv[2]);
  string file = string(argv[1]);
  string output_file = "output.jpg";
  
  load_img(file, &h_image, &num_rows, &num_cols);
  
  cudaMalloc((void **) &d_in, num_rows * num_cols * sizeof(uchar4));
  cudaMemcpy(d_in, h_image, num_rows * num_cols * sizeof(uchar4), cudaMemcpyHostToDevice);
   

  if(arg == "-b" || arg == "-blur") {
    
    h_out = blur(d_in, num_rows, num_cols, 9);
  }
  else if(arg == "-g" || arg == "-grey" || arg == "-gray") {
    h_out = greyscale(d_in, num_rows, num_cols);
  }
  else if(arg == "-e" || arg == "-edge") {
    h_out = edge_detect(d_in, num_rows, num_cols);
  }
  else {
    printf("INCORRECT ARGUMENTS\n");
    exit(1);
  } 
    free(h_image);

    save_img(h_out, output_file, num_rows, num_cols);
    cudaFree(d_in);
}



