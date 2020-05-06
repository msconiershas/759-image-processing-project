#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include "include/img_load.h"
#include <string>
using namespace std;
using namespace cv;

void save_img(uchar4* img, string &output_file, size_t num_rows, size_t num_cols) {
  int sizes[2] = {num_rows, num_cols};

  Mat imageRGBA(2, sizes, CV_8UC4, (void *) img);
  Mat imageOutputBGR;
  cvtColor(imageRGBA, imageOutputBGR, 3);
  imwrite(output_file.c_str(), imageOutputBGR);

}

void load_img(string &filename, uchar4** image, size_t *num_rows, size_t *num_cols)
{

  Mat img = imread(filename.c_str(), 1);

  if(img.empty()) {
    printf("error loading image\n");
    exit(1);
  }

   Mat imageRGBA;

   cvtColor(img, imageRGBA, 2);

   *image = new uchar4[img.rows * img.cols];
   unsigned char *cvPtr = imageRGBA.ptr<unsigned char>(0);

   for(size_t i = 0; i < img.rows * img.cols; ++i){
     (*image)[i].x = cvPtr[4*i + 0];
     (*image)[i].y = cvPtr[4*i + 1];
     (*image)[i].z = cvPtr[4*i + 2];
     (*image)[i].w = cvPtr[4*i + 3];
   }	 
     *num_rows = img.rows;
     *num_cols = img.cols;

}

