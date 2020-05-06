#include<stdlib.h>

#include "include/img_load.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>     
using namespace std;

unsigned char *d_red, *d_green, *d_blue;
uchar4 *g_inputImg;
uchar4 *g_outputImg;
float *d_filter;


__global__ void sobel_filter(const unsigned char* img, unsigned char* img_new, const unsigned int num_rows, const unsigned int num_cols) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  float dx, dy;
  
  if( x > 0 && y > 0 && x < num_rows - 1 && y < height-1) {
    dx = (-1* img[(y-1)* num_rows + (x-1)]) + (-2*img[y*num_rows+(x-1)]) + (-1*img[(y+1)*num_rows+(x-1)]) + (img[(y-1)*num_rows + (x+1)]) + (2*img[y*num_rows+(x+1)]) + (img[(y+1)*num_rows+(x+1)]);
    dy = (img[(y-1)*num_rows + (x-1)]) + ( 2*img[(y-1)*num_rows+x]) + (   img[(y-1)*num_rows+(x+1)]) + (-1* img[(y+1)*num_rows + (x-1)]) + (-2*img[(y+1)*num_rows+x]) + (-1*img[(y+1)*num_rows+(x+1)]);
    img_new[y*num_rows + x] = sqrt( (dx*dx) + (dy*dy) );
  }
}
__global__
void combine_channels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const img_out,
                       int numRows,
                       int numCols)
{
  
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_idx = thread_2D_pos.y * numCols + thread_2D_pos.x;

  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_idx];
  unsigned char green = greenChannel[thread_idx];
  unsigned char blue  = blueChannel[thread_idx];

  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  img_out[thread_idx] = outputPixel;
}

__global__
void separate_channels(const uchar4* const s_input_img,
                      int num_rows,
                      int num_cols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{  
  
  int loc_x = blockDim.x * blockIdx.x + threadIdx.x;
  int loc_y = blockDim.y * blockIdx.y + threadIdx.y;

  if ( loc_x >= num_cols ||
      loc_y >= num_rows )
  {
       return;
  }
  
  int id = loc_y * num_cols + loc_x;

  alpha_channel[id] = s_input_img[id].wg;
 }

uchar4* edge_detect(uchar4* img, size_t num_rows, size_t num_cols) {
  
   const dim3 blockSize(16,16,1);
  
   //  alculate Grid SIze
  int a=num_cols/blockSize.x, b=num_rows/blockSize.y;	
  const dim3 gridSize(a+1,b+1,1);
  const size_t num_pix = num_rows * num_cols;

  uchar4 *output_img;
  cudaMalloc((void **)&output_img, sizeof(uchar4) * num_pix);
    
  g_inputImg  = input_img;
  g_outputImg = output_img;

  unsigned char *d_red_blur, *d_green_blur, *d_blue_blur;
  cudaMalloc(&d_red_blur,    sizeof(unsigned char) * num_pix);
  cudaMalloc(&d_green_blur,  sizeof(unsigned char) * num_pix);
  cudaMalloc(&d_blue_blur,   sizeof(unsigned char) * num_pix);
  
  cudaMalloc(&d_red, sizeof(unsigned char*) * num_rows * num_cols);
  cudaMalloc(&d_green, sizeof(unsigned char*) * num_rows * num_cols);
  cudaMalloc(&d_blue, sizeof(unsigned char*) * num_rows * num_cols);
  
  cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth);
  cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice); 

  separate_channels<<<gridSize, blockSize>>>(input_img, num_rows, num_cols, d_red,d_green, d_blue);

  cudaDeviceSynchronize(); 
  
  //blur each color
  sobel_filter<<<gridSize, blockSize>>>(d_red, d_red_blur, num_rows, num_cols,  d_filter, filterWidth);
  sobel_filter<<<gridSize, blockSize>>>(d_green, d_green_blur, num_rows, num_cols,  d_filter, filterWidth);
  sobel_filter<<<gridSize, blockSize>>>(d_blue, d_blue_blur, num_rows, num_cols,  d_filter, filterWidth);

  cudaDeviceSynchronize(); 

  combine_channels<<<gridSize, blockSize>>>(d_red_blur, d_green_blur, d_blue_blur, output_img, num_rows, num_cols);
  cudaDeviceSynchronize(); 

  cudaFree(d_red);
  cudaFree(d_green);
  cudaFree(d_blue);
  cudaFree(d_filter);
  cudaFree(d_red_blur);
  cudaFree(d_green_blur);
  cudaFree(d_blue_blur);

  cudaDeviceSynchronize();  


  uchar4* h_out;
  h_out = (uchar4*)malloc(sizeof(uchar4) * num_pix);

  
  cudaMemcpy(h_out, output_img, sizeof(uchar4) * num_pix, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize(); 
  
  cudaFree(g_inputImg);
  cudaFree(g_outputImg);
  
  return h_out;

}


