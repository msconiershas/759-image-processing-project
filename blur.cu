#include<stdlib.h>

#include "include/img_load.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>     
using namespace std;

unsigned char *d_red, *d_green, *d_blue;
float *d_filter;
uchar4 *d_input_img;
uchar4 *d_output_img;

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols, const float* const filter, const int filter_width)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if ( col >= numCols || row >= numRows )
  {
   return;
  }

  float result = 0.f;
    for (int filter_r = -filter_width/2; filter_r <= filter_width/2; ++filter_r) 
    {
      for (int filter_c = -filter_width/2; filter_c <= filter_width/2; ++filter_c) 
      {
        int image_r = min(max(row + filter_r, 0), static_cast<int>(numRows - 1));
        int image_c = min(max(col + filter_c, 0), static_cast<int>(numCols - 1));

        float image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);
        float filter_value = filter[(filter_r + filter_width/2) * filter_width + filter_c + filter_width/2];

        result += image_value * filter_value;
      }
    }
  outputChannel[row * numCols + col] = result;
}

__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
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

  outputImageRGBA[thread_idx] = outputPixel;
}

void make_filter(float **h_filter, int *filter_width, int kernel_width, float blur_diff)
{ 
  *h_filter = new float[kernel_width * kernel_width];
  *filter_width = kernel_width;
  float filterSum = 0.f; 

  for (int r = -kernel_width/2; r <= kernel_width/2; ++r)
   {
    for (int c = -kernel_width/2; c <= kernel_width/2; ++c) 
    {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * blur_diff * blur_diff));
      (*h_filter)[(r + kernel_width/2) * kernel_width + c + kernel_width/2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;
  for (int r = -kernel_width/2; r <= kernel_width/2; ++r) 
    for (int c = -kernel_width/2; c <= kernel_width/2; ++c) 
      (*h_filter)[(r + kernel_width/2) * kernel_width + c + kernel_width/2] *= normalizationFactor;
}
__global__
void separateChannels(const uchar4* const inputImageRGBA, int num_rows, int num_cols, unsigned char* const redChannel, unsigned char* const greenChannel, unsigned char* const blueChannel)
{  
  
  int loc_x = blockDim.x * blockIdx.x + threadIdx.x;
  int loc_y = blockDim.y * blockIdx.y + threadIdx.y;

  if ( loc_x >= num_cols ||
      loc_y >= num_rows )
  {
       return;
  }
  
  int thread_idx = loc_y * num_cols + loc_x;

  redChannel[thread_idx] = inputImageRGBA[thread_idx].x;
  greenChannel[thread_idx] = inputImageRGBA[thread_idx].y;
  blueChannel[thread_idx] = inputImageRGBA[thread_idx].z;
}

uchar4* blur(uchar4* d_in_img, size_t num_rows, size_t num_cols, int kernel_width)
{ 
  float blur_diff = kernel_width/4.0f;
  
  float* h_filter;
  int filter_width;
  make_filter(&h_filter, &filter_width, kernel_width, blur_diff);

  const dim3 blockSize(16,16,1);

  
  int a  = num_cols/blockSize.x;
  int b = num_rows/blockSize.y;	
  const dim3 gridSize(a + 1,b + 1,1);
  const size_t num_pix = num_rows * num_cols;

  uchar4 *d_out_img;
  cudaMalloc((void **)&d_out_img, sizeof(uchar4) * num_pix);
  
  d_input_img  = d_in_img;
  d_output_img = d_out_img;

  
  unsigned char *d_red_blur, *d_green_blur, *d_blue_blur;
  cudaMalloc(&d_red_blur,    sizeof(unsigned char) * num_pix);
  cudaMalloc(&d_green_blur,  sizeof(unsigned char) * num_pix);
  cudaMalloc(&d_blue_blur,   sizeof(unsigned char) * num_pix);
  
  cudaMalloc(&d_red, sizeof(unsigned char*) * num_rows * num_cols);
  cudaMalloc(&d_green, sizeof(unsigned char*) * num_rows * num_cols);
  cudaMalloc(&d_blue, sizeof(unsigned char*) * num_rows * num_cols);
  
  cudaMalloc(&d_filter, sizeof(float) * filter_width * filter_width);
  cudaMemcpy(d_filter, h_filter, sizeof(float) * filter_width * filter_width, cudaMemcpyHostToDevice); 

    separateChannels<<<gridSize, blockSize>>>(d_in_img, num_rows, num_cols, d_red,d_green, d_blue);

  cudaDeviceSynchronize(); 
  
    gaussian_blur<<<gridSize, blockSize>>>(d_red, d_red_blur, num_rows, num_cols,  d_filter, filter_width);
  gaussian_blur<<<gridSize, blockSize>>>(d_green, d_green_blur, num_rows, num_cols,  d_filter, filter_width);
  gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blue_blur, num_rows, num_cols,  d_filter, filter_width);

  cudaDeviceSynchronize(); 

    recombineChannels<<<gridSize, blockSize>>>(d_red_blur, d_green_blur, d_blue_blur, d_out_img, num_rows, num_cols);
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

    cudaMemcpy(h_out, d_out_img, sizeof(uchar4) * num_pix, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize(); 
  

 	return h_out;
}
