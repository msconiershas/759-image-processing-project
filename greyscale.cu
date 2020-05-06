#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <math.h>
using namespace std;

__global__ void grey(const uchar4* input, uchar4* output, size_t num_rows, size_t num_cols) {

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  int idx = y * num_cols + x;

  if(x < num_cols && y < num_rows) {
    unsigned char val = 0.299 * input[idx].x + 0.587 * input[idx].y + 0.114 * input[idx].z;
    output[idx] = make_uchar4(val, val, val, 255);
  }

}

uchar4* greyscale(uchar4 *d_in, const size_t num_rows, const size_t num_cols)
{
  uchar4 *d_out;
  cudaMalloc((void **) &d_out, num_rows * num_cols * sizeof(uchar4));

  const dim3 block_size(16, 16, 1);
  const dim3 grid_size(num_cols/16 + 1, num_rows/16 + 1, 1);

  grey<<<grid_size, block_size>>>(d_in, d_out, num_rows, num_cols);

  uchar4* h_out = new uchar4[num_rows * num_cols];
  cudaMemcpy(h_out, d_out, num_rows * num_cols * sizeof(uchar4), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  return h_out;
}
