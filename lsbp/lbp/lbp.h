#include "../structures.h"



// ............................LSBP..........................
// lbp por pixel - device
__device__ int lbp_pixel(float* mat, int h, int w, int i2, int j2){
  float tau = 0.05;
  float svd_pixel = at2d(mat, i2, j2, w);
  int sum = 0;
  int num_neighbor = 0;
  for (int i = i2-1; i <= i2+1; i++)
  {
    for (int j = j2-1; j <= j2+1; j++)
    {
      float svd_neighbor = 0;
      if (i >= 0 && i < h && j>=0 && j<w)
        svd_neighbor = at2d(mat, i, j, w);
      if (fabs(svd_neighbor-svd_pixel) < tau)
        sum += pow(2, num_neighbor);
      num_neighbor++;
    }
  }
  return sum;
}

// lsbp kernel
__global__ void lsbp_kernel(float* d_mat, int* d_lbp, int h, int w){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if(row < h && col < w){
    at2d(d_lbp, row, col, w) = lbp_pixel(d_mat, h, w, row, col);
  }

}

// function to call kernel
void cuda_lsbp(float* d_mat, int* d_lbp, int h, int w, dim3 block, dim3 grid){
  lsbp_kernel<<<grid, block>>>(d_mat, d_lbp, h, w);
}
