#ifndef FUNCTIONSB_H
#define FUNCTIONSB_H

#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../structures.h"
#include "../lbp/lbp.h"
#include "../svd/svd.h"

using namespace std;
using namespace cv;


// clip
__device__ float clip(float i, float a, float b){
  if(i < a)
    i = a;
  if(i > b)
    i = b;
  return i;
}


// hamming distance
__device__ int HammingDist(int x, int y)
{
  int dist = 0;
  char val = x^y;// calculate differ bit
  while(val)   //this dist veriable calculate set bit in loop
  {
    ++dist;
    val &= val - 1;
  }
  return dist;
}


// random numbers from curand states
__device__ int random(curandState_t* states, int h, int w, int i, int j,int min, int max) {
  curandState localState = at2d(states, i, j, w);

  // generar número pseudoaleatorio
  int ran = min + curand(&localState) % (max - min);;

  //copiar state de regreso a memoria global
  at2d(states, i, j, w) = localState;

  //almacenar resultados
  // result[ind] = r;

  return ran;
}


// initialization of random states gpu
__global__ void init_randoms(unsigned int seed, curandState_t* states, int h, int w) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if(row < h & col < w)
    curand_init(seed, row + col, 0, &states[row * w + col]);
}


// global initialization of model
__global__ void initialization_kernel(float* d_B_int, int* d_B_lsbp, float* d_int, float* d_svd, int* d_lbp, int h, int w, int S, curandState_t* states){
  int r = S / 2;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < h & j < w){


    int i0 = clip(i, r, h - r - 1);
    int j0 = clip(j, r, w - r - 1);
    at3d(d_B_int, i, j, 0, w, S) = at2d(d_int, i, j, w);
    at3d(d_B_lsbp, i, j, 0, w, S) = at2d(d_lbp, i, j, w);

    for(int k = 1; k < S; k++){
      int i1 = i0 + random(states, i, j, h, w, -r, r + 1);
      int j1 = j0 + random(states, i, j, h, w, -r, r + 1);
      at3d(d_B_int, i, j, k, w, S) = at2d(d_int, i1, j1, w);
      at3d(d_B_lsbp, i, j, k, w, S) = at2d(d_lbp, i1, j1, w);
    }

  }
}

__global__ void cuda_dmin(float* d_D, float* d_d, int h, int w, int S){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < h & j < w){
    at2d(d_d, i, j, w) = 0;
    for(int k = 0; k < S; k++){
      at2d(d_d, i, j, w) += at3d(d_D, i, j, k, w, S);
    }
    at2d(d_d, i, j, w) /= S;
  }
}


__global__ void cuda_update_R(float* d_R, float* d_d, int h, int w, int S, float Rscale, float Rlower, float Rlr){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < h & j < w){

    if(at2d(d_R, i, j, w) > at2d(d_d, i, j, w) * Rscale)
      at2d(d_R, i, j, w) = (1 - Rlr) * at2d(d_R, i, j, w);
    else
      at2d(d_R, i, j, w) = (1 + Rlr) * at2d(d_R, i, j, w);
    at2d(d_R, i, j, w) = clip(at2d(d_R, i, j, w), Rlower, 255);
  }
}


__global__ void cuda_update_T(float* d_T, bool* d_mask, float* d_d, int h, int w, int S, float Tinc, float Tdec, float Tlower, float Tupper){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < h & j < w){


    if(at2d(d_mask, i, j, w) == true){
      at2d(d_T, i, j, w) = at2d(d_T, i, j, w) + Tinc / at2d(d_d, i, j, w);
    }
    else{
      at2d(d_T, i, j, w) = at2d(d_T, i, j, w) - Tdec / at2d(d_d, i, j, w);
    }
    at2d(d_T, i, j, w) = clip(at2d(d_T, i, j, w), Tlower, Tupper);

  }
}

__global__ void cuda_update_models(float* d_B_int, int* d_B_lsbp, float* d_D, float* d_R, float* d_T, float* d_int, int* d_lbp, bool* d_mask, float* d_d, int h, int w, int S, curandState_t* states){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < h & j < w){
    if(at2d(d_mask, i, j, w) == false){
      if(random(states, i, j, h, w, 0, 100) < (1 / at2d(d_T, i, j, w))){
        int p = random(states, i, j, h, w, 0, S);
        int min = fabs(at3d(d_B_int, i, j, 0, w, S) - at2d(d_int, i, j, w));
        for(int k = 1; k < S; k++){
          float temp = fabs(at3d(d_B_int, i, j, k, w, S) - at2d(d_int, i, j, w));
          if(temp < min && p != k)
            min = temp;
        }
        at3d(d_B_int, i, j, p, w, S) = at2d(d_int, i, j, w);
        at3d(d_B_lsbp, i, j, p, w, S) = at2d(d_lbp, i, j, w);
        at3d(d_D, i, j, p, w, S) = min;
      }
      if(random(states, i, j, h, w, 0, 100) < (1 / at2d(d_T, i, j, w))){
        int i0 = clip(i + random(states, i, j, h, w, -1, 2), 0, h - 1);
        int j0 = clip(j + random(states, i, j, h, w, -1, 2), 0, w - 1);
        int p = random(states, i, j, h, w, 0, S);

        int min = fabs(at3d(d_B_int, i0, j0, 0, w, S) - at2d(d_T, i0, j0, w));
        for(int k = 1; k < S; k ++){
          float temp = fabs(at3d(d_B_int, i0, j0, k, w, S) - at2d(d_int, i0, j0, w));
          if(temp < min && p != k)
            min = temp;
        }
        at3d(d_B_int, i0, j0, p, w, S) = at2d(d_int, i0, j0, w);
        at3d(d_B_lsbp, i0, j0, p, w, S) = at2d(d_lbp, i0, j0, w);
        at3d(d_D, i0, j0, p, w, S) = min;
      }
    }




  }
}


__global__ void cuda_step(float* d_B_int, int* d_B_lsbp, float* d_int, float* d_svd, int* d_lbp, bool* d_mask, int h, int w, int S, int threshold, int HR, float* d_R){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < h & j < w){

    int count = 0;
    for(int k = 0; k < S && count < threshold; k++){
      if(fabs(at2d(d_int, i, j, w) - at3d(d_B_int, i, j, k, w, S)) < at2d(d_R, i, j, w) && HammingDist(at2d(d_lbp, i, j, w), at3d(d_B_lsbp,i, j, k, w, S)) < HR ){
        count++;
      }
    }
    if(count < threshold){
      at2d(d_mask, i, j, w) = true;
    }
    else{
      at2d(d_mask, i, j, w) = false;
    }
  }
}

__global__ void printV(float* v, int h, int w){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < h & j < w){
    printf("%f ",v[i * w + j] );
  }
}

__global__ void printVi(int* v, int h, int w){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < h & j < w){
    printf("%d ",v[i * w + j] );
  }
}

#endif
