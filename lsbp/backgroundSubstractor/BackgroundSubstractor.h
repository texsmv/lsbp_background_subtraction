#ifndef BACKGROUNDSUBSTRACTOR_H
#define BACKGROUNDSUBSTRACTOR_H

#include "functionsB.h"

class BackgroundSubstractor{
public:
  BackgroundSubstractor();
  ~BackgroundSubstractor();

  void initialize(int h, int w);
  void initialize_model();
  void step();
  void update_R();
  void update_T();
  void dmin();
  void update_models();

  void set_intensities(vector<float*>*);
  void set_frames(vector<Mat>*);

  //new functions
  void process_image();
  void save_image();

  vector<Mat>* frames;
  vector<float*>* frames_int;
  int frame_pos = 0;

  dim3 block;
  dim3 grid;

  int s = 20;
  int h, w;

  // 3d
  float* h_B_int;
  int* h_B_lsbp;
  float* h_D;

  float* d_B_int;
  int* d_B_lsbp;
  float* d_D;

  //2d
  float* h_d;
  float* h_R;
  float* h_T;


  float* d_d;
  float* d_R;
  float* d_T;


  int threshold = 2;
  int HR = 4;
  float Rlr = 0.05;
  float Tlr = 0.02;
  float Tinc = 1;
  float Tdec = 0.05;
  float Rscale = 5;
  float Tlower = 0.05;
  float Tupper = 0.8;
  float Rlower = 18;

  //2d arrays
  float* h_int;
  float* h_svd;
  int* h_lbp;
  bool* h_mask;

  float* d_int;
  float* d_svd;
  int* d_lbp;
  bool* d_mask;

  // for random numbers
  curandState_t* states;


  int cont = 0;
};


BackgroundSubstractor::BackgroundSubstractor(){

}

BackgroundSubstractor::~BackgroundSubstractor(){
  delete(h_B_int);
  delete(h_B_lsbp);
  delete(h_D);
  delete(h_d);
  delete(h_R);
  delete(h_T);
  delete(h_svd);

  cudaFree(d_B_int);
  cudaFree(d_B_lsbp);
  cudaFree(d_D);
  cudaFree(d_d);
  cudaFree(d_R);
  cudaFree(d_T);
  cudaFree(d_svd);
}


void BackgroundSubstractor::initialize(int h, int w){
  printf("---------Initializing-----------\n");
  printf("Height : %d, Width : %d\n", h, w);

  this->h = h;
  this->w = w;

  block = dim3(16, 16, 1);
  grid = dim3(ceil(h / float(block.x)), ceil(w / float(block.y)));

  printf("Block dimension: %d - %d - %d \n", block.x, block.y, block.z);
  printf("Grid dimension: %d - %d - %d \n", grid.x, grid.y, grid.z);


  printf("Creating host arrays\n");
  // Inicializacion de arrays

  h_B_int = new float[h * w * s];
  h_B_lsbp = new int[h * w * s];
  h_D = new float[h * w * s];

  h_d = new float[h * w];
  h_R = new float[h * w];
  h_T = new float[h * w];

  h_mask = new bool[h * w];
  h_lbp = new int[h * w];
  h_svd = new float[h * w];


  for(int i = 0; i < h; i ++){
    for(int j = 0; j < w; j ++){
      at2d(h_d, i, j, w) = 0.1;
      at2d(h_R, i, j, w) = 18;
      at2d(h_T, i, j, w) = 0.05;

      at2d(h_mask, i, j, w) = false;
      at2d(h_lbp, i, j, w) = 0;
      at2d(h_svd, i, j, w) = 8;

      for(int k = 0; k < s; k ++){
        at3d(h_B_int, i, j, k, w, s) = 1;
        at3d(h_B_lsbp, i, j, k, w, s) = 0;
        at3d(h_D, i, j, k, w, s) = 0.2;
      }
    }
  }

  printf("Creating device arrays\n");

  d_B_int = cuda_array<float>(h * w * s);
  cuda_H2D<float>(h_B_int, d_B_int, h * w * s);

  d_B_lsbp = cuda_array<int>(h * w * s);
  cuda_H2D<int>(h_B_lsbp, d_B_lsbp, h * w * s);

  d_D = cuda_array<float>(h * w * s);
  cuda_H2D<float>(h_D, d_D, h * w * s);


  d_d = cuda_array<float>(h * w);
  cuda_H2D<float>(h_d, d_d, h * w);

  d_R = cuda_array<float>(h * w);
  cuda_H2D<float>(h_R, d_R, h * w);

  d_T = cuda_array<float>(h * w);
  cuda_H2D<float>(h_T, d_T, h * w);

  d_svd = cuda_array<float>(h * w);
  cuda_H2D<float>(h_svd, d_svd, h * w);

  // initialize arrays for svd, lbp, e intensity matrix on __device__
  d_int = cuda_array<float>(h * w);
  d_lbp = cuda_array<int>(h * w);
  d_mask = cuda_array<bool>(h * w);


  printf("Creating curand states\n");
  states = cuda_array<curandState_t>(h * w);


  printf("Initializing curand states\n");
  init_randoms<<<grid, block>>>(time(0), states, h, w);

  cuda_dmin<<<grid, block>>>(d_D, d_d, h, w, s);

  cudaDeviceSynchronize();

  printf("---------End of Initialization-----------\n");

}


void BackgroundSubstractor::set_intensities(vector<float*>* v){
  frames_int = v;
}

void BackgroundSubstractor::set_frames(vector<Mat>* v){
  frames = v;
}

void BackgroundSubstractor::process_image(){
  if(frame_pos >= frames_int->size()){
    frame_pos = 0;
  }
  printf("processing  : %d", frame_pos);

  h_int = frames_int->at(frame_pos);
  cuda_H2D<float>(h_int, d_int, h * w);


  cuda_svd(d_int, d_svd, h, w, block, grid);

  CHECK(cudaDeviceSynchronize());

  cuda_lsbp(d_svd, d_lbp, h, w, block, grid);

  cudaDeviceSynchronize();
  printf("    -   done\n" );
}


void BackgroundSubstractor::initialize_model(){
  process_image();
  frame_pos = 0;
  initialization_kernel<<<grid, block>>>(d_B_int, d_B_lsbp, d_int, d_svd, d_lbp, h, w, s, states);

}

void BackgroundSubstractor::step(){
  process_image();


  // printf("d_int: \n");
  // printV<<<grid,block>>>(d_int, 10, 10);
  // printf("\n");

  // printf("d_lbp: \n");
  // printV<<<grid,block>>>(d_B_int, 3, 3);
  // printf("\n");

  cuda_step<<<grid, block>>>(d_B_int, d_B_lsbp, d_int, d_svd, d_lbp, d_mask, h, w, s, threshold, HR, d_R);

  cudaDeviceSynchronize();
  cuda_D2H(d_mask, h_mask, h * w);

  save_image();

  cuda_update_R<<<grid, block>>>(d_R, d_d, h, w, s, Rscale, Rlower, Rlr);

  cuda_update_T<<<grid, block>>>(d_T, d_mask, d_d, h, w, s, Tinc, Tdec, Tlower, Tupper);

  cuda_update_models<<<grid, block>>>(d_B_int, d_B_lsbp, d_D, d_R, d_T, d_int, d_lbp, d_mask, d_d, h, w, s, states);

  cuda_dmin<<<grid, block>>>(d_D, d_d, h, w, s);


  frame_pos++;
  cont++;
}

void BackgroundSubstractor::save_image(){
  Mat mat = frames->at(frame_pos);

  uchar *ptrDst[mat.rows];
  for(int i = 0; i < mat.rows; ++i) {
      ptrDst[i] = mat.ptr<uchar>(i);
      for(int j = 0; j < mat.cols; ++j) {
          if(at2d(h_mask, i, j, w) == true)
            ptrDst[i][j] = 255;
          else
            ptrDst[i][j] = 0;
      }
  }
  imwrite("data/output/" + to_string(cont) + ".png" , mat);
}

#endif
