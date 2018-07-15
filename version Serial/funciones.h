#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <algorithm>
#include <glob.h>
#include <random>


using namespace std;



//  - -------------------------------------- SVD

float svd_pixel(float** intensidad, int h, int w, int i2, int j2){
  Mat m = Mat::ones(3, 3, CV_32F);
  for(int i = -1; i < 2; i ++){
    for(int j = -1; j < 2; j ++){
        int row = i + i2;
        int col = j + j2;
        if(row < 0 || row >= h || col < 0 || col >= w)
            m.at<float>(i + 1, j + 1) = 0;
        else
            m.at<float>(i + 1, j + 1) = intensidad[row][col];
    }
  }
  Mat W, u, vt;
  SVD::compute(m, W, u, vt);
  return (W.at<float>(1) + W.at<float>(1)) / W.at<float>(0);
}



float** svd(float** intensidad, int h, int w){
  float** g = new float*[h];
  for(int i = 0; i < h; i++){
    g[i] = new float[w];
    for(int j =0; j < w; j++){
      g[i][j] = svd_pixel(intensidad, h, w, i, j);
    }
  }
  return g;
}

//  - -------------------------------------- LSBP

int lbp_pixel(float** mat, int h, int w, int i2, int j2){
  float tau = 0.05;
  float svd_pixel = mat[i2][j2];
  int sum = 0;
  int num_neighbor = 0;
  for (int i = i2-1; i <= i2+1; i++)
  {
    for (int j = j2-1; j <= j2+1; j++)
    {
      float svd_neighbor = 0;
      if (i >= 0 && i < h && j>=0 && j<w)
        svd_neighbor = mat[i][j];
      if (fabs(svd_neighbor-svd_pixel) < tau)
        sum += pow(2, num_neighbor);
      num_neighbor++;
    }
  }
  return sum;
}

int** lsbp(float** mat, int h, int w){
  int** lbp = new int*[h];
  for(int i = 0; i < h; i++){
    lbp[i] = new int[w];
    for(int j = 0; j < w; j++){
      lbp[i][j] = lbp_pixel(mat, h, w, i , j);
    }
  }
  return lbp;
}


// Get random number given a range

int random(int min, int max) //range : [min, max)
{
   static bool first = true;
   if (first)
   {
      srand( time(NULL) ); //seeding for the first time only!
      first = false;
   }
   return min + rand() % (( max) - min);
}

// listing a directory

vector<string> globVector(const string& pattern){
    glob_t glob_result;
    glob(pattern.c_str(),GLOB_TILDE,NULL,&glob_result);
    vector<string> files;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        files.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}

// Hamming distance

int HammingDist(int x, int y)
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


// clip from python

float clip(float i, float a, float b){
  if(i < a)
    i = a;
  if(i > b)
    i = b;
  return i;
}
