#include "BackgroundSubstractor.h"


BackgroundSubstractor::BackgroundSubstractor(){
  vector<string> files = globVector("input/*");
  sort(files.begin(), files.end());
  for (int i = 0 ; i < files.size(); i++){
    cout<<files[i]<<endl;
    Frame f(files[i]);
    frames.push_back(f);
  }

  h = frames[0].mat.rows;
  w = frames[0].mat.cols;


  // Inicializacion de arrays

  B_int = new float**[h];
  B_lsbp = new int**[h];
  D = new float**[h];
  d = new float*[h];
  R = new float*[h];
  T = new float*[h];

  for(int i = 0; i < h; i++){
    B_int[i] = new float*[w];
    B_lsbp[i] = new int*[w];
    D[i] = new float*[w];
    d[i] = new float[w];
    R[i] = new float[w];
    T[i] = new float[w];
    for(int j = 0; j < w; j++){
      B_int[i][j] = new float[S];
      B_lsbp[i][j] = new int[S];
      D[i][j] = new float[S];
      d[i][j] = 0.1;
      R[i][j] = 18;
      T[i][j] = 0.05;
      for(int k = 0; k < S; k ++){
        B_int[i][j][k] = 0;
        B_lsbp[i][j][k] = 0;
        D[i][j][k] = 0.2;
      }
    }
  }





  dmin();


}


void BackgroundSubstractor::init(){
  cout<<"Init"<<endl;

  int r = S / 2;
  float** intensidad = frames[0].intensidad;
  float** SVD = svd(intensidad, h, w);
  int** lbp = lsbp(SVD, h, w);

  for(int i = 0; i < h; i++){
    for(int j = 0; j < w; j++){
      int i0 = clip(i, r, h - r - 1);
      int j0 = clip(j, r, w - r - 1);
      B_int[i][j][0] = intensidad[i][j];
      B_lsbp[i][j][0] = lbp[i][j];
      for(int k = 1; k < S; k++){
        int i1 = i0 + random(-r, r + 1);
        int j1 = j0 + random(-r, r + 1);
        B_int[i][j][k] = intensidad[i1][j1];
        B_lsbp[i][j][k] = lbp[i1][j1];
      }
    }
  }
  cout<<"End init"<<endl;
}


void BackgroundSubstractor::step(int pos){
  float** intensidad = frames[pos].intensidad;
  float** SVD = svd(intensidad, h, w);
  int** lbp = lsbp(SVD, h, w);

  bool** mask = new bool*[h];
  for(int i = 0; i < h; i++){
    mask[i] = new bool[w];
    for(int j = 0; j < w; j++){
      int count = 0;
      for(int k = 0; k < S; k++){
        if(fabs(intensidad[i][j] - B_int[i][j][k]) < R[i][j] && HammingDist(lbp[i][j], B_lsbp[i][j][k]) < HR){
          count++;
        }
      }
      if(count < threshold){
        frames[pos].mat.at<uchar>(i, j) = 255;
        mask[i][j] = true;
      }
      else{
        frames[pos].mat.at<uchar>(i, j) = 0;
        mask[i][j] = false;
      }
    }
  }

  update_R();
  update_T(mask);
  update_models(mask, intensidad, lbp);
  cout<<pos<<endl;
  imwrite("output/" + to_string(pos) + ".jpg" , frames[pos].mat);
}


void BackgroundSubstractor::dmin(){
  for(int i = 0; i < h; i++){
    for(int j = 0; j < w; j++){
      d[i][j] = 0;
      for(int k = 0; k < S; k++){
        d[i][j] += D[i][j][k];
      }
      d[i][j] /= S;
    }
  }
}

void BackgroundSubstractor::update_R(){
  for(int i = 0; i < h; i++){
    for(int j = 0; j < w; j++){
      if(R[i][j] > d[i][j] * Rscale)
        R[i][j] = (1 - Rlr) * R[i][j];
      else
        R[i][j] = (1 + Rlr) * R[i][j];
      R[i][j] = clip(R[i][j], Rlower, 255);
    }
  }
}


void BackgroundSubstractor::update_T(bool** mask){
  for(int i = 0; i < h; i++){
    for(int j = 0; j < w; j++){
      if(mask[i][j] == true){
        T[i][j] = T[i][j] + Tinc / d[i][j];
      }
      else{
        T[i][j] = T[i][j] - Tdec / d[i][j];
      }
      T[i][j] = clip(T[i][j], Tlower, Tupper);
    }
  }
}


void BackgroundSubstractor::update_models(bool** mask, float** intensidad, int** lbp){
  for(int i = 0; i < h; i++){
    for(int j = 0; j < w; j++){
      if(mask[i][j] == false){
        if(random(0, 100) < (1 / T[i][j])){
          // cout<<"Actualizacion modelo"<<endl;
          int p = random(0, S);
          int min = fabs(B_int[i][j][0] - intensidad[i][j]);
          for(int k = 1; k < S; k++){
            float temp = fabs(B_int[i][j][k] - intensidad[i][j]);
            if(temp < min && p != k)
              min = temp;
          }
          B_int[i][j][p] = intensidad[i][j];
          B_lsbp[i][j][p] = lbp[i][j];
          D[i][j][p] = min;
        }
        if(random(0, 100) < (1 / T[i][j])){
          int i0 = clip(i + random(-1, 2), 0, h - 1);
          int j0 = clip(j + random(-1, 2), 0, w - 1);
          int p = random(0, S);

          int min = fabs(B_int[i0][j0][0] - intensidad[i0][j0]);
          for(int k = 1; k < S; k++){
            float temp = fabs(B_int[i0][j0][k] - intensidad[i0][j0]);
            if(temp < min && p != k)
              min = temp;
          }
          B_int[i0][j0][p] = intensidad[i0][j0];
          B_lsbp[i0][j0][p] = lbp[i0][j0];
          D[i0][j0][p] = min;
        }
      }
    }
  }
}




















// fin
