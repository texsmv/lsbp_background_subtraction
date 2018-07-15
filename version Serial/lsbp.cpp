#include "BackgroundSubstractor.cpp"

int main(int argc, char const *argv[]) {
  BackgroundSubstractor b;
  b.init();
  for(int i = 0; i < 100; i++)
    b.step(i);
  //
  // int h, w, S;
  // h = 10;
  // w = 10;
  // S = 4;
  // int*** a;
  // a = new int**[h];
  // for(int i = 0; i < w; i++){
  //   a[i] = new int*[w];
  // }
  // a[0][0] = new int[h * w * S];
  // int* temp = a[0][0];
  // for(int i = 0; i < h; i++){
  //   for(int j = 0; j < w; j++){
  //     a[i][j] = &temp[i * (w * S) + j * S];
  //   }
  // }
  //
  // for(int i = 0; i < h; i++){
  //   for(int j = 0; j < w; j++){
  //     for(int k = 0; k < S; k++){
  //       a[i][j][k] = 0;
  //     }
  //   }
  // }



}
