#include "Frame.h"

Frame::Frame(string path){
  Mat m = imread(path.c_str());

  // Convert to double (much faster than a simple for loop)
  cvtColor(m, mat, CV_BGR2GRAY);



  uchar *ptrDst[mat.rows];

  intensidad = new float*[mat.rows];

  for(int i = 0; i < mat.rows; ++i) {
      ptrDst[i] = mat.ptr<uchar>(i);
      intensidad[i] = new float[mat.cols];
      for(int j = 0; j < mat.cols; ++j) {
          intensidad[i][j] = ptrDst[i][j];
      }
  }
}
