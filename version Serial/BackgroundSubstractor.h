#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "Frame.cpp"
#include "funciones.h"
#include "string"

using namespace cv;

class BackgroundSubstractor{
public:
  BackgroundSubstractor();
  ~BackgroundSubstractor(){}

  void init();
  void step(int);
  void update_R();
  void update_T(bool**);
  void dmin();
  void update_models(bool**, float**, int**);

  vector<Frame> frames;
  int S = 35;
  int h, w;
  float*** B_int;
  int*** B_lsbp;
  float*** D;
  float** d;
  float** R;
  float** T;
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

};
