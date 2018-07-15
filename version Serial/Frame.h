#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
// #include "Frame.cpp"

using namespace cv;
using namespace std;

class Frame{
public:
  Frame(){}
  Frame(string);
  Frame(int){}
  ~Frame(){}


  float ** intensidad;
  Mat mat;
};
