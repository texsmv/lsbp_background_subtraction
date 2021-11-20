#ifndef SCRIPTS_H
#define SCRIPTS_H


#include <string>
#include <vector>
#include <algorithm>
#include <glob.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/opencv.hpp"


using namespace std;
using namespace cv;



void video_to_images_resize(string path, int n, int h, int w){

  VideoCapture* cap = new VideoCapture(path);
  // if(!cap->isOpened())
  //   CV_Error(cv::StsError, "Can not open Video file");
  int num_frames = cap->get(CV_CAP_PROP_FRAME_COUNT);
  if(n != 0){
    num_frames = n;
  }

  for(int actual_frame = 0; actual_frame <  num_frames; actual_frame++){
    printf("%d \n", actual_frame);
    Mat frame;
    (*cap) >> frame;

    resize(frame, frame, Size(h, w));

    imwrite("../../data/input/" + to_string(actual_frame) + ".png" , frame);

  }
}


void video_to_images(string path, int n){

  VideoCapture* cap = new VideoCapture(path);
  // if(!cap->isOpened())
  //   CV_Error(cv::StsError, "Can not open Video file");
  int num_frames = cap->get(CV_CAP_PROP_FRAME_COUNT);
  if(n != 0){
    num_frames = n;
  }

  for(int actual_frame = 0; actual_frame <  num_frames; actual_frame++){
    printf("%d \n", actual_frame);
    Mat frame;
    (*cap) >> frame;


    imwrite("../../data/input/" + to_string(actual_frame) + ".png" , frame);

  }
}


#endif
