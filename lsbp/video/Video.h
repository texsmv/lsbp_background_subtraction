#ifndef VIDEO_H
#define VIDEO_H

#include "functionsV.h"


class Video{
public:
  Video();
  Video(string);
  ~Video();
  void capture();
  void capture_batch(unsigned int);
  void erase_frames();
  void set_size(unsigned int,unsigned int);
private:
  string path;
  unsigned int height, width, r_height, r_width;
  vector<Mat> frames;
  vector<float*> frames_int;
  VideoCapture* cap;
  unsigned int num_frames;
  unsigned int actual_frame;
  bool b_resize = false;

  friend class Controller;
};




Video::Video(){
  erase_frames();
}

Video::Video(string path){
  this->path = path;
  cap = new VideoCapture(path);
  if(!cap->isOpened())
    CV_Error(CV_StsError, "Can not open Video file");
  num_frames = cap->get(CV_CAP_PROP_FRAME_COUNT);
  actual_frame = 0;
}

Video::~Video(){

}





void Video::capture_batch(unsigned int batch_size){

  unsigned int l_frame = actual_frame + batch_size;
  // try{
    Mat* p_frame;
    for(; actual_frame < l_frame && actual_frame < num_frames; actual_frame++){
      // printf("Frame %d \n", actual_frame);

      Mat frame;
      // grey = new Mat();
      (*cap) >> frame;
      // imwrite("../../data/input/" + to_string(actual_frame) + ".png", frame);
      // Mat frame = imread("../../data/input/" + to_string(actual_frame) + ".png", 0);

      p_frame = &frame;
      // if(b_resize)
      resize(frame, frame, Size(500, 500));
      cvtColor(frame, frame, CV_BGR2GRAY);
      frames.push_back(frame);

      float* intensidad = new float[frame.rows * frame.cols];
      //
      // printf("%d %d \n", frame.rows, frame.cols);
      uchar *ptrDst[frame.rows];
      for(int i = 0; i < frame.rows; ++i) {
        ptrDst[i] = frame.ptr<uchar>(i);
        for(int j = 0; j < frame.cols; ++j) {
          at2d(intensidad, i, j, frame.rows) = ptrDst[i][j];
        }
      }
      //
      frames_int.push_back(intensidad);

    }
    height = p_frame->rows;
    width = p_frame->cols;
  // }
  // catch( cv::Exception& e ){
  //   cerr << e.msg << endl;
  //   exit(1);
  // }
}

void Video::erase_frames(){
  for(int i = 0; i < frames.size(); i++){
    frames[i].release();
  }
  frames.clear();
  frames_int.clear();
}


void Video::set_size(unsigned int h,unsigned int w){
  b_resize = true;
  r_height = h;
  r_width = w;
}


#endif
