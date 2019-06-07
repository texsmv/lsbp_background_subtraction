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
  vector<Mat> real_frames;
  vector<float*> frames_int;
  vector<float*> frames_green;
  vector<float*> frames_blue;
  vector<float*> frames_red;
  vector<float*> frames_cb;
  vector<float*> frames_cr;
  VideoCapture* cap;
  unsigned int num_frames;
  unsigned int actual_frame;
  bool b_resize = false;
  int cont = 0;
  friend class Controller;
  friend class BackgroundSubstractor;
};




Video::Video(){
  erase_frames();
}

Video::Video(string path){
  this->path = path;
  cap = new VideoCapture(0);
  // cap = new VideoCapture(path);
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

    resize(frame, frame, Size(r_width, r_height));

    Mat color = frame.clone();

    Mat color_ycbcr = frame.clone();

    cvtColor(color_ycbcr, color_ycbcr, CV_BGR2YCrCb);
    // cvtColor(color_ycbcr, color_ycbcr, CV_BGR2HSV);

    real_frames.push_back(color);

    cvtColor(frame, frame, CV_BGR2GRAY);
    frames.push_back(frame);

    float* intensidad = new float[frame.rows * frame.cols];
    float* green = new float[frame.rows * frame.cols];
    float* red = new float[frame.rows * frame.cols];
    float* blue = new float[frame.rows * frame.cols];
    float* cb = new float[frame.rows * frame.cols];
    float* cr = new float[frame.rows * frame.cols];

    // printf("%d %d \n", frame.rows, frame.cols);
    uchar *ptrDst[frame.rows];
    Vec3b *ptrColor[frame.rows];
    Vec3b *ptrYCbCr[frame.rows];


    for(int i = 0; i < frame.rows; ++i) {
      ptrDst[i] = frame.ptr<uchar>(i);
      ptrColor[i] = color.ptr<Vec3b>(i);
      ptrYCbCr[i] = color_ycbcr.ptr<Vec3b>(i);
      for(int j = 0; j < frame.cols; ++j) {
        at2d(intensidad, i, j, frame.cols) = ptrDst[i][j];

        at2d(red, i, j, frame.cols) = ptrColor[i][j][0];
        at2d(green, i, j, frame.cols) = ptrColor[i][j][1];
        at2d(blue, i, j, frame.cols) = ptrColor[i][j][2];

        at2d(cr, i, j, frame.cols) = ptrYCbCr[i][j][1];
        at2d(cb, i, j, frame.cols) = ptrYCbCr[i][j][2];
      }
    }
    //
    frames_int.push_back(intensidad);
    frames_red.push_back(red);
    frames_blue.push_back(blue);
    frames_green.push_back(green);
    frames_cb.push_back(cb);
    frames_cr.push_back(cr);

    cont++;

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
    delete(frames_int[i]);
    delete(frames_red[i]);
    delete(frames_green[i]);
    delete(frames_blue[i]);
    delete(frames_cb[i]);
    delete(frames_cr[i]);
  }
  frames.clear();
  real_frames.clear();
  frames_int.clear();
  frames_red.clear();
  frames_blue.clear();
  frames_green.clear();
  frames_cb.clear();
  frames_cr.clear();
}


void Video::set_size(unsigned int h,unsigned int w){
  b_resize = true;
  r_height = h;
  r_width = w;
}


#endif
