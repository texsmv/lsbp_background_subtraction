#ifndef FUNCTIONSV_H
#define FUNCTIONSV_H

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


#include "../structures.h"

using namespace std;
using namespace cv;




/*
It saves a vector of frames into jpg images into the outputDir as 1.jpg,2.jpg etc where 1,2 etc represents the frame number
*/
void save_frames(vector<Mat>& frames, const string& outputDir){
  vector<int> compression_params;
  compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);
  int frameNumber;
  vector<Mat>::iterator frame;
	for(frame = frames.begin(), frameNumber=0; frame != frames.end(); ++frame, frameNumber++){
	  string filePath = outputDir + to_string(frameNumber) + ".jpg";
	  imwrite(filePath,*frame,compression_params);
	}
}

#endif
